# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host destination geometry layered on Mooncake's source endpoint plan."""

from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from math import gcd

import numpy as np

MAX_REGISTER_MEMORY_BYTES = 64 * 1024**3
MAX_CACHED_DSA_MAPPINGS = 4
MAX_VECTORIZED_DSA_SEGMENTS = 65536
_MAX_INT64 = np.iinfo(np.int64).max


@dataclass(frozen=True)
class DsaCacheLayout:
    layer_name: str
    position: int
    base: int
    block_bytes: int
    stride: int
    scale: int
    block_tokens: int
    dtype: str
    capacity: int = 0


@dataclass(frozen=True)
class DsaRegisterAtom:
    """One transfer component that must not straddle registered regions."""

    start: int
    end: int
    location: str
    allocation: Hashable


@dataclass(frozen=True)
class DsaRegisterRegions:
    ptrs: list[int]
    lengths: list[int]
    locations: list[str]


def layout_span_bytes(layout: DsaCacheLayout) -> int:
    """Return the address span touched by all physical pages in a layout."""
    physical_pages = layout.capacity * layout.scale
    if physical_pages <= 0 or layout.block_bytes <= 0 or layout.stride < layout.block_bytes:
        raise ValueError("invalid DSA registration layout")
    return (physical_pages - 1) * layout.stride + layout.block_bytes


def collect_bounded_register_regions(
    atoms: list[DsaRegisterAtom],
    *,
    max_region_bytes: int = MAX_REGISTER_MEMORY_BYTES,
) -> DsaRegisterRegions:
    """Merge atoms per allocation without cutting an atom at a region edge.

    An atom larger than Mooncake's per-call limit is registered in deterministic
    chunks anchored at the atom base. Transfer entries use the same anchor.
    """
    if max_region_bytes <= 0:
        raise ValueError("max_region_bytes must be positive")
    grouped: OrderedDict[tuple[Hashable, str], list[DsaRegisterAtom]] = OrderedDict()
    for atom in atoms:
        if atom.start < 0 or atom.end <= atom.start:
            raise ValueError("invalid DSA register atom")
        grouped.setdefault((atom.allocation, atom.location), []).append(atom)

    ptrs: list[int] = []
    lengths: list[int] = []
    locations: list[str] = []

    def append_region(start: int, end: int, location: str) -> None:
        ptrs.append(start)
        lengths.append(end - start)
        locations.append(location)

    for (_, location), allocation_atoms in grouped.items():
        allocation_atoms.sort(key=lambda atom: (atom.start, atom.end))
        # Shared layers can expose the exact same tensor more than once. Drop
        # exact aliases first, including oversized atoms whose chunks would
        # otherwise be registered twice. Partially overlapping atoms are safe
        # only when their whole union fits in one registered region.
        unique_atoms: list[DsaRegisterAtom] = []
        for atom in allocation_atoms:
            if unique_atoms and (atom.start, atom.end) == (unique_atoms[-1].start, unique_atoms[-1].end):
                continue
            unique_atoms.append(atom)
        overlap_clusters: list[list[DsaRegisterAtom]] = []
        for atom in unique_atoms:
            if not overlap_clusters or atom.start >= max(item.end for item in overlap_clusters[-1]):
                overlap_clusters.append([atom])
            else:
                overlap_clusters[-1].append(atom)

        normalized_atoms: list[DsaRegisterAtom] = []
        for cluster in overlap_clusters:
            cluster_start = cluster[0].start
            cluster_end = max(atom.end for atom in cluster)
            if cluster_end - cluster_start > max_region_bytes:
                for atom in cluster:
                    offset = atom.start - cluster_start
                    atom_size = atom.end - atom.start
                    boundary_aligned = atom_size > max_region_bytes and offset % max_region_bytes == 0
                    within_one_chunk = atom_size <= max_region_bytes and (
                        offset // max_region_bytes == (offset + atom_size - 1) // max_region_bytes
                    )
                    if not (boundary_aligned or within_one_chunk):
                        raise ValueError("overlapping DSA register atoms cross a registration boundary")
            normalized_atoms.append(
                DsaRegisterAtom(
                    cluster_start,
                    cluster_end,
                    location,
                    cluster[0].allocation,
                )
            )

        current_start: int | None = None
        current_end = 0
        for atom in normalized_atoms:
            atom_size = atom.end - atom.start
            if atom_size > max_region_bytes:
                if current_start is not None:
                    append_region(current_start, current_end, location)
                    current_start = None
                chunk_start = atom.start
                while chunk_start < atom.end:
                    chunk_end = min(chunk_start + max_region_bytes, atom.end)
                    append_region(chunk_start, chunk_end, location)
                    chunk_start = chunk_end
                continue
            if current_start is None:
                current_start, current_end = atom.start, atom.end
            elif atom.end - current_start <= max_region_bytes:
                # The allocation key proves that any alignment gap is backed
                # by the same allocation and can safely be registered.
                current_end = atom.end
            else:
                append_region(current_start, current_end, location)
                current_start, current_end = atom.start, atom.end
        if current_start is not None:
            append_region(current_start, current_end, location)
    return DsaRegisterRegions(ptrs, lengths, locations)


def _build_component_read_scalar(
    local: DsaCacheLayout,
    remote: DsaCacheLayout,
    source_ids: tuple[int, ...],
    destination_ids: tuple[int, ...],
    start_token: int,
    end_token: int,
    *,
    cp_size: int,
    cp_rank: int,
    writer_rank: int,
    writer_size: int,
    indexer: bool,
    statistics: dict[str, int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Map token intervals, retaining full-request ordinals and page offsets.

    Main source blocks are CP-sharded. Replicated Indexer stores all CP pages
    inside each source manager block. Its destination is never writer-filtered.
    Tensor pages and manager blocks have distinct IDs; packing is determined
    from token geometry, then checked against bytes and dtype.
    """
    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError("invalid source CP geometry")
    if writer_size <= 0 or not 0 <= writer_rank < writer_size:
        raise ValueError("invalid Main writer geometry")
    if start_token < 0 or end_token < start_token:
        raise ValueError("invalid request token interval")
    remote_span = remote.block_tokens * (cp_size if indexer else 1)
    if (
        local.scale <= 0
        or remote.scale <= 0
        or local.block_tokens <= 0
        or remote.block_tokens <= 0
        or local.block_tokens % local.scale
        or remote_span % remote.scale
    ):
        raise ValueError("cache pages must divide manager token geometry")
    local_page = local.block_tokens // local.scale
    remote_page = remote_span // remote.scale
    if (
        local.block_bytes % local_page
        or remote.block_bytes % remote_page
        or local.block_bytes // local_page != remote.block_bytes // remote_page
        or local.dtype != remote.dtype
    ):
        raise ValueError("incompatible DSA component dtype/token bytes")
    if local.stride < local.block_bytes or remote.stride < remote.block_bytes:
        raise ValueError("overlapping component page stride")
    token_bytes = local.block_bytes // local_page
    result: tuple[list[int], list[int], list[int]] = ([], [], [])
    token = start_token
    while token < end_token:
        global_source_block, source_offset = divmod(token, remote.block_tokens)
        destination_ordinal, destination_offset = divmod(token, local.block_tokens)
        source_ordinal = global_source_block // cp_size
        if indexer:
            source_offset += global_source_block % cp_size * remote.block_tokens
        local_slot, local_offset = divmod(destination_offset, local_page)
        remote_slot, remote_offset = divmod(source_offset, remote_page)
        count = min(
            end_token - token,
            local_page - local_offset,
            remote_page - remote_offset,
            remote.block_tokens - token % remote.block_tokens,
        )
        selected = indexer or (
            global_source_block % cp_size == cp_rank and destination_ordinal % writer_size == writer_rank
        )
        if selected:
            if source_ordinal >= len(source_ids) or destination_ordinal >= len(destination_ids):
                raise ValueError("incomplete DSA source/destination block coverage")
            if (local.capacity and destination_ids[destination_ordinal] >= local.capacity) or (
                remote.capacity and source_ids[source_ordinal] >= remote.capacity
            ):
                raise ValueError("DSA physical block ID exceeds registered capacity")
            local_id = destination_ids[destination_ordinal] * local.scale + local_slot
            remote_id = source_ids[source_ordinal] * remote.scale + remote_slot
            result[0].append(local.base + local_id * local.stride + local_offset * token_bytes)
            result[1].append(remote.base + remote_id * remote.stride + remote_offset * token_bytes)
            result[2].append(count * token_bytes)
        token += count
    merged = coalesce_transfer_lists(*result)
    if (
        local.capacity
        and remote.capacity
        and (
            layout_span_bytes(local) > MAX_REGISTER_MEMORY_BYTES
            or layout_span_bytes(remote) > MAX_REGISTER_MEMORY_BYTES
        )
    ):
        merged = split_transfer_lists_at_region_boundaries(
            *merged,
            local_base=local.base,
            remote_base=remote.base,
        )
    if statistics is not None:
        statistics["entries_before"] = statistics.get("entries_before", 0) + len(result[0])
        statistics["entries_after"] = statistics.get("entries_after", 0) + len(merged[0])
    return merged


@dataclass(frozen=True)
class _DsaPageMapping:
    local_pages: np.ndarray
    remote_pages: np.ndarray
    local_offsets: np.ndarray
    remote_offsets: np.ndarray
    counts: np.ndarray
    max_local_id: int
    max_remote_id: int


def _aligned_page_mapping(
    local, remote, source_ids, destination_ids, start, end, cp_size, cp_rank, writer_rank, writer_size, indexer
):
    """Select logical blocks before expanding their page segments.

    Only equal manager-block geometry with nested page boundaries is handled.
    Partial first/last blocks are clipped without changing full-request
    ordinals. Main ownership solves BOTH CP and writer congruences; Indexer
    remains replicated and is never writer-filtered.
    """
    block = local.block_tokens
    local_page = block // local.scale
    remote_page = remote.block_tokens * (cp_size if indexer else 1) // remote.scale
    if (
        block != remote.block_tokens
        or remote_page > block
        or max(local_page, remote_page) % min(local_page, remote_page)
    ):
        return None
    if max(end, block, cp_size, writer_size) > _MAX_INT64:
        return None
    first_block = start // block
    last_block = first_block if start == end else (end + block - 1) // block
    if last_block * block > _MAX_INT64:
        return None
    if indexer:
        first, period = first_block, 1
    else:
        divisor = gcd(cp_size, writer_size)
        if (writer_rank - cp_rank) % divisor:
            first, period = last_block, 1
        else:
            writer_period = writer_size // divisor
            residue = cp_rank + cp_size * (
                ((writer_rank - cp_rank) // divisor * pow(cp_size // divisor, -1, writer_period)) % writer_period
            )
            period = cp_size * writer_period
            first = first_block + (residue - first_block) % period
    step = min(local_page, remote_page)
    block_count = max(0, (last_block - first + period - 1) // period)
    if (
        block_count * (block // step) > MAX_VECTORIZED_DSA_SEGMENTS
        or period > _MAX_INT64
        or first > _MAX_INT64
    ):
        return None
    blocks = np.arange(first, last_block, period, dtype=np.int64)
    source_ordinals = blocks // cp_size
    if blocks.size and (int(source_ordinals[-1]) >= len(source_ids) or int(blocks[-1]) >= len(destination_ids)):
        # Let the reference preserve first-error ordering if an earlier block
        # also has invalid capacity, rather than prioritizing later coverage.
        return None
    try:
        source = np.asarray(source_ids)
        destination = np.asarray(destination_ids)
        if source.dtype.kind not in "iu" or destination.dtype.kind not in "iu":
            return None
        source = source[source_ordinals]
        destination = destination[blocks]
    except (OverflowError, ValueError):
        return None
    if (
        np.any(source < 0) or np.any(destination < 0)
        or np.any(source > _MAX_INT64) or np.any(destination > _MAX_INT64)
    ):
        return None  # Preserve scalar behavior for unusual physical IDs.
    source = source.astype(np.int64, copy=False)
    destination = destination.astype(np.int64, copy=False)
    max_local_id = int(destination.max()) if destination.size else -1
    max_remote_id = int(source.max()) if source.size else -1
    if (max_local_id + 1) * local.scale > _MAX_INT64 or (max_remote_id + 1) * remote.scale > _MAX_INT64:
        return None
    offsets = np.arange(0, block, step, dtype=np.int64)
    tokens = blocks[:, None] * block + offsets
    clipped = np.maximum(tokens, start)
    counts = np.minimum(tokens + step, end) - clipped
    active = counts > 0
    local_offsets = clipped - blocks[:, None] * block
    remote_offsets = local_offsets + (blocks[:, None] % cp_size * block if indexer else 0)
    arrays = (
        (destination[:, None] * local.scale + local_offsets // local_page)[active],
        (source[:, None] * remote.scale + remote_offsets // remote_page)[active],
        (local_offsets % local_page)[active],
        (remote_offsets % remote_page)[active],
        counts[active],
    )
    for array in arrays:
        array.flags.writeable = False
    return _DsaPageMapping(*arrays, max_local_id, max_remote_id)


def build_component_read(
    local: DsaCacheLayout,
    remote: DsaCacheLayout,
    source_ids: tuple[int, ...],
    destination_ids: tuple[int, ...],
    start_token: int,
    end_token: int,
    *,
    cp_size: int,
    cp_rank: int,
    writer_rank: int,
    writer_size: int,
    indexer: bool,
    statistics: dict[str, int] | None = None,
    mapping_cache: dict[tuple, _DsaPageMapping] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Vectorize compatible page mappings, otherwise use the scalar reference.

    The caller may share a bounded cache ONLY within one endpoint task. Keys
    include request IDs' block tuples, interval, component and ownership/page
    geometry. Bases, strides, token bytes and capacities are deliberately not
    cached: they are checked/applied separately for every layer. No final
    addresses, device tensors or mutable command state are retained.
    """
    # Keep validation before fast-path selection, including empty/unowned tasks.
    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError("invalid source CP geometry")
    if writer_size <= 0 or not 0 <= writer_rank < writer_size:
        raise ValueError("invalid Main writer geometry")
    if start_token < 0 or end_token < start_token:
        raise ValueError("invalid request token interval")
    remote_span = remote.block_tokens * (cp_size if indexer else 1)
    if (
        local.scale <= 0
        or remote.scale <= 0
        or local.block_tokens <= 0
        or remote.block_tokens <= 0
        or local.block_tokens % local.scale
        or remote_span % remote.scale
    ):
        raise ValueError("cache pages must divide manager token geometry")
    local_page, remote_page = local.block_tokens // local.scale, remote_span // remote.scale
    if (
        local.block_bytes % local_page
        or remote.block_bytes % remote_page
        or local.block_bytes // local_page != remote.block_bytes // remote_page
        or local.dtype != remote.dtype
    ):
        raise ValueError("incompatible DSA component dtype/token bytes")
    if local.stride < local.block_bytes or remote.stride < remote.block_bytes:
        raise ValueError("overlapping component page stride")
    token_bytes = local.block_bytes // local_page
    fallback_args = dict(
        cp_size=cp_size, cp_rank=cp_rank, writer_rank=writer_rank, writer_size=writer_size,
        indexer=indexer, statistics=statistics,
    )

    def scalar():
        return _build_component_read_scalar(
            local, remote, source_ids, destination_ids, start_token, end_token, **fallback_args
        )

    if token_bytes <= 0 or token_bytes * (end_token - start_token) > _MAX_INT64:
        return scalar()
    key = (
        source_ids, destination_ids, start_token, end_token, cp_size, cp_rank, writer_rank, writer_size, indexer,
        local.block_tokens, remote.block_tokens, local.scale, remote.scale,
    )
    mapping = None if mapping_cache is None else mapping_cache.get(key)
    if mapping is None:
        mapping = _aligned_page_mapping(
            local, remote, source_ids, destination_ids, start_token, end_token,
            cp_size, cp_rank, writer_rank, writer_size, indexer,
        )
        if mapping is None:
            return scalar()
        if mapping_cache is not None and len(mapping_cache) < MAX_CACHED_DSA_MAPPINGS:
            mapping_cache[key] = mapping
    if (local.capacity and mapping.max_local_id >= local.capacity) or (
        remote.capacity and mapping.max_remote_id >= remote.capacity
    ):
        if mapping.counts.size:
            raise ValueError("DSA physical block ID exceeds registered capacity")
    # Guard every multiply/add in Python integers before int64 address math.
    # Wider/negative address layouts preserve the scalar Python-int semantics.
    for layout, maximum, page in (
        (local, mapping.max_local_id, local_page), (remote, mapping.max_remote_id, remote_page)
    ):
        if layout.base < 0 or layout.stride < 0 or (
            layout.base + (maximum + 1) * layout.scale * layout.stride + page * token_bytes > _MAX_INT64
        ):
            return scalar()
    dst = local.base + mapping.local_pages * local.stride + mapping.local_offsets * token_bytes
    src = remote.base + mapping.remote_pages * remote.stride + mapping.remote_offsets * token_bytes
    lengths = mapping.counts * token_bytes
    if lengths.size:
        contiguous = (dst[1:] == dst[:-1] + lengths[:-1]) & (src[1:] == src[:-1] + lengths[:-1])
        starts = np.concatenate((np.array([0], dtype=np.int64), np.flatnonzero(~contiguous) + 1))
        merged = (dst[starts].tolist(), src[starts].tolist(), np.add.reduceat(lengths, starts).tolist())
    else:
        merged = ([], [], [])
    if local.capacity and remote.capacity and (
        layout_span_bytes(local) > MAX_REGISTER_MEMORY_BYTES or layout_span_bytes(remote) > MAX_REGISTER_MEMORY_BYTES
    ):
        merged = split_transfer_lists_at_region_boundaries(
            *merged, local_base=local.base, remote_base=remote.base
        )
    if statistics is not None:
        statistics["entries_before"] = statistics.get("entries_before", 0) + int(lengths.size)
        statistics["entries_after"] = statistics.get("entries_after", 0) + len(merged[0])
    return merged


def coalesce_transfer_lists(local, remote, lengths):
    """Merge only byte ranges contiguous at both ends of one endpoint read."""
    if len(local) != len(remote) or len(local) != len(lengths):
        raise ValueError("transfer list coverage mismatch")
    result = ([], [], [])
    for dst, src, size in zip(local, remote, lengths):
        if size <= 0:
            raise ValueError("transfer length must be positive")
        if result[2] and dst == result[0][-1] + result[2][-1] and src == result[1][-1] + result[2][-1]:
            result[2][-1] += size
        else:
            result[0].append(dst)
            result[1].append(src)
            result[2].append(size)
    return result


def split_transfer_lists_at_region_boundaries(
    local,
    remote,
    lengths,
    *,
    local_base: int,
    remote_base: int,
    max_region_bytes: int = MAX_REGISTER_MEMORY_BYTES,
):
    """Split reads at both endpoints' deterministic registration edges."""
    if len(local) != len(remote) or len(local) != len(lengths):
        raise ValueError("transfer list coverage mismatch")
    if max_region_bytes <= 0:
        raise ValueError("max_region_bytes must be positive")
    result = ([], [], [])
    for dst, src, size in zip(local, remote, lengths):
        if size <= 0 or dst < local_base or src < remote_base:
            raise ValueError("invalid transfer range for registered layout")
        remaining = size
        while remaining:
            local_available = max_region_bytes - (dst - local_base) % max_region_bytes
            remote_available = max_region_bytes - (src - remote_base) % max_region_bytes
            part = min(remaining, local_available, remote_available)
            result[0].append(dst)
            result[1].append(src)
            result[2].append(part)
            dst += part
            src += part
            remaining -= part
    return result
