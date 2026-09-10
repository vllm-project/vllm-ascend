# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host destination geometry layered on Mooncake's source endpoint plan."""

from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass

MAX_REGISTER_MEMORY_BYTES = 64 * 1024**3


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
