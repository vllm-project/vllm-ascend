# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host destination geometry layered on Mooncake's source endpoint plan."""

from dataclasses import dataclass


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
