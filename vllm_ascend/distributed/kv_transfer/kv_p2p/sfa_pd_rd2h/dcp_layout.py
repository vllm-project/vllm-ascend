# SPDX-License-Identifier: Apache-2.0
"""Address mapping for PCP1, full-TP DCP with block-sized interleaving.

Main MLA blocks are rank-local. Indexer blocks are replicated and expanded:
physical indexer block = scheduler block ID * DCP + global-block-within-stripe.
No model data is copied here; output descriptors use the existing transport.
"""

from collections.abc import Iterator, Mapping, Sequence


def validate_dcp_layout_metadata(
    metadata: Mapping[str, int],
    *,
    tp_size: int,
    tp_rank: int,
    member: int,
    ratio: int,
    main_block_size: int,
    indexer_block_size: int,
) -> tuple[int, int, int]:
    """Reject incompatible source layouts before issuing any remote read."""
    size, rank, block = int(metadata["size"]), int(metadata["rank"]), int(metadata["block_size"])
    if (
        ratio < 1
        or tp_size < 1
        or size <= 1
        or not 0 <= rank < size
        or size != tp_size * ratio
        or member != rank % ratio
        or tp_rank != rank // ratio
    ):
        raise ValueError("DCP contributor topology mismatch")
    if block <= 0 or block != main_block_size or int(metadata["indexer_block_size"]) != indexer_block_size:
        raise ValueError("DCP P/D physical block sizes must match")
    if block != indexer_block_size:
        raise ValueError("DCP main and indexer block sizes must match")
    return size, rank, block


def dcp_block_segments(
    source_blocks: Sequence[int],
    destination_blocks: Sequence[int],
    *,
    start_block: int,
    token_start: int,
    token_end: int,
    block_size: int,
    dcp_size: int,
    owner_rank: int | None = None,
    contributor: int = 0,
    contributors: int = 1,
) -> Iterator[tuple[int, int, int, int, int]]:
    """Yield (source block, source offset, destination block, offset, tokens).

    owner_rank selects a main-KV shard. None selects the replicated indexer,
    partitioned between the P contributors feeding this D rank.
    """
    if (
        block_size <= 0
        or dcp_size <= 1
        or token_start < 0
        or token_end < token_start
        or start_block < 0
        or contributors < 1
        or not 0 <= contributor < contributors
    ):
        raise ValueError("Invalid DCP transfer interval or topology")
    if owner_rank is not None and not 0 <= owner_rank < dcp_size:
        raise ValueError("Invalid DCP owner rank")
    if token_start == token_end:
        return
    first = token_start // block_size
    stop = (token_end + block_size - 1) // block_size
    if stop > len(destination_blocks):
        raise ValueError("DCP transfer exceeds destination block allocation")
    for global_block in range(first, stop):
        if owner_rank is not None:
            if global_block % dcp_size != owner_rank:
                continue
        elif global_block % contributors != contributor:
            continue
        source_index = global_block // dcp_size - start_block
        if not 0 <= source_index < len(source_blocks):
            raise ValueError("DCP transfer exceeds source block allocation")
        source_block = source_blocks[source_index]
        if owner_rank is None:
            source_block = source_block * dcp_size + global_block % dcp_size
        offset = max(token_start - global_block * block_size, 0)
        end = min(token_end - global_block * block_size, block_size)
        yield (
            source_block,
            offset,
            destination_blocks[global_block],
            offset,
            end - offset,
        )
