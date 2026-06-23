import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

BLOCK_SIZE = 1024


@triton.jit
def _count_compressed_tokens_kernel(
    query_start_loc,
    positions,
    compressed_counts,
    compressed_query_start_loc,
    COMPRESS_RATIO: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:
        tl.store(compressed_query_start_loc, 0)

    start = tl.load(query_start_loc + req_idx)
    end = tl.load(query_start_loc + req_idx + 1)
    has_tokens = start < end
    first_pos = tl.load(positions + start, mask=has_tokens, other=0).to(tl.int64)
    last_pos = tl.load(positions + end - 1, mask=has_tokens, other=0).to(tl.int64)
    count = (last_pos + 1) // COMPRESS_RATIO - first_pos // COMPRESS_RATIO
    count = tl.where(has_tokens, count, 0)
    tl.store(compressed_counts + req_idx, count)


@triton.jit
def _build_single_req_compressed_slot_mapping_kernel(
    max_output_tokens,
    query_start_loc,
    positions,
    block_table,
    block_table_stride,
    block_size,
    slot_mapping,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE_: tl.constexpr,
):
    block_pid = tl.program_id(0)
    offsets = block_pid * BLOCK_SIZE_ + tl.arange(0, BLOCK_SIZE_)
    output_mask = offsets < max_output_tokens

    start = tl.load(query_start_loc)
    end = tl.load(query_start_loc + 1)
    has_tokens = start < end
    first_pos = tl.load(positions + start, mask=has_tokens, other=0).to(tl.int64)
    last_pos = tl.load(positions + end - 1, mask=has_tokens, other=0).to(tl.int64)
    first_window = first_pos // COMPRESS_RATIO
    compressed_count = (last_pos + 1) // COMPRESS_RATIO - first_window
    compressed_count = tl.where(has_tokens, compressed_count, 0)

    valid_out = output_mask & (offsets < compressed_count)
    compressed_cache_pos = first_window + offsets
    block_idx = compressed_cache_pos // block_size
    block_offset = compressed_cache_pos - block_idx * block_size
    block_number = tl.load(block_table + block_idx, mask=valid_out, other=0)
    slot_id = block_number * block_size + block_offset
    slot_id = tl.where(valid_out, slot_id, PAD_ID)
    tl.store(slot_mapping + offsets, slot_id, mask=output_mask)


@triton.jit
def _build_compressed_slot_mapping_kernel(
    max_output_tokens,
    query_start_loc,
    positions,
    block_table,
    block_table_stride,
    block_size,
    compressed_query_start_loc,
    slot_mapping,
    NUM_REQS: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE_: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx >= NUM_REQS:
        total_compressed_tokens = tl.load(compressed_query_start_loc + NUM_REQS)
        fill_idx = req_idx - NUM_REQS
        row_start = total_compressed_tokens + fill_idx * BLOCK_SIZE_
        offsets = row_start + tl.arange(0, BLOCK_SIZE_)
        tl.store(slot_mapping + offsets, PAD_ID, mask=offsets < max_output_tokens)
        return

    start = tl.load(query_start_loc + req_idx)
    end = tl.load(query_start_loc + req_idx + 1)
    first_pos = tl.load(positions + start, mask=start < end, other=0).to(tl.int64)
    first_window = first_pos // COMPRESS_RATIO
    req_prefix = tl.load(compressed_query_start_loc + req_idx).to(tl.int64)
    req_end = tl.load(compressed_query_start_loc + req_idx + 1).to(tl.int64)

    for row_start in range(req_prefix, req_end, BLOCK_SIZE_):
        offsets = row_start + tl.arange(0, BLOCK_SIZE_)
        valid_out = offsets < req_end
        compressed_cache_pos = first_window + offsets - req_prefix
        block_idx = compressed_cache_pos // block_size
        block_offset = compressed_cache_pos - block_idx * block_size
        block_number = tl.load(
            block_table + req_idx * block_table_stride + block_idx,
            mask=valid_out,
            other=0,
        )
        tl.store(
            slot_mapping + offsets,
            block_number * block_size + block_offset,
            mask=valid_out & (offsets < max_output_tokens),
        )


def build_compressed_slot_mapping(
    *,
    num_reqs: int,
    max_output_tokens: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    compressed_counts_buffer: torch.Tensor,
    compressed_query_start_loc_buffer: torch.Tensor,
) -> None:
    """Build the original compact compressor slot_mapping without host sync."""
    if num_reqs == 0 or positions.shape[0] == 0:
        return

    if num_reqs == 1:
        fill_blocks = triton.cdiv(max_output_tokens, BLOCK_SIZE)
        _build_single_req_compressed_slot_mapping_kernel[(fill_blocks,)](
            max_output_tokens,
            query_start_loc,
            positions,
            block_table,
            block_table.stride(0),
            block_size,
            slot_mapping,
            COMPRESS_RATIO=compress_ratio,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE_=BLOCK_SIZE,
        )
        return

    compressed_counts = compressed_counts_buffer[:num_reqs]
    compressed_query_start_loc = compressed_query_start_loc_buffer[: num_reqs + 1]
    _count_compressed_tokens_kernel[(num_reqs,)](
        query_start_loc,
        positions,
        compressed_counts,
        compressed_query_start_loc,
        COMPRESS_RATIO=compress_ratio,
    )
    torch.cumsum(compressed_counts, dim=0, out=compressed_query_start_loc[1:])

    # The padding rows must be initialized on device because metadata build no
    # longer knows a CPU total for this compact buffer. Scatter treats PAD_ID as
    # invalid and skips the dirty rows.
    fill_blocks = triton.cdiv(max_output_tokens, BLOCK_SIZE)
    _build_compressed_slot_mapping_kernel[(num_reqs + fill_blocks,)](
        max_output_tokens,
        query_start_loc,
        positions,
        block_table,
        block_table.stride(0),
        block_size,
        compressed_query_start_loc,
        slot_mapping,
        NUM_REQS=num_reqs,
        COMPRESS_RATIO=compress_ratio,
        PAD_ID=PAD_SLOT_ID,
        BLOCK_SIZE_=BLOCK_SIZE,
    )
