from vllm.triton_utils import tl, triton


@triton.jit
def build_local_metadata_triton(
    query_start_loc_ptr,  # [num_reqs + 1], int32
    seq_lens_ptr,  # [num_reqs], int32
    local_query_start_loc_ptr,  # [max_num_seqs + 1], int32  (output, pre-zeroed)
    local_seq_lens_ptr,  # [max_num_seqs], int32      (output, pre-zeroed)
    local_start,
    local_end,
    num_reqs,
    start_pos_out_ptr,  # [max_num_seqs], int32      (output)
    BLOCK_NUM_REQS: tl.constexpr,
    COMPUTE_START_POS: tl.constexpr,
):
    """Fused NPU kernel for local token metadata computation

    reduce kernel launch overhead.
    """
    offsets = tl.arange(0, BLOCK_NUM_REQS)
    mask = offsets < num_reqs

    q_start = tl.load(query_start_loc_ptr + offsets, mask=mask, other=0)
    q_end = tl.load(query_start_loc_ptr + offsets + 1, mask=mask, other=0)
    seq_len = tl.load(seq_lens_ptr + offsets, mask=mask, other=0)

    lqs = tl.maximum(tl.minimum(q_start, local_end), local_start)
    lqe = tl.maximum(tl.minimum(q_end, local_end), local_start)
    lql = lqe - lqs

    cum = tl.cumsum(lql, axis=0)

    tl.store(local_query_start_loc_ptr + 1 + offsets, cum, mask=mask)

    offset = q_end - lqe
    result = tl.where(lql > 0, seq_len - offset, 0)
    tl.store(local_seq_lens_ptr + offsets, result, mask=mask)

    if COMPUTE_START_POS:
        tl.store(start_pos_out_ptr + offsets, seq_len - (q_end - q_start), mask=mask)
