from vllm.triton_utils import tl, triton

BUILD_LOCAL_METADATA_BLOCK_SIZE: tl.constexpr = 1024


# These scalar values vary per batch and TP rank; keep them out of the JIT cache key.
@triton.jit(
    do_not_specialize=[
        "local_start",
        "local_end",
        "num_reqs",
    ]
)
def build_local_metadata_triton(
    query_start_loc_ptr,  # [num_reqs + 1], int32
    seq_lens_ptr,  # [num_reqs], int32
    local_query_start_loc_ptr,  # [max_num_seqs + 1], int32  (output, pre-zeroed)
    local_seq_lens_ptr,  # [max_num_seqs], int32      (output, pre-zeroed)
    local_start,
    local_end,
    num_reqs,
    start_pos_out_ptr,  # [max_num_seqs], int32      (output)
    COMPUTE_START_POS,
):
    """Fused NPU kernel for local token metadata computation

    reduce kernel launch overhead.
    """
    offsets = tl.program_id(0) * BUILD_LOCAL_METADATA_BLOCK_SIZE + tl.arange(0, BUILD_LOCAL_METADATA_BLOCK_SIZE)
    mask = offsets < num_reqs

    q_base = tl.load(query_start_loc_ptr)
    q_start = tl.load(query_start_loc_ptr + offsets, mask=mask, other=0)
    q_end = tl.load(query_start_loc_ptr + offsets + 1, mask=mask, other=0)
    seq_len = tl.load(seq_lens_ptr + offsets, mask=mask, other=0)

    lqs = tl.maximum(tl.minimum(q_start, local_end), local_start)
    lqe = tl.maximum(tl.minimum(q_end, local_end), local_start)
    lql = lqe - lqs

    local_base = tl.maximum(tl.minimum(q_base, local_end), local_start)

    tl.store(local_query_start_loc_ptr + 1 + offsets, lqe - local_base, mask=mask)

    offset = q_end - lqe
    result = tl.where((lql > 0) & (seq_len > 0), tl.maximum(seq_len - offset, 0), 0)
    tl.store(local_seq_lens_ptr + offsets, result, mask=mask)

    tl.store(
        start_pos_out_ptr + offsets,
        seq_len - (q_end - q_start),
        mask=mask & COMPUTE_START_POS,
    )
