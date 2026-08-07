
from vllm.triton_utils import tl, triton

@triton.jit
def _max_softmax_kernel(
    logits_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    stride_row: tl.constexpr,
    stride_col: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ):
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    row_max = tl.full((), -float("inf"), tl.float32) 

    for start in tl.range(0, n_cols, BLOCK_SIZE): 
        cols = start + offs 
        mask = cols < n_cols 

        vals = tl.load( 
            logits_ptr + row_idx * stride_row + cols * stride_col, 
            mask=mask, 
            other=-float("inf"), 
        ).to(tl.float32) 

        block_max = tl.max(vals, axis=0) 
        row_max = tl.maximum(row_max, block_max) 

    denom = tl.full((), 0.0, tl.float32) 

    for start in tl.range(0, n_cols, BLOCK_SIZE): 
        cols = start + offs 
        mask = cols < n_cols 

        vals = tl.load( 
            logits_ptr + row_idx * stride_row + cols * stride_col, 
            mask=mask, 
            other=-float("inf"), 
        ).to(tl.float32) 

        exp_vals = tl.exp(vals - row_max) 
        exp_vals = tl.where(mask, exp_vals, 0.0) 

        denom += tl.sum(exp_vals, axis=0) 
    # exp(max - logsumexp) = 1 / sum(exp(x - max)) 
    conf = 1.0 / denom 

    tl.store(out_ptr + row_idx, conf)
