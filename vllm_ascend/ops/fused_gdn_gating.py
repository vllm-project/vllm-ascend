import torch

import triton
import triton.language as tl
import triton.runtime.driver as driver 

# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    COL_ITER: tl.constexpr,
    BLK_BATCHES: tl.constexpr,
    ROW_ITER: tl.constexpr,
):
    # New impl
    i_b, i_s = tl.program_id(0), tl.program_id(1)
    for row_idx in range(0, ROW_ITER):
        batch_off = i_b * ROW_ITER * BLK_BATCHES + row_idx * BLK_BATCHES + tl.arange(0, BLK_BATCHES)

        for col_idx in range(0, COL_ITER):
            head_off = col_idx * BLK_HEADS + tl.arange(0, BLK_HEADS)

            off = batch_off[:, None] * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off[None, :]
            head_mask = head_off < NUM_HEADS
            mask = head_mask[None, :] & (batch_off[:, None] < NUM_BATCHES)
            blk_A_log = tl.load(A_log + head_off, mask=head_mask)
            blk_a = tl.load(a + off, mask=mask)
            blk_bias = tl.load(dt_bias + head_off, mask=head_mask)
            
            x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
            softplus_x = tl.where(beta * x <= threshold,
                                (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)

            blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x

            tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)

def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    batch, num_heads = a.shape
    seq_len = 1

    num_cores = driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"]

    # a_log_size = A_log.element_size() * A_log.nelement()
    # a_size = a.element_size() * a.nelement()
    # dt_bias_size = dt_bias.element_size() * dt_bias.nelement()
    
    # 1. Row
    BLK_HEADS = 8 # TODO
    COL_ITER = triton.cdiv(num_heads, BLK_HEADS)

    # 2. Col
    if batch <= num_cores:
        progs = batch
        BLK_BATCHES = 1
        ROW_ITER = 1
    else:
        progs = num_cores

        factor = 64 # Black box ub factor
        row_per_core = triton.cdiv(batch, num_cores)
        BLK_BATCHES = triton.next_power_of_2(triton.cdiv(1572864, factor * BLK_HEADS) // a.element_size()) // 2
        ROW_ITER = triton.cdiv(row_per_core, BLK_BATCHES) 


    g = torch.empty_like(a, dtype=torch.float32)

    grid = (progs, seq_len)
    fused_gdn_gating_kernel[grid](g,
                                  A_log,
                                  a,
                                  dt_bias,
                                  seq_len,
                                  num_heads,
                                  batch,
                                  beta,
                                  threshold,
                                  BLK_HEADS=BLK_HEADS,
                                  COL_ITER=COL_ITER,
                                  BLK_BATCHES=BLK_BATCHES,
                                  ROW_ITER=ROW_ITER,)
    return g
