import torch

import triton
import triton.language as tl

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
    BLK_BATCHES: tl.constexpr
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    batch_off = i_b * BLK_BATCHES + tl.arange(0, BLK_BATCHES)
    off = batch_off[:, None] * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off[None, :]
    head_mask = head_off < NUM_HEADS
    mask = head_mask[None, :] & (batch_off[:, None] < NUM_BATCHES)
    blk_A_log = tl.load(A_log + head_off, mask=head_mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=head_mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
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
    NUM_BATCH_GROUPS = batch
    BLK_BATCHES = 1
    if batch > 40:
        BLK_BATCHES = triton.next_power_of_2(triton.cdiv(batch, 32))
        NUM_BATCH_GROUPS = triton.cdiv(batch, BLK_BATCHES)
        
    grid = (NUM_BATCH_GROUPS, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty_like(a, dtype=torch.float32)
    fused_gdn_gating_kernel[grid](g,
                                  A_log,
                                  a,
                                  dt_bias,
                                  seq_len,
                                  num_heads,
                                  batch,
                                  beta,
                                  threshold,
                                  8,
                                  BLK_BATCHES=BLK_BATCHES,
                                  num_warps=1)
    return g