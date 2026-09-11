# Isolate the block-verify bias on NPU. Run on the NPU box:
#   python _tmp_block_verify_repro.py
import torch
import triton
from vllm.triton_utils import tl  # noqa: F401

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_cumulative_log_p_kernel,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_local_logits_stats_kernel as _compute_block_stats_kernel,
)
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

torch.manual_seed(42)
device = "npu"
V = 4096
VOCAB_BLOCK_SIZE = 8192


def build(K: int, N: int, placeholders: int = 0):
    target_1d = torch.randn(V, device=device, dtype=torch.float32)
    draft_1d = torch.randn(V, device=device, dtype=torch.float32)
    num_logits = N * (K + 1)
    target_logits = target_1d.unsqueeze(0).expand(num_logits, -1).contiguous()
    draft_logits = draft_1d.view(1, 1, V).expand(N, K, -1).contiguous()
    q = torch.softmax(draft_1d, dim=0)
    draft_tokens = torch.multinomial(q.cpu().expand(N, -1), K, replacement=True).to(device)
    draft_2d = torch.zeros(N, K + 1, dtype=torch.int64, device=device)
    draft_2d[:, 1:] = draft_tokens
    if placeholders:
        draft_2d[:, K + 1 - placeholders :] = -1
    return (
        dict(
            target_logits=target_logits,
            draft_logits=draft_logits,
            draft_sampled=draft_2d.reshape(-1),
            cu_num_logits=torch.arange(N + 1, dtype=torch.int32, device=device) * (K + 1),
            pos=torch.arange(num_logits, dtype=torch.int32, device=device),
            idx_mapping=torch.arange(N, dtype=torch.int32, device=device),
            expanded_idx_mapping=torch.arange(N, dtype=torch.int32, device=device).repeat_interleave(K + 1),
            expanded_local_pos=torch.arange(K + 1, dtype=torch.int32, device=device).repeat(N),
            temperature=torch.ones(N, dtype=torch.float32, device=device),
            seed=torch.arange(N, dtype=torch.int64, device=device),
        ),
        target_1d,
        draft_1d,
    )


# ---- Check 1: block vs standard on identical inputs (must be equivalent) ----
# K=1 block degenerates to Leviathan; K=3 + 2 trailing placeholders tests only
# position 0 in both modes. Same u, same accept event => outputs must match.
for label, K, ph in [("K=1", 1, 0), ("K=3+2ph", 3, 2)]:
    inputs, _, _ = build(K, N=4096, placeholders=ph)
    std = rejection_sample(**inputs, num_speculative_steps=K)
    blk = rejection_sample(**inputs, num_speculative_steps=K, use_block_verification=True)
    same_n = torch.equal(std[1], blk[1])
    steps = torch.arange(K + 1, device=device)
    valid = steps.unsqueeze(0) < std[1].unsqueeze(1)
    same_s = torch.equal(std[0][valid], blk[0][valid])
    print(f"[{label}] num_sampled identical: {same_n} | sampled identical: {same_s}")
    print(f"[{label}] accept-len std={(std[1] - 1).float().mean():.4f} blk={(blk[1] - 1).float().mean():.4f}")

# ---- Check 2: empirical vs theoretical acceptance (K=1, block) ----
inputs, t1d, d1d = build(1, N=4096)
p = torch.softmax(t1d, dim=0)
q = torch.softmax(d1d, dim=0)
x1 = inputs["draft_sampled"].view(-1, 2)[:, 1]
ref_h = (p / q)[x1].clamp(max=1.0)
blk = rejection_sample(**inputs, num_speculative_steps=1, use_block_verification=True)
emp = (blk[1] == 2).float().mean().item()
print(f"[K=1] empirical accept={emp:.4f} vs E[min(p/q,1)]={ref_h.mean().item():.4f}")

# ---- Check 3: _compute_cumulative_log_p_kernel vs torch reference ----
K, N = 1, 4096
inputs, t1d, d1d = build(K, N)
num_logits = N * (K + 1)
vocab_num_blocks = triton.cdiv(V, VOCAB_BLOCK_SIZE)
padded = triton.next_power_of_2(vocab_num_blocks)
t_argmax = torch.empty(num_logits, vocab_num_blocks, dtype=torch.int64, device=device)
t_max = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=device)
t_sumexp = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=device)
d_max = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=device)
d_sumexp = torch.empty(num_logits, vocab_num_blocks, dtype=torch.float32, device=device)
_compute_block_stats_kernel[(num_logits, vocab_num_blocks)](
    t_argmax,
    t_argmax.stride(0),
    t_max,
    t_max.stride(0),
    t_sumexp,
    t_sumexp.stride(0),
    d_max,
    d_max.stride(0),
    d_sumexp,
    d_sumexp.stride(0),
    inputs["target_logits"],
    inputs["target_logits"].stride(0),
    inputs["draft_logits"],
    inputs["draft_logits"].stride(0),
    inputs["draft_logits"].stride(1),
    inputs["expanded_idx_mapping"],
    inputs["expanded_local_pos"],
    inputs["temperature"],
    V,
    K,
    BLOCK_SIZE=VOCAB_BLOCK_SIZE,
    HAS_DRAFT_LOGITS=True,
    has_auto_blockify_blacklist_op=True,
)
cum = torch.empty(num_logits, dtype=torch.float32, device=device)
_compute_cumulative_log_p_kernel[(N,)](
    cum,
    inputs["target_logits"],
    inputs["target_logits"].stride(0),
    t_max,
    t_max.stride(0),
    t_sumexp,
    t_sumexp.stride(0),
    inputs["draft_sampled"],
    inputs["draft_logits"],
    inputs["draft_logits"].stride(0),
    inputs["draft_logits"].stride(1),
    d_max,
    d_max.stride(0),
    d_sumexp,
    d_sumexp.stride(0),
    inputs["cu_num_logits"],
    inputs["idx_mapping"],
    inputs["temperature"],
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS=padded,
    HAS_DRAFT_LOGITS=True,
    num_warps=1,
)
x1 = inputs["draft_sampled"].view(N, K + 1)[:, 1]
ref = torch.log((p / q)[x1].clamp(max=1.0))
got = cum.view(N, K + 1)[:, 0]
err = (got - ref).abs()
print(f"[cumulative_log_p] max_abs_err={err.max():.6e} mean_abs_err={err.mean():.6e}")
bad = err > 1e-3
print(f"[cumulative_log_p] rows off by >1e-3: {int(bad.sum())} / {N}")
if bad.any():
    i = int(bad.nonzero()[0])
    print(f"  example row {i}: kernel={got[i]:.6f} ref={ref[i]:.6f} token={int(x1[i])}")
