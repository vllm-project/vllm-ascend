# Decisive diagnostics for the block-verify chi2 failures. Run on the NPU box:
#   python _tmp_block_verify_repro2.py
import math

import torch
import triton

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_cumulative_log_p_kernel,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_local_logits_stats_kernel as _compute_block_stats_kernel,
)
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

V = 4096
VOCAB_BLOCK_SIZE = 8192
DEVICE = "npu"
TRIALS_PER_CALL = 10000


def make_inputs(target_1d, draft_1d, K, N, temp=1.0, draft_tokens=None):
    num_logits = N * (K + 1)
    target_logits = target_1d.unsqueeze(0).expand(num_logits, -1).contiguous()
    draft_logits = draft_1d.view(1, 1, V).expand(N, K, -1).contiguous()
    if draft_tokens is None:
        q = torch.softmax(draft_1d / temp, dim=0)
        draft_tokens = torch.multinomial(q.cpu().expand(N, -1), K, replacement=True).to(DEVICE)
    draft_2d = torch.zeros(N, K + 1, dtype=torch.int64, device=DEVICE)
    draft_2d[:, 1:] = draft_tokens
    return dict(
        target_logits=target_logits,
        draft_logits=draft_logits,
        draft_sampled=draft_2d.reshape(-1),
        cu_num_logits=torch.arange(N + 1, dtype=torch.int32, device=DEVICE) * (K + 1),
        pos=torch.arange(num_logits, dtype=torch.int32, device=DEVICE),
        idx_mapping=torch.arange(N, dtype=torch.int32, device=DEVICE),
        expanded_idx_mapping=torch.arange(N, dtype=torch.int32, device=DEVICE).repeat_interleave(K + 1),
        expanded_local_pos=torch.arange(K + 1, dtype=torch.int32, device=DEVICE).repeat(N),
        temperature=torch.full((N,), temp, dtype=torch.float32, device=DEVICE),
        seed=torch.arange(N, dtype=torch.int64, device=DEVICE),
    )


def chi2_label(tokens, probs, label):
    """Same statistic/threshold as the test's _assert_distribution_match."""
    num = tokens.shape[0]
    observed = torch.zeros(V, device=DEVICE)
    observed.scatter_add_(0, tokens, torch.ones(num, device=DEVICE))
    expected = probs * num
    sufficient = expected >= 5.0
    obs_all = torch.cat([observed[sufficient], observed[~sufficient].sum().unsqueeze(0)])
    exp_all = torch.cat([expected[sufficient], expected[~sufficient].sum().unsqueeze(0)])
    keep = exp_all >= 5.0
    obs_all, exp_all = obs_all[keep], exp_all[keep]
    chi2 = ((obs_all - exp_all) ** 2 / exp_all).sum().item()
    df = obs_all.shape[0] - 1
    thr = df + 10 * math.sqrt(2 * df)
    verdict = "PASS" if chi2 < thr else "FAIL"
    print(f"  [{label}] chi2={chi2:.1f} df={df} threshold={thr:.1f} -> {verdict}")
    return chi2 < thr


def iter_chunks(build, num_trials, K):
    for start in range(0, num_trials, TRIALS_PER_CALL):
        n = min(TRIALS_PER_CALL, num_trials - start)
        inputs = build(n)
        inputs["seed"] = inputs["seed"] + start
        inputs["pos"] = inputs["pos"] + start * (K + 1)
        yield inputs


print("=" * 30, "Exp D: cumulative kernel, known single value", "=" * 30)
# Fixed draft token 777 everywhere, anchor token 555. The correct
# cumulative_log_p[start] is ONE known number; any other value identifies
# the failure mode (swap / wrong slot / garbage).
torch.manual_seed(42)
t1d = torch.randn(V, device=DEVICE)
d1d = torch.randn(V, device=DEVICE)
p, q = torch.softmax(t1d, 0), torch.softmax(d1d, 0)
K, N = 1, 256
tokens = torch.full((N, 1), 777, dtype=torch.int64, device=DEVICE)
inputs = make_inputs(t1d, d1d, K, N, draft_tokens=tokens)
inputs["draft_sampled"].view(N, 2)[:, 0] = 555
num_logits = N * (K + 1)
vnb = triton.cdiv(V, VOCAB_BLOCK_SIZE)
pnb = triton.next_power_of_2(vnb)
t_am = torch.empty(num_logits, vnb, dtype=torch.int64, device=DEVICE)
t_mx = torch.empty(num_logits, vnb, dtype=torch.float32, device=DEVICE)
t_se = torch.empty(num_logits, vnb, dtype=torch.float32, device=DEVICE)
d_mx = torch.full((num_logits, vnb), float("nan"), dtype=torch.float32, device=DEVICE)
d_se = torch.full((num_logits, vnb), float("nan"), dtype=torch.float32, device=DEVICE)
_compute_block_stats_kernel[(num_logits, vnb)](
    t_am, t_am.stride(0), t_mx, t_mx.stride(0), t_se, t_se.stride(0),
    d_mx, d_mx.stride(0), d_se, d_se.stride(0),
    inputs["target_logits"], inputs["target_logits"].stride(0),
    inputs["draft_logits"], inputs["draft_logits"].stride(0), inputs["draft_logits"].stride(1),
    inputs["expanded_idx_mapping"], inputs["expanded_local_pos"], inputs["temperature"], V, K,
    BLOCK_SIZE=VOCAB_BLOCK_SIZE, HAS_DRAFT_LOGITS=True, has_auto_blockify_blacklist_op=True,
)
cum = torch.empty(num_logits, dtype=torch.float32, device=DEVICE)
_compute_cumulative_log_p_kernel[(N,)](
    cum, inputs["target_logits"], inputs["target_logits"].stride(0),
    t_mx, t_mx.stride(0), t_se, t_se.stride(0),
    inputs["draft_sampled"], inputs["draft_logits"], inputs["draft_logits"].stride(0),
    inputs["draft_logits"].stride(1), d_mx, d_mx.stride(0), d_se, d_se.stride(0),
    inputs["cu_num_logits"], inputs["idx_mapping"], inputs["temperature"], vnb,
    PADDED_VOCAB_NUM_BLOCKS=pnb, HAS_DRAFT_LOGITS=True, num_warps=1,
)
ref_ok = min(math.log(p[777].item() / q[777].item()), 0.0)
swap = min(math.log(q[777].item() / p[777].item()), 0.0)
wrongslot = min(math.log(p[555].item() / q[555].item()), 0.0)
print(f"  correct(log p/q @777)={ref_ok:+.6f}  swap(log q/p @777)={swap:+.6f}  wrongslot(@555)={wrongslot:+.6f}")
print(f"  kernel cum[start+0] first 4 rows: {[round(v, 6) for v in cum.view(N, 2)[:, 0][:4].tolist()]}")
print(f"  stats check row0: t_max={t_mx[0,0]:.4f} (torch {t1d.max():.4f})  d_max={d_mx[0,0]:.4f} (torch {d1d.max():.4f})")
print(f"  stats check row1 (bonus, early-return expected): d_max={d_mx[1,0]:.4f} (torch {d1d.max():.4f})")
# Cross-check the pipeline's effective h for the same token: acceptance rate
# over many trials with the SAME drafted token must equal min(p(777)/q(777),1).
torch.manual_seed(7)
tokens = torch.full((8192, 1), 777, dtype=torch.int64, device=DEVICE)
big = make_inputs(t1d, d1d, 1, 8192, draft_tokens=tokens)
blk = rejection_sample(**big, num_speculative_steps=1, use_block_verification=True)
h_emp = (blk[1] == 2).float().mean().item()
h_ref = min(p[777].item() / q[777].item(), 1.0)
print(f"  pipeline h(777): empirical={h_emp:.4f} vs theoretical={h_ref:.4f}")

print("=" * 30, "Exp A: per-token h readout (K=1, block)", "=" * 30)
# Cycle 32 fixed tokens; per-token acceptance rate estimates h(x) directly.
torch.manual_seed(0)
t1d = torch.randn(V, device=DEVICE)
d1d = torch.randn(V, device=DEVICE)
p, q = torch.softmax(t1d, 0), torch.softmax(d1d, 0)
S = torch.arange(0, 32, dtype=torch.int64, device=DEVICE) * 100 + 3
N = 16384
tokens = S.repeat(N // 32).unsqueeze(1)
inputs = make_inputs(t1d, d1d, 1, N, draft_tokens=tokens)
blk = rejection_sample(**inputs, num_speculative_steps=1, use_block_verification=True)
acc = (blk[1] == 2)
h_emp = acc.view(-1, 32).float().mean(0)
h_correct = (p / q)[S].clamp(max=1.0)
h_swap = (q / p)[S].clamp(max=1.0)
e_corr = (h_emp - h_correct).abs().mean().item()
e_swap = (h_emp - h_swap).abs().mean().item()
print(f"  mean|h_emp - min(p/q,1)| = {e_corr:.4f}   mean|h_emp - min(q/p,1)| = {e_swap:.4f}")
print(f"  h_emp  : {[round(v, 3) for v in h_emp[:8].tolist()]}")
print(f"  correct: {[round(v, 3) for v in h_correct[:8].tolist()]}")

print("=" * 30, "Exp B: all-rejected residual distribution (K=1)", "=" * 30)
# Disjoint near-one-hot p (token 300) and q (token 3800): acceptance ~ 0, so
# every position-0 token is resampled from max(p - q, 0)/Z ~ p. Any spread
# away from token 300 means the full-draft residual is broken.
torch.manual_seed(1)
t1d = torch.full((V,), -20.0, device=DEVICE)
t1d[300] = 20.0
d1d = torch.full((V,), -20.0, device=DEVICE)
d1d[3800] = 20.0
p, q = torch.softmax(t1d, 0), torch.softmax(d1d, 0)
N = 4096
inputs = make_inputs(t1d, d1d, 1, N)
for mode, kw in [("standard", {}), ("block", {"use_block_verification": True})]:
    out = rejection_sample(**inputs, num_speculative_steps=1, **kw)
    top = torch.bincount(out[0][:, 0], minlength=V).argmax().item()
    frac300 = (out[0][:, 0] == 300).float().mean().item()
    frac3800 = (out[0][:, 0] == 3800).float().mean().item()
    print(f"  [{mode}] argmax token={top}  frac@300={frac300:.4f}  frac@3800(excluded)={frac3800:.4f}")

print("=" * 30, "Exp C: K=3+2ph std-vs-blk diff breakdown", "=" * 30)
torch.manual_seed(0)
t1d = torch.randn(V, device=DEVICE)
d1d = torch.randn(V, device=DEVICE)
N, K, ph = 4096, 3, 2
inputs = make_inputs(t1d, d1d, K, N)
inputs["draft_sampled"].view(N, K + 1)[:, K + 1 - ph :] = -1
std = rejection_sample(**inputs, num_speculative_steps=K)
blk = rejection_sample(**inputs, num_speculative_steps=K, use_block_verification=True)
valid = torch.arange(K + 1, device=DEVICE).unsqueeze(0) < std[1].unsqueeze(1)
diff = (std[0] != blk[0]) & valid
rows = diff.any(1).nonzero().flatten()
print(f"  differing rows: {len(rows)} / {N}; by num_sampled: "
      f"{torch.bincount(std[1][rows], minlength=K + 2).tolist()}")
pos_hist = torch.bincount(diff.nonzero()[:, 1], minlength=K + 1)
print(f"  differing cells by position: {pos_hist.tolist()}")
for r in rows[:4].tolist():
    print(f"   row {r}: num_sampled={std[1][r].item()} std={std[0][r].tolist()} blk={blk[0][r].tolist()}")

print("=" * 30, "Exp E: pytest [True-1-1.0] replication + variants", "=" * 30)
torch.manual_seed(42)
t1d = torch.randn(V, device=DEVICE)
d1d = torch.randn(V, device=DEVICE)
p = torch.softmax(t1d, 0)


def build(n):
    return make_inputs(t1d, d1d, 1, n)


# (a) exact pytest conditions: 10*V trials in 10000-sized chunks
sampled_chunks = []
for inp in iter_chunks(build, 10 * V, 1):
    s, _ = rejection_sample(**inp, num_speculative_steps=1, use_block_verification=True)
    sampled_chunks.append(s)
blk_out = torch.cat(sampled_chunks)
chi2_label(blk_out[:, 0], p, "blk 10*V chunked (pytest config)")
# (b) single call at repro scale, both modes
small = build(4096)
s_std = rejection_sample(**small, num_speculative_steps=1)
chi2_label(s_std[:, 0], p, "std single-call N=4096")
s_blk = rejection_sample(**small, num_speculative_steps=1, use_block_verification=True)
chi2_label(s_blk[:, 0], p, "blk single-call N=4096")
print("  NOTE: resampled-only chi2 (rejected trials) isolates the residual path:")
rej_std = s_std[0][s_std[1] == 1][:, 0]
chi2_label(rej_std, p, "std resampled-only N~2048")
rej_blk = s_blk[0][s_blk[1] == 1][:, 0]
chi2_label(rej_blk, p, "blk resampled-only N~2048")
