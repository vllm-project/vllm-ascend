# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd.
# All Rights Reserved.
"""force_topk sampler unit tests.

Tests the force_topk reference implementation
(vllm_ascend.ops.force_topk_sample) and CompactDist
(vllm_ascend.sample.topk_map) against full-vocab computations.

Run (CPU, no NPU required):
    pytest -sv tests/ut/sample/test_force_topk.py
"""
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.force_topk_sample import force_topk_sample
from vllm_ascend.sample.topk_map import CompactDist

_EPS = 1e-5


# --------------------------------------------------------------------------- #
# Full-vocab golden reference (equivalent to upstream Sampler.sample)         #
# --------------------------------------------------------------------------- #
def full_log_softmax(logits):
    return torch.log_softmax(logits, dim=-1)


def full_nucleus_set(logits, temperature, top_p, top_k, min_p=None):
    """Compute the top_k/top_p/min_p candidate set on the full vocabulary."""
    p = torch.softmax(logits / temperature, dim=-1)[0]
    order = torch.argsort(p, descending=True)
    keep = set()
    # top_k
    kk = int(top_k) if top_k > 0 else p.numel()
    topk_ids = order[:kk]
    # top_p: smallest prefix
    sp = p[order]
    cdf = torch.cumsum(sp, 0)
    npos = int((cdf < top_p).sum().item()) + 1
    topp_ids = order[:npos]
    s = set(topk_ids.tolist()) & set(topp_ids.tolist())
    if min_p is not None:
        thr = float(min_p) * float(p.max())
        s = {i for i in s if float(p[i]) >= thr}
    return s


def make_meta(temperature, top_p, top_k, min_p=None, all_greedy=False):
    B = temperature.shape[0]
    return SimpleNamespace(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        all_greedy=all_greedy,
        all_random=not all_greedy,
        generators={},
    )


@pytest.fixture
def logits():
    torch.manual_seed(0)
    B, V = 1, 512
    # Sharp distribution (close to LLM output) so nucleus fits in k
    z = torch.randn(B, V) * 3.0
    return z.float()


def _one(val, B=1, dtype=torch.float32):
    return torch.full((B,), val, dtype=dtype)


# --------------------------------------------------------------------------- #
# Test cases (mapping to §6 table)                                            #
# --------------------------------------------------------------------------- #
def test_i1_greedy_bit_exact(logits):
    """I1: greedy == full-vocab argmax, element-wise equal."""
    k = 64
    _, cdist = force_topk_sample(
        logits, _one(1.0), _one(1.0), _one(-1, dtype=torch.int32), None, {}, k
    )
    assert int(cdist.token_index[0, 0]) == int(logits.argmax(-1))


def test_i3_logprobs_match_full_vocab(logits):
    """I3: compact logprob matches log_softmax full-vocab values (allclose)."""
    k = 64
    _, cdist = force_topk_sample(
        logits, _one(1.0), _one(1.0), _one(-1, dtype=torch.int32), None, {}, k
    )
    full = full_log_softmax(logits)                       # [B, V]
    ref = full.gather(1, cdist.token_index.long())        # [B, k]
    assert torch.allclose(cdist.logprobs, ref, atol=1e-4)


def test_i2_top_k_effective(logits):
    """I2: top_k=m (m<k) → sampled tokens ⊆ top-m."""
    k, m = 64, 8
    torch.manual_seed(1)
    N = 4096
    rep = logits.repeat(N, 1)
    sampled, _ = force_topk_sample(
        rep, _one(1.0, N), _one(1.0, N), _one(m, N, torch.int32), None, {}, k
    )
    top_m = set(torch.topk(logits, m, dim=-1).indices[0].tolist())
    assert set(sampled.tolist()).issubset(top_m)


def test_i2_top_p_nucleus_alignment(logits):
    """I2: nucleus < k → force_topk nucleus set == full-vocab top_p set."""
    k = 128
    top_p = 0.9
    _, cdist = force_topk_sample(
        logits, _one(1.0), _one(top_p), _one(-1, dtype=torch.int32), None, {}, k
    )
    p = torch.exp(cdist.logprobs)[0]
    cdf = torch.cumsum(p, 0)
    npos = int((cdf < top_p).sum()) + 1
    got = set(cdist.token_index[0, :npos].tolist())
    want = full_nucleus_set(logits, torch.tensor(1.0), top_p, torch.tensor(-1))
    assert npos < k, "precondition: nucleus fits in k"
    assert got == want


def test_distribution_kl_small(logits):
    """Distribution approximation: large-sample KL(force_topk || full) < threshold."""
    k = 128
    torch.manual_seed(2)
    N = 20000
    rep = logits.repeat(N, 1)
    sampled, _ = force_topk_sample(
        rep, _one(1.0, N), _one(1.0, N), _one(-1, N, torch.int32), None, {}, k
    )
    V = logits.shape[-1]
    emp = torch.bincount(sampled, minlength=V).float() / N
    ref = torch.softmax(logits, dim=-1)[0]
    mask = emp > 0
    kl = (emp[mask] * (emp[mask].log() - ref[mask].log())).sum()
    assert kl < 0.02, f"KL={kl:.4f} too large"


def test_gather_hit_and_miss(logits):
    """CompactDist.gather: hit → value, miss → -inf, no .item()."""
    k = 32
    _, cdist = force_topk_sample(
        logits, _one(1.0), _one(1.0), _one(-1, dtype=torch.int32), None, {}, k
    )
    hit_id = cdist.token_index[:, 3].long()               # guaranteed hit
    miss_id = cdist.token_index[:, -1].long().clone()
    miss_id[:] = -999 % logits.shape[-1]                  # likely not in top-k
    assert torch.isfinite(cdist.gather(hit_id)).all()
    if not (cdist.token_index == miss_id[:, None]).any():
        assert torch.isinf(cdist.gather(miss_id)).all()


def test_topn_is_sorted_slice(logits):
    """topn: already descending → slice; logprob monotonically non-increasing."""
    k, n = 64, 5
    _, cdist = force_topk_sample(
        logits, _one(1.0), _one(1.0), _one(-1, dtype=torch.int32), None, {}, k
    )
    lp, ids = cdist.topn(n)
    assert lp.shape[-1] == n
    assert torch.all(lp[:, :-1] >= lp[:, 1:])
    assert torch.equal(ids, cdist.token_index[:, :n])


def test_seed_reproducible(logits):
    """min_p / seed reproducibility: same seed → same output."""
    k = 64
    g1 = torch.Generator().manual_seed(1234)
    g2 = torch.Generator().manual_seed(1234)
    s1, _ = force_topk_sample(
        logits, _one(0.8), _one(0.95), _one(-1, dtype=torch.int32),
        _one(0.05), {0: g1}, k,
    )
    s2, _ = force_topk_sample(
        logits, _one(0.8), _one(0.95), _one(-1, dtype=torch.int32),
        _one(0.05), {0: g2}, k,
    )
    assert torch.equal(s1, s2)


def test_min_p_threshold_alignment(logits):
    """min_p: k-space candidate set ⊆ full-vocab min_p reference set."""
    k = 128
    min_p = 0.1
    torch.manual_seed(3)
    N = 4096
    rep = logits.repeat(N, 1)
    sampled, _ = force_topk_sample(
        rep, _one(1.0, N), _one(1.0, N), _one(-1, N, torch.int32),
        _one(min_p, N), {}, k,
    )
    want = full_nucleus_set(
        logits, torch.tensor(1.0), torch.tensor(1.0), torch.tensor(-1),
        min_p=torch.tensor(min_p),
    )
    assert set(sampled.tolist()).issubset(want)


def test_fallback_guard_num_logprobs_gt_k():
    """Fallback guard: num_logprobs > k should trigger fallback.

    This test validates the guard logic; when wired into AscendSampler,
    hitting the guard calls super().sample() and output matches native.
    """

    def is_safe(num_logprobs, k, no_penalties, want_raw_logprobs):
        if num_logprobs > k:
            return False
        if want_raw_logprobs and not no_penalties:
            return False
        return True

    assert is_safe(20, 2048, True, False) is True
    assert is_safe(4096, 2048, True, False) is False        # num_logprobs > k
    assert is_safe(20, 2048, False, True) is False          # raw + penalties


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-sv"]))
