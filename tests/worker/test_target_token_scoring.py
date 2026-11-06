"""Unit tests for the target-token-scoring compact fast path (Ascend).

These test the three concerns in isolation: the compact projector's numerical
equivalence to full-vocab candidate columns, the wave admission gate (one
rejection per condition, plus an eligible wave), and the compact sampler's
shape/stability/NaN contract. The runner integration itself is covered by the
existing sampler integration tests; this suite guards the math and the gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.worker.v1.target_token_scoring.admission import (
    AdmissionDecision,
    TargetTokenScoringState,
    evaluate_wave_admission,
)
from vllm_ascend.worker.v1.target_token_scoring.compact_sampler import compact_sample
from vllm_ascend.worker.v1.target_token_scoring.projector import (
    project_target_token_logits,
)


@pytest.fixture(autouse=True)
def _tts_env_on(monkeypatch):
    """Enable the fast path by default; tests that need it off unset locally."""
    monkeypatch.setenv("VLLM_ASCEND_TARGET_TOKEN_SCORING", "1")


def _make_model(vocab, hidden, *, bias=True, device="cpu"):
    weight = torch.randn(vocab, hidden, device=device)
    lm_head = SimpleNamespace(
        weight=weight,
        bias=torch.randn(vocab, device=device) if bias else None,
        tp_size=1,
        quant_method=None,
    )
    return SimpleNamespace(lm_head=lm_head, logits_processor=None)


def _decision(target_ids):
    return AdmissionDecision(ok=True, target_token_ids=list(target_ids))


# --------------------------------------------------------------------------- #
# Projector: compact[:, j] == full[:, target_ids[j]]
# --------------------------------------------------------------------------- #


def test_projector_matches_full_vocab_columns():
    torch.manual_seed(0)
    model = _make_model(vocab=50, hidden=8)
    hidden_states = torch.randn(4, 8)
    target_ids = [3, 5, 7, 11]
    out = project_target_token_logits(
        model, hidden_states, _decision(target_ids)
    )
    assert out is not None
    compact_logits, _state = out

    full = F.linear(hidden_states, model.lm_head.weight, model.lm_head.bias)
    expected = full[:, torch.tensor(target_ids)]
    assert torch.allclose(compact_logits, expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# Admission: one rejection per gate condition, plus an eligible wave
# --------------------------------------------------------------------------- #


def _sp(**over):
    # Base request carries logprob_token_ids and meets the greedy/single-token
    # contract; this is what admits the compact path.
    base = dict(
        logprob_token_ids=[3, 5],
        max_tokens=1,
        temperature=0,
        n=1,
        prompt_logprobs=None,
        logit_bias=None,
        allowed_token_ids=None,
        bad_words=[],
        guided_decoding=None,
        logits_processors=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _batch(req_ids):
    return SimpleNamespace(req_ids=list(req_ids), num_reqs=len(req_ids))


def _requests(req_ids, sp):
    return {rid: SimpleNamespace(sampling_params=sp) for rid in req_ids}


def _lm_head_eligible(vocab=100):
    return SimpleNamespace(
        weight=torch.randn(vocab, 4), tp_size=1, quant_method=None
    )


def _model_config():
    return SimpleNamespace(logits_processors=None)


def test_admission_eligible():
    ids = ["a", "b"]
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), _requests(ids, _sp()),
        _lm_head_eligible(), spec_decode_metadata=None,
    )
    assert d.ok and d.target_token_ids == [3, 5]


@pytest.mark.parametrize(
    "field,value,frag",
    [
        ("max_tokens", 2, "max_tokens"),
        ("temperature", 0.7, "temperature"),
        ("n", 2, "n="),
        ("prompt_logprobs", 1, "prompt_logprobs"),
        ("logit_bias", {1: 0.1}, "logit_bias"),
        ("allowed_token_ids", [1], "allowed_token_ids"),
        ("bad_words", ["x"], "bad_words"),
        ("guided_decoding", object(), "guided"),
        ("logits_processors", [object()], "logits_processors"),
        ("logprob_token_ids", None, "no logprob_token_ids"),
    ],
)
def test_admission_rejects(field, value, frag):
    ids = ["a"]
    sp = _sp(**{field: value})
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), _requests(ids, sp),
        _lm_head_eligible(), spec_decode_metadata=None,
    )
    assert not d.ok
    assert frag in (d.reason or "")


def test_admission_rejects_mismatched_candidate_ids():
    ids = ["a", "b"]
    requests = {
        "a": SimpleNamespace(sampling_params=_sp(logprob_token_ids=[3, 5])),
        "b": SimpleNamespace(sampling_params=_sp(logprob_token_ids=[3, 7])),
    }
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), requests,
        _lm_head_eligible(), spec_decode_metadata=None,
    )
    assert not d.ok and "differ" in d.reason


def test_admission_rejects_spec_decode_or_env_off_or_sharded_head(monkeypatch):
    ids = ["a"]
    # spec decode
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), _requests(ids, _sp()),
        _lm_head_eligible(), spec_decode_metadata=object(),
    )
    assert not d.ok and "speculative" in d.reason
    # env var off
    monkeypatch.delenv("VLLM_ASCEND_TARGET_TOKEN_SCORING", raising=False)
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), _requests(ids, _sp()),
        _lm_head_eligible(), spec_decode_metadata=None,
    )
    assert not d.ok and "env var" in d.reason
    # sharded head
    monkeypatch.setenv("VLLM_ASCEND_TARGET_TOKEN_SCORING", "1")
    head = _lm_head_eligible()
    head.tp_size = 2
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), _requests(ids, _sp()),
        head, spec_decode_metadata=None,
    )
    assert not d.ok and "sharded" in d.reason


# --------------------------------------------------------------------------- #
# Compact sampler: shape, stability, NaN
# --------------------------------------------------------------------------- #


def _state(target_ids, device="cpu"):
    return TargetTokenScoringState.from_ids(target_ids, torch.device(device))


def test_compact_sample_shape_and_logprobs_sum_to_one():
    torch.manual_seed(2)
    B, K = 5, 3
    logits = torch.randn(B, K)
    state = _state([10, 20, 30])
    out = compact_sample(logits, state, sampling_metadata=None)
    assert out.sampled_token_ids.shape == (B, 1)
    lt = out.logprobs_tensors
    assert lt is not None
    assert lt.logprob_token_ids.shape == (B, K + 1)
    assert lt.logprobs.shape == (B, K + 1)
    # candidate columns (1..K) are target-set logprobs -> exp sums to 1.
    probs = lt.logprobs[:, 1:].exp()
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)
    # sampled is the argmax candidate id; ranks 0 for the argmax.
    assert (lt.selected_token_ranks == 0).all()
    assert out.sampled_token_ids[:, 0].tolist() == [
        [10, 20, 30][int(i)] for i in logits.argmax(-1)
    ]


def test_compact_sample_overflow_stability():
    # Huge logits must not overflow the stable logsumexp.
    logits = torch.tensor([[1e30, -1e30, 0.0]])
    out = compact_sample(logits, _state([1, 2, 3]), sampling_metadata=None)
    lp = out.logprobs_tensors.logprobs[0, 1:]
    assert torch.isfinite(lp).all()
    assert torch.allclose(lp.exp().sum(), torch.tensor(1.0), atol=1e-4)


def test_compact_sample_nan_fail_closed():
    logits = torch.tensor([[0.1, float("nan"), 0.3]])
    out = compact_sample(logits, _state([1, 2, 3]), sampling_metadata=None)
    # NaN row's emitted logprobs must be NaN (not a silently-wrong finite pick).
    assert torch.isnan(out.logprobs_tensors.logprobs[0, 1:]).any()
    assert out.sampled_token_ids.shape == (1, 1)
