# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable unit tests for the Domino correction rollout.

``vllm_ascend.utils`` hard-imports ``torch_npu`` at module load, so importing the
``vllm_ascend`` package fails on a CPU-only machine. The correction logic lives
in ``vllm_ascend/spec_decode/domino_correction.py`` which imports only torch, so
we load that file directly (via importlib) to test it on CPU. On an NPU machine
the normal package import also works.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "vllm_ascend" / "spec_decode" / "domino_correction.py"


def _load_domino_correct():
    try:
        from vllm_ascend.spec_decode.domino_correction import domino_correct  # type: ignore
        return domino_correct
    except Exception:
        spec = importlib.util.spec_from_file_location(
            "vllm_ascend_domino_correction", _MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.domino_correct


class _LogitsProcessor:
    """Minimal stand-in for vLLM's LogitsProcessor: lm_head(h) with a scale."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, lm_head: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        return lm_head(hidden_states) * self.scale


class StubDominoModel(nn.Module):
    """Duck-typed Domino draft model for testing domino_correct.

    Exposes the same surface the proposer relies on: ``lm_head``,
    ``logits_processor``, ``embed_input_ids``, ``embed_proj``, ``_run_prefix_gru``
    and ``pure_draft_prefix_len``.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 16,
        gru_hidden_dim: int = 8,
        emb_dim: int = 12,
        pure_draft_prefix_len: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.gru_hidden_dim = gru_hidden_dim
        self.pure_draft_prefix_len = pure_draft_prefix_len
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.logits_processor = _LogitsProcessor()
        self.prefix_gru = nn.GRU(
            hidden_size, gru_hidden_dim, num_layers=1, batch_first=True, bias=False
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size + gru_hidden_dim, emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(emb_dim, vocab_size, bias=False),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _run_prefix_gru(self, embeds, gru_hidden=None):
        # CPU test: no fp16 cast (NPU's DynamicGRUV2 fp16 constraint does not
        # apply here). The proposer's real helper casts to the GRU param dtype.
        if gru_hidden is not None:
            return self.prefix_gru(embeds, gru_hidden)
        return self.prefix_gru(embeds)


def _reference_domino_correct(
    hidden_states: torch.Tensor,
    k: int,
    model: StubDominoModel,
    bonus_token_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Independent reimplementation of the Domino correction for cross-checking."""
    num_indices = hidden_states.shape[0]
    batch_size = num_indices // k
    h = hidden_states.view(batch_size, k, -1)
    base_logits = model.lm_head(hidden_states).view(batch_size, k, -1)
    prefix_len = model.pure_draft_prefix_len

    out = torch.empty(batch_size, k, dtype=torch.long, device=hidden_states.device)
    if prefix_len > 0:
        prefix_tokens = base_logits[:, :prefix_len, :].argmax(dim=-1)
        out[:, :prefix_len] = prefix_tokens
    else:
        prefix_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=hidden_states.device)

    if prefix_len >= k:
        return out.view(-1)

    if bonus_token_ids is None or bonus_token_ids.shape[0] < batch_size:
        bonus = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
    else:
        bonus = bonus_token_ids[:batch_size]
    seed = torch.cat([bonus.view(batch_size, 1), prefix_tokens], dim=1)
    _, state = model._run_prefix_gru(model.embed_tokens(seed))  # [1, B, gru]

    for i in range(prefix_len, k):
        z = h[:, i, :]
        s = state.squeeze(0).to(z.dtype)
        bias = model.embed_proj(torch.cat([z, s], dim=-1))
        tok = (base_logits[:, i, :] + bias).argmax(dim=-1)
        out[:, i] = tok
        if i + 1 < k:
            _, state = model._run_prefix_gru(model.embed_tokens(tok).unsqueeze(1), state)
    return out.view(-1)


@pytest.fixture
def deterministic():
    torch.manual_seed(1234)


def test_domino_correct_matches_reference(deterministic):
    domino_correct = _load_domino_correct()
    k = 5
    batch_size = 3
    model = StubDominoModel(vocab_size=64, hidden_size=16, gru_hidden_dim=8, emb_dim=12,
                            pure_draft_prefix_len=1)
    model.eval()
    hidden = torch.randn(batch_size * k, model.hidden_size, dtype=torch.float32)
    bonus = torch.randint(0, model.vocab_size, (batch_size,))

    with torch.no_grad():
        got = domino_correct(hidden, k, model, bonus)
        ref = _reference_domino_correct(hidden, k, model, bonus)

    assert got.shape == (batch_size * k,)
    assert got.dtype == torch.long
    assert torch.equal(got, ref)


def test_domino_correct_batched(deterministic):
    """Multiple requests advance their GRU independently and in lockstep."""
    domino_correct = _load_domino_correct()
    k = 4
    model = StubDominoModel(pure_draft_prefix_len=1)
    model.eval()
    batch_size = 6
    hidden = torch.randn(batch_size * k, model.hidden_size, dtype=torch.float32)
    bonus = torch.randint(0, model.vocab_size, (batch_size,))

    with torch.no_grad():
        got = domino_correct(hidden, k, model, bonus)
        ref = _reference_domino_correct(hidden, k, model, bonus)

    assert torch.equal(got, ref)
    # Each request's tokens are independent: per-request comparison matches.
    got_b = got.view(batch_size, k)
    ref_b = ref.view(batch_size, k)
    assert torch.equal(got_b, ref_b)


def test_domino_correct_zero_prefix(deterministic):
    """pure_draft_prefix_len=0: GRU seeded with the bonus token only."""
    domino_correct = _load_domino_correct()
    k = 4
    model = StubDominoModel(pure_draft_prefix_len=0)
    model.eval()
    batch_size = 2
    hidden = torch.randn(batch_size * k, model.hidden_size, dtype=torch.float32)
    bonus = torch.randint(0, model.vocab_size, (batch_size,))

    with torch.no_grad():
        got = domino_correct(hidden, k, model, bonus)
        ref = _reference_domino_correct(hidden, k, model, bonus)

    assert torch.equal(got, ref)


def test_domino_correct_dummy_bonus(deterministic):
    """bonus_token_ids=None (dummy_run): still produces correct shape/dtype."""
    domino_correct = _load_domino_correct()
    k = 3
    model = StubDominoModel(pure_draft_prefix_len=1)
    model.eval()
    batch_size = 2
    hidden = torch.randn(batch_size * k, model.hidden_size, dtype=torch.float32)

    with torch.no_grad():
        got = domino_correct(hidden, k, model, None)
        ref = _reference_domino_correct(hidden, k, model, None)

    assert got.shape == (batch_size * k,)
    assert got.dtype == torch.long
    assert torch.equal(got, ref)


def test_domino_correct_prefix_ge_k(deterministic):
    """pure_draft_prefix_len >= k: no correction, pure greedy from base_logits."""
    domino_correct = _load_domino_correct()
    k = 3
    model = StubDominoModel(pure_draft_prefix_len=3)  # == k
    model.eval()
    batch_size = 2
    hidden = torch.randn(batch_size * k, model.hidden_size, dtype=torch.float32)
    bonus = torch.randint(0, model.vocab_size, (batch_size,))

    with torch.no_grad():
        got = domino_correct(hidden, k, model, bonus)

    base_logits = model.lm_head(hidden).view(batch_size, k, -1)
    expected = base_logits.argmax(dim=-1).view(-1)
    assert torch.equal(got, expected)


def test_domino_correct_fp16_gru(deterministic):
    """The correction works with an fp16 GRU (NPU casts prefix_gru to fp16)."""
    domino_correct = _load_domino_correct()
    k = 4
    model = StubDominoModel(pure_draft_prefix_len=1)
    model.prefix_gru = model.prefix_gru.to(torch.float16)
    model.eval()
    batch_size = 2
    # hidden stays fp32 (bf16 stand-in); _run_prefix_gru casts embeds to the GRU
    # param dtype (fp16) on the real model. The stub does not cast, so feed the
    # GRU-compatible path by keeping embeds fp16 here.
    hidden = torch.randn(batch_size * k, model.hidden_size, dtype=torch.float32)
    bonus = torch.randint(0, model.vocab_size, (batch_size,))

    # The real _run_prefix_gru casts inputs to param dtype; emulate that so the
    # fp16 GRU receives fp16 inputs.
    real_run = model._run_prefix_gru

    def cast_run(embeds, gru_hidden=None):
        embeds = embeds.to(model.prefix_gru.weight_ih_l0.dtype)
        if gru_hidden is not None:
            gru_hidden = gru_hidden.to(model.prefix_gru.weight_ih_l0.dtype)
        return real_run(embeds, gru_hidden)

    model._run_prefix_gru = cast_run

    with torch.no_grad():
        got = domino_correct(hidden, k, model, bonus)
        ref = _reference_domino_correct(hidden, k, model, bonus)

    assert got.shape == (batch_size * k,)
    assert torch.equal(got, ref)


def test_domino_correct_fused_runs_on_npu(deterministic):
    """Fused path (triton kernels + precompute) runs on NPU in bf16.

    Skipped on CPU (no triton-ascend). Numerical equivalence vs the eager path
    is verified end-to-end by the E2E acceptance test (same baseline), since the
    fused and eager GRU implementations differ slightly at the dtype level.
    """
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("fused path targets NPU triton-ascend")
    try:
        from vllm_ascend.ops.triton.spec_decode.domino import fused_gru_cell  # noqa: F401
    except Exception:
        pytest.skip("triton-ascend fused kernels not importable")

    device = torch.device("npu", 0)
    from vllm_ascend.spec_decode.domino_correction import domino_correct_fused

    k = 4
    batch_size = 2
    model = StubDominoModel(pure_draft_prefix_len=1)
    model = model.to(device).to(torch.bfloat16).eval()
    hidden = torch.randn(
        batch_size * k, model.hidden_size, dtype=torch.bfloat16, device=device
    )
    bonus = torch.randint(0, model.vocab_size, (batch_size,), device=device)

    with torch.no_grad():
        out = domino_correct_fused(hidden, k, model, bonus)

    assert out.shape == (batch_size * k,)
    assert out.dtype == torch.long
