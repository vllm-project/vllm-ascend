# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Unit tests for reduced-vocab DSpark draft-logits computation in
``AscendSpecDecodeBaseProposer._run_merged_draft``.

Regression coverage for the Qwen3.6-35B-A3B + DSpark startup failure:

    aclnnInplaceAdd failed ... The tensor whose shape is [16, 32000] and the
    tensor whose shape is [16, 248320] do not meet the broadcast condition.

Reduced-vocab drafters (e.g. ``Qwen3DSparkForCausalLM`` with
``draft_vocab_size=32000`` vs target ``vocab_size=248320``) must:

1. compute base logits in the DRAFT vocab space via ``compute_draft_logits``
   so the Markov bias (``draft_vocab_size``-wide) can be added in place;
2. remap sampled draft ids back to target ids via ``map_draft_to_target``
   (only when ``draft_id_to_target_id`` exists), because the next Markov
   embedding indexes the TARGET vocab and the verifier consumes target ids.

This mirrors the GPU reference
``vllm/v1/worker/gpu/spec_decode/dspark/speculator.py``.

The tests drive the real ``_run_merged_draft`` method on CPU with a fake
drafter; ``parallel_drafting=True`` makes the method return
``draft_token_ids[:, 1:]`` right after the Markov loop so the produced draft
tokens can be asserted directly.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.spec_decode.llm_base_proposer as proposer_module
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# ---------------------------------------------------------------------------
# test geometry
# ---------------------------------------------------------------------------
_NUM_SPEC = 3  # num_speculative_tokens (K)
_NUM_BLK = 2  # number of draft blocks / requests (B)
_NUM_ROWS = _NUM_BLK * _NUM_SPEC  # sample_hidden_states rows = B * K
_HIDDEN = 4
_DRAFT_VOCAB = 8  # reduced draft vocab
_FULL_VOCAB = 16  # full (target-sized) vocab for legacy drafters
_SEEDS = [5, 7]

# d2t: target_id = draft_id + d2t[draft_id] = 100 + 11 * draft_id
_D2T = torch.tensor([100 + 10 * d for d in range(_DRAFT_VOCAB)], dtype=torch.long)


def _map_d2t(draft_ids: torch.Tensor) -> torch.Tensor:
    return draft_ids + _D2T[draft_ids]


# ---------------------------------------------------------------------------
# fake drafters
# ---------------------------------------------------------------------------


class _ReducedVocabDrafter:
    """Mimics ``Qwen3DSparkForCausalLM`` with draft_vocab_size < target vocab.

    Base draft logits are all zero, so each step's argmax is fully determined
    by the Markov bias, which peaks at ``prev_token % draft_vocab`` — making
    the whole autoregressive chain deterministic and exactly assertable.
    """

    draft_vocab_size = _DRAFT_VOCAB

    def __init__(self) -> None:
        self.draft_id_to_target_id = _D2T.clone()
        self.compute_draft_logits_calls = 0
        self.compute_logits_calls = 0
        self.markov_embed_inputs: list[torch.Tensor] = []

    def __call__(self, **kwargs) -> torch.Tensor:
        return torch.zeros(_NUM_ROWS, _HIDDEN)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.compute_draft_logits_calls += 1
        return torch.zeros(hidden_states.shape[0], _DRAFT_VOCAB)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.compute_logits_calls += 1
        raise AssertionError(
            "reduced-vocab drafter must use compute_draft_logits, not compute_logits"
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.markov_embed_inputs.append(token_ids.clone())
        return token_ids

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        bias = torch.zeros(markov_embed.shape[0], _DRAFT_VOCAB)
        bias[torch.arange(markov_embed.shape[0]), markov_embed % _DRAFT_VOCAB] = 10.0
        return bias

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return _map_d2t(draft_ids)


class _LegacyFullVocabDrafter:
    """Old-style drafter: NO ``compute_draft_logits``, ``draft_id_to_target_id``
    is None. The proposer must fall back to ``compute_logits`` and skip the
    d2t remap (pre-fix behaviour for full-vocab models)."""

    def __init__(self) -> None:
        self.draft_id_to_target_id = None
        self.compute_logits_calls = 0
        self.markov_embed_inputs: list[torch.Tensor] = []

    def __call__(self, **kwargs) -> torch.Tensor:
        return torch.zeros(_NUM_ROWS, _HIDDEN)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.compute_logits_calls += 1
        return torch.zeros(hidden_states.shape[0], _FULL_VOCAB)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.markov_embed_inputs.append(token_ids.clone())
        return token_ids

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        bias = torch.zeros(markov_embed.shape[0], _FULL_VOCAB)
        bias[torch.arange(markov_embed.shape[0]), markov_embed % _FULL_VOCAB] = 10.0
        return bias


class _FullVocabDrafterWithDraftLogits(_LegacyFullVocabDrafter):
    """New-interface drafter whose draft vocab EQUALS the target vocab:
    ``compute_draft_logits`` must be used, but with ``draft_id_to_target_id``
    None no remap may happen (identity path)."""

    def __init__(self) -> None:
        super().__init__()
        self.compute_draft_logits_calls = 0

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.compute_draft_logits_calls += 1
        return torch.zeros(hidden_states.shape[0], _FULL_VOCAB)


# ---------------------------------------------------------------------------
# proposer driver
# ---------------------------------------------------------------------------


def _make_proposer(model, monkeypatch) -> AscendSpecDecodeBaseProposer:
    """Build a minimal proposer instance (via __new__) wired for the dspark
    Markov-decoding branch of ``_run_merged_draft``."""
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.method = "dspark"
    proposer.num_speculative_tokens = _NUM_SPEC
    proposer.model = model
    proposer.parallel_drafting = True  # early-return draft_token_ids[:, 1:]
    # v0.28.0 dspark branch reads these directly:
    #   - _enable_probabilistic_draft_probs (probabilistic sampling toggle)
    #   - dynamic_spec (dynamic verify-length; None => disabled here)
    proposer._enable_probabilistic_draft_probs = False
    proposer.dynamic_spec = None
    # _run_merged_draft reads self.runner when sampling_metadata is not passed
    proposer.runner = None

    # buffers consumed by _run_merged_draft / _get_positions
    proposer.input_ids = torch.zeros(_NUM_ROWS, dtype=torch.int64)
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.positions = torch.arange(_NUM_ROWS, dtype=torch.int32)
    proposer._share_mtp_indices = False
    proposer._context_slot_mapping_buffers = MagicMock()
    proposer.build_model_inputs_first_pass = MagicMock()
    proposer.maybe_all_gather_and_unpad = (
        lambda last_hidden, positions, hidden=None: (last_hidden, positions, hidden)
    )

    # dspark persistent buffers: [max_batch, K + 1] and [max_batch]
    proposer._dspark_draft_buffer = torch.zeros((_NUM_BLK, _NUM_SPEC + 1), dtype=torch.int64)
    proposer._dspark_seed_buffer = torch.tensor(_SEEDS, dtype=torch.int64)

    # neutralise environment hooks around the patched code block
    monkeypatch.setattr(proposer_module, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(
        proposer_module,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_reduce_sample=False),
    )
    return proposer


def _run(proposer) -> torch.Tensor:
    return proposer._run_merged_draft(
        num_input_tokens=_NUM_ROWS,
        batch_size=_NUM_BLK,
        token_indices_to_sample=torch.arange(_NUM_ROWS, dtype=torch.int64),
        target_positions=None,
        inputs_embeds=None,
        multi_steps_attn_metadata=None,
        num_tokens=_NUM_ROWS,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestReducedVocabDrafter:
    """The fixed behaviour: draft-space logits + d2t remap per Markov step."""

    def test_uses_draft_space_and_remaps(self, monkeypatch):
        model = _ReducedVocabDrafter()
        proposer = _make_proposer(model, monkeypatch)

        draft_tokens = _run(proposer)

        # 1) base logits computed in DRAFT vocab space exactly once;
        #    compute_logits (target vocab) never touched.
        assert model.compute_draft_logits_calls == 1
        assert model.compute_logits_calls == 0

        # 2) Markov chain consumes TARGET-space ids: seeds first, then the
        #    remapped ids of the previous step (proves the remap happens
        #    before the next embedding, matching the GPU reference).
        assert len(model.markov_embed_inputs) == _NUM_SPEC
        assert torch.equal(model.markov_embed_inputs[0], torch.tensor([5, 7]))
        assert torch.equal(model.markov_embed_inputs[1], torch.tensor([155, 177]))
        assert torch.equal(model.markov_embed_inputs[2], torch.tensor([133, 111]))

        # 3) full deterministic chain:
        #    seed 5 -> draft 5 -> target 155; 155 % 8 = 3 -> target 133; ...
        #    seed 7 -> draft 7 -> target 177; 177 % 8 = 1 -> target 111; ...
        expected = torch.tensor(
            [[155, 133, 155], [177, 111, 177]], dtype=torch.int64
        )
        assert torch.equal(draft_tokens, expected)

    def test_bias_added_in_draft_space_shape(self, monkeypatch):
        """Shape guard: without the fix, logits are [B, target_vocab] while the
        Markov bias is [B, draft_vocab] and the in-place add broadcasts/fails
        (the original aclnnInplaceAdd crash). Here both must be draft-vocab
        wide, so the add is well-defined."""
        model = _ReducedVocabDrafter()
        proposer = _make_proposer(model, monkeypatch)

        observed: list[int] = []
        orig_bias = model.markov_bias

        def spy_bias(markov_embed):
            bias = orig_bias(markov_embed)
            observed.append(bias.shape[-1])
            return bias

        model.markov_bias = spy_bias
        _run(proposer)

        assert observed == [_DRAFT_VOCAB] * _NUM_SPEC
        # and the base logits the bias was added to are draft-vocab wide too
        assert model.compute_draft_logits_calls == 1


class TestBackwardCompatibility:
    """Full-vocab drafters keep working exactly as before the fix."""

    def test_legacy_drafter_falls_back_to_compute_logits(self, monkeypatch):
        model = _LegacyFullVocabDrafter()
        proposer = _make_proposer(model, monkeypatch)

        draft_tokens = _run(proposer)

        assert model.compute_logits_calls == 1
        # no compute_draft_logits attribute at all
        assert not hasattr(model, "compute_draft_logits")
        # no d2t table -> raw argmax ids stored, bias re-peaks at the same id
        expected = torch.tensor([[5, 5, 5], [7, 7, 7]], dtype=torch.int64)
        assert torch.equal(draft_tokens, expected)

    def test_full_vocab_draft_logits_without_mapping(self, monkeypatch):
        model = _FullVocabDrafterWithDraftLogits()
        proposer = _make_proposer(model, monkeypatch)

        draft_tokens = _run(proposer)

        # new interface is preferred when present ...
        assert model.compute_draft_logits_calls == 1
        assert model.compute_logits_calls == 0
        # ... but with draft_id_to_target_id=None the ids pass through unchanged
        expected = torch.tensor([[5, 5, 5], [7, 7, 7]], dtype=torch.int64)
        assert torch.equal(draft_tokens, expected)
        # embed saw the unmapped ids (identity remap)
        assert torch.equal(model.markov_embed_inputs[1], torch.tensor([5, 7]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

