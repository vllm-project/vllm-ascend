# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for checkpoint alignment, with only vLLM dependencies mocked."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[3]


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, _ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def domino_modules(monkeypatch):
    correction = _load_file("alignment_correction", "vllm_ascend/spec_decode/domino_correction.py")

    class DflashProposer:
        def __init__(self, vllm_config, device, runner=None):
            self.speculative_config = vllm_config.speculative_config
            self.num_speculative_tokens = 4

        def load_model(self, model):
            self.model = model

        def set_inputs_first_pass(self, **kwargs):
            return self.base_result

    class DflashModel(nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.config = vllm_config

    dependencies = {
        "vllm.logger": {"logger": Mock()},
        "vllm.config": {"VllmConfig": object},
        "vllm_ascend.utils": {"lmhead_tp_enable": lambda: False},
        "vllm.v1.attention.backends.utils": {"CommonAttentionMetadata": object},
        "vllm.model_executor.model_loader.weight_utils": {"default_weight_loader": Mock()},
        "vllm.model_executor.models.qwen3_dflash": {"DFlashQwen3ForCausalLM": DflashModel},
        "vllm_ascend.spec_decode.dflash_proposer": {"AscendDflashProposer": DflashProposer},
    }
    for name, attrs in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.domino_correction", correction)
    proposer = _load_file("alignment_proposer", "vllm_ascend/spec_decode/domino_proposer.py")
    model = _load_file("alignment_model", "vllm_ascend/models/qwen3_domino.py")
    return proposer.AscendDominoProposer, model.DominoQwen3ForCausalLM, correction


@pytest.mark.parametrize("shift_label", [True, False])
def test_model_accepts_boolean_alignment(domino_modules, shift_label):
    _, model_cls, _ = domino_modules
    config = SimpleNamespace(
        hidden_size=8,
        vocab_size=64,
        dflash_config={"projector_type": "domino", "emb_dim": 4, "shift_label": shift_label},
    )
    assert model_cls(vllm_config=config).shift_label is shift_label


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", "true"])
def test_rejects_non_boolean_alignment(domino_modules, invalid):
    proposer_cls, model_cls, _ = domino_modules
    config = SimpleNamespace(
        dflash_config={"projector_type": "domino", "shift_label": invalid},
    )
    with pytest.raises(ValueError, match="boolean"):
        model_cls(vllm_config=config)
    proposer = object.__new__(proposer_cls)
    with pytest.raises(ValueError, match="boolean"):
        proposer.load_model(SimpleNamespace(shift_label=invalid))


def _prepare(proposer, bonus):
    return proposer.set_inputs_first_pass(
        target_token_ids=bonus,
        next_token_ids=bonus,
        target_positions=torch.zeros_like(bonus),
        target_hidden_states=torch.empty(0),
        token_indices_to_sample=None,
        cad=proposer.base_result[2],
        num_rejected_tokens_gpu=None,
    )


@pytest.mark.parametrize("shift_label", [True, False])
@pytest.mark.parametrize("batch_size,k", [(1, 1), (1, 15), (3, 4)])
@pytest.mark.parametrize("prefix_len", [0, 1])
@pytest.mark.parametrize("candidate_topk", [0, 64])
def test_alignment_and_correction_rollout(
    domino_modules, monkeypatch, shift_label, batch_size, k, prefix_len, candidate_topk
):
    proposer_cls, _, correction = domino_modules
    helpers = _load_file("alignment_test_helpers", "tests/ut/spec_decode/test_domino_correction.py")
    torch.manual_seed(72)
    model = helpers.StubDominoModel(vocab_size=96, pure_draft_prefix_len=prefix_len).eval()
    model.shift_label = shift_label
    model.model = SimpleNamespace(embed_tokens=model.embed_tokens)
    proposer = object.__new__(proposer_cls)
    proposer.domino_candidate_topk = candidate_topk
    proposer.num_speculative_tokens = k
    # Weight precomputation is covered by correction tests, not this CPU model stub.
    monkeypatch.setattr(correction, "_get_domino_correction_weights", Mock())
    proposer.load_model(model)
    proposer._domino_bonus_buf = torch.zeros(batch_size + 2, dtype=torch.long)
    bonus_ptr = proposer._domino_bonus_buf.data_ptr()
    # Include gaps between requests to catch accidental contiguous-block assumptions.
    starts = torch.arange(batch_size) * (k + 3) + 2
    mask_indices = (starts[:, None] + torch.arange(1, k + 1)).flatten().int()
    original_indices = mask_indices.clone()
    cad = object()
    proposer.base_result = (batch_size * (k + 1), mask_indices, cad, None)
    all_hidden = torch.randn(batch_size * (k + 3) + 2, model.hidden_size)

    # Two rounds also exercise fixed-buffer refresh, as required by graph replay.
    for _ in range(2):
        bonus = torch.randint(0, model.vocab_size, (batch_size,))
        count, selected, returned_cad, extra = _prepare(proposer, bonus)
        expected_indices = (starts[:, None] + torch.arange(k) + (0 if shift_label else 1)).flatten()
        assert torch.equal(selected, expected_indices)
        assert torch.equal(mask_indices, original_indices)
        assert count == batch_size * (k + 1)
        assert returned_cad is cad and extra is None
        assert proposer._domino_bonus_buf.data_ptr() == bonus_ptr
        assert torch.equal(proposer._domino_bonus_buf[:batch_size], bonus)

        # Independent official-style block slice, before recurrent correction.
        blocks = torch.stack([all_hidden[start : start + k + 1] for start in starts])
        z = blocks[:, :k] if shift_label else blocks[:, 1:]
        with torch.inference_mode():
            expected = correction.domino_correct(
                z.reshape(-1, model.hidden_size), k, model, bonus, candidate_topk=candidate_topk
            )
            # Exercise both proposer dispatches without requiring Triton/NPU.
            for fallback in (False, True):

                def run_fused(fallback=fallback, **kwargs):
                    if fallback:
                        raise ImportError("CPU test")
                    return correction.domino_correct(**kwargs)

                monkeypatch.setitem(proposer_cls.compute_draft_token_ids.__globals__, "domino_correct_fused", run_fused)
                got = proposer.compute_draft_token_ids(all_hidden[selected.long()])
                assert got.shape == (batch_size * k,)
                assert torch.equal(got, expected)


def test_unvalidated_alignment_cannot_propose(domino_modules):
    proposer_cls, _, _ = domino_modules
    proposer = object.__new__(proposer_cls)
    proposer._domino_shift_label = None
    proposer.base_result = (2, torch.tensor([1]), object(), None)
    with pytest.raises(RuntimeError, match="validated boolean"):
        _prepare(proposer, torch.tensor([3]))
