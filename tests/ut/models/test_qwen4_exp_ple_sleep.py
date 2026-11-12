# SPDX-License-Identifier: Apache-2.0
"""Regressions for PLE constants in the level-2 sleep buffer lifecycle."""

from unittest.mock import patch

import pytest
import torch
from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState

from vllm_ascend.models.qwen4_exp.model_state import AscendQwen4ExpModelState


def make_state(model, *, enabled=True, context_len=2):
    def initialize(state, *_args):
        state.uses_ngram_embedding = enabled
        if enabled:
            state.ngram_context_offsets = torch.arange(-context_len, 0, dtype=torch.int64)

    # Isolate model-state allocation from unrelated runner/configuration setup.
    with patch.object(Qwen4ExpModelState, "__init__", initialize):
        return AscendQwen4ExpModelState(None, model, None, torch.device("cpu"))


@pytest.mark.parametrize("context_len", [1, 2, 4])
def test_ple_offsets_survive_repeated_level2_buffer_restore(context_len):
    model = torch.nn.Module()
    state = make_state(model, context_len=context_len)
    expected = torch.arange(-context_len, 0, dtype=torch.int64)
    eos_token, first_token = 248044, 760
    for _ in range(2):
        # Match NPUWorker's named-buffer backup/restore around discarded HBM.
        backup = {name: value.cpu().clone() for name, value in model.named_buffers()}
        state.ngram_context_offsets.zero_()
        for name, value in model.named_buffers():
            value.copy_(backup[name])
        assert torch.equal(state.ngram_context_offsets, expected)
        context = torch.where(state.ngram_context_offsets < 0, eos_token, first_token)
        assert context.tolist() == [eos_token] * context_len
    assert model._qwen4_exp_ngram_context_offsets is state.ngram_context_offsets
    assert "_qwen4_exp_ngram_context_offsets" not in model.state_dict()


def test_disabled_ple_does_not_require_or_register_offsets():
    model = torch.nn.Module()
    state = make_state(model, enabled=False)
    assert not hasattr(state, "ngram_context_offsets")
    assert not dict(model.named_buffers())


def test_recreated_state_replaces_the_model_buffer_alias():
    model = torch.nn.Module()
    old_state = make_state(model, context_len=2)
    new_state = make_state(model, context_len=4)
    assert model._qwen4_exp_ngram_context_offsets is new_state.ngram_context_offsets
    assert model._qwen4_exp_ngram_context_offsets is not old_state.ngram_context_offsets
