# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash aux hidden-state capture semantics for GLM-5.3-Flash.

DFlash/DFlash2 drafters consume the target model's per-layer hidden states
contracted to a single stream. GLM-5.3-Flash is an mHC model: between decoder
layers flows the *deferred* hc_post state (raw FFN output + the n residual
streams + post/comb weights), so the completed output of layer k must be
materialized with hc_post before contracting. The required value equals
``hc_contract(hidden_states + residual)`` -- the semantics SGLang implements
for the same drafter (sgl-project/sglang#36708).
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu
from torch import nn
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

from vllm_ascend.models.glm5next.model import (
    Glm5NextDecoderLayer,
    Glm5NextModel,
)


def _make_model(mhc: bool) -> Glm5NextModel:
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=mhc, mhc_num_residual_streams=4)
    model.is_sequence_parallel = False
    model.aux_hidden_state_layers = ()
    return model


def _make_layer() -> Glm5NextDecoderLayer:
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.n = 4
    return layer


def test_capture_contracts_mhc_bundle_to_single_stream(monkeypatch):
    """The captured state is the stream-mean of layer k's completed output."""

    def native_post(x, residual, post, comb):
        assert post.ndim == 3 and residual.ndim == 4
        return mhc_post_torch(x, residual, post.unsqueeze(-1), comb)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_hc_post", native_post, raising=False)

    model = _make_model(mhc=True)
    layer = _make_layer()

    torch.manual_seed(0)
    # State at the top of layer k+1, as returned by layer k.
    hidden_states = torch.randn(2, 8).bfloat16()  # raw FFN output of layer k
    residual = torch.randn(2, 4, 8).bfloat16()  # streams entering layer k's FFN
    post = torch.rand(2, 4, 1).float()
    comb = torch.rand(2, 4, 4).float()

    actual = model._capture_completed_layer_output(layer, hidden_states, residual, post, comb, full_num_tokens=2)

    bundle = mhc_post_torch(hidden_states, residual, post, comb)
    expected = bundle.mean(dim=1)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 8)


def test_capture_keeps_non_mhc_output():
    """Non-mHC layers return the completed single-stream output directly."""
    model = _make_model(mhc=False)
    layer = _make_layer()
    hidden_states = torch.randn(2, 8).bfloat16()

    actual = model._capture_completed_layer_output(layer, hidden_states, None, None, None, full_num_tokens=2)

    torch.testing.assert_close(actual, hidden_states)


def test_supports_eagle3_sets_aux_layers_on_inner_model():
    from vllm.model_executor.models.interfaces import SupportsEagle3

    from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM

    target = Glm5NextForCausalLM.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(target)
    inner = _make_model(mhc=True)
    target.model = inner

    assert isinstance(target, SupportsEagle3)
    # DFlash's target_layer_ids [5, 14, 24, 33, 42] arrive as capture
    # points [6, 15, 25, 34, 43] (see get_eagle3_aux_layers_from_config).
    target.set_aux_hidden_state_layers((6, 15, 25, 34, 43))
    assert inner.aux_hidden_state_layers == (6, 15, 25, 34, 43)


def test_conditional_generation_unwraps_language_model():
    from vllm_ascend.models.glm5next.model import (
        Glm5NextForCausalLM,
        Glm5NextForConditionalGeneration,
    )

    wrapper = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(wrapper)
    target = Glm5NextForCausalLM.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(target)
    inner = _make_model(mhc=True)
    target.model = inner
    wrapper.language_model = target

    assert wrapper.get_language_model() is target
    wrapper.set_aux_hidden_state_layers((6, 15, 25, 34, 43))
    assert inner.aux_hidden_state_layers == (6, 15, 25, 34, 43)


if __name__ == "__main__":
    pytest.main([__file__])
