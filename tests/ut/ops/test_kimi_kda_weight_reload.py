# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real startup tracking and deferred reload of Kimi's derived convolution."""

from inspect import signature
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from vllm_ascend.ops.kimi_kda import _PACKED_CONV_WEIGHT_NAME, AscendKimiK3DeltaAttention


@pytest.mark.parametrize(
    "defer_processing",
    [False, True] if "defer_processing" in signature(initialize_layerwise_reload).parameters else [False],
)
def test_checkpoint_load_and_two_reloads_preserve_derived_buffer(defer_processing):
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.model_config = SimpleNamespace(dtype=torch.bfloat16)
    attention.register_buffer(_PACKED_CONV_WEIGHT_NAME, None, persistent=False)
    attention.A_log = nn.Parameter(torch.zeros(2))
    attention.dt_bias = nn.Parameter(torch.zeros(6))
    attention.conv1d = nn.Module()
    initial = torch.arange(18 * 4, dtype=torch.float32).reshape(18, 1, 4)
    attention.conv1d.weight = nn.Parameter(initial.clone())
    model = nn.Module()
    model.add_module("kda", attention)
    config = SimpleNamespace(dtype=torch.bfloat16, quantization=None, word_embeddings_untied_by_checkpoint=False)

    # initialize_model records checkpoint-shaped metadata before post-processing.
    record_metadata_for_reloading(model)
    with torch.inference_mode():
        process_weights_after_loading(model, config, torch.device("cpu"))
    captured = attention.get_buffer(_PACKED_CONV_WEIGHT_NAME)
    assert captured is not None
    pointer = captured.data_ptr()
    expected = initial[:, 0, :].T.to(torch.bfloat16)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    assert "kda." + _PACKED_CONV_WEIGHT_NAME not in dict(model.named_parameters())
    assert "kda." + _PACKED_CONV_WEIGHT_NAME not in model.state_dict()
    assert dict(model.named_buffers())["kda." + _PACKED_CONV_WEIGHT_NAME] is captured
    # A real startup loader must accept a checkpoint containing only source weights.
    loader = DefaultModelLoader.__new__(DefaultModelLoader)
    loader.track_weights_loading(model, set(dict(model.named_parameters())))

    for offset in (10.0, 20.0):
        if defer_processing:
            initialize_layerwise_reload(model, defer_processing=True)
        else:
            initialize_layerwise_reload(model)
        assert attention.A_log.is_meta
        # Load direct KDA state first. Packing must wait for child finalization.
        for name, parameter in list(model.named_parameters()):
            value = initial + offset if name.endswith("conv1d.weight") else torch.full(parameter.shape, offset)
            parameter.weight_loader(parameter, value)
        finalize_layerwise_reload(model, config)
        packed = attention.get_buffer(_PACKED_CONV_WEIGHT_NAME)
        assert packed is captured
        assert packed.data_ptr() == pointer
        assert packed.dtype == torch.bfloat16
        assert packed.is_contiguous()
        torch.testing.assert_close(packed, (initial[:, 0, :] + offset).T.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(attention.A_log, torch.full((2,), offset))
        torch.testing.assert_close(attention.dt_bias, torch.full((6,), offset))
