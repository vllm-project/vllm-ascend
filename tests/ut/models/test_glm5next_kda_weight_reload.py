# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real deferred-loading protocol and graph-owned KDA cache."""

from types import SimpleNamespace

import torch
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention


def _make_layer():
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.register_buffer("_merged_conv_weight", None, persistent=False)
    # Direct KDA parameters and child projections follow the same ownership
    # split as the complete layer, without allocating its full hidden size.
    layer.A_log = torch.nn.Parameter(torch.zeros(1, 1, 2, 1))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(2))
    for value, name in enumerate(("q_conv1d", "k_conv1d", "v_conv1d"), start=1):
        projection = torch.nn.Module()
        projection.register_parameter("weight", torch.nn.Parameter(torch.full((2, 1, 4), float(value))))
        layer.add_module(name, projection)
    return layer


def _assert_cache(layer, value):
    expected = torch.tensor([value, value, value + 1, value + 1, value + 2, value + 2], dtype=torch.bfloat16)
    torch.testing.assert_close(layer._merged_conv_weight, expected.expand(4, -1))


def test_deferred_reload_refreshes_cache_in_place():
    layer = _make_layer()
    model = torch.nn.Module()
    model.add_module("kda", layer)
    config = SimpleNamespace(dtype=torch.bfloat16, quantization=None, word_embeddings_untied_by_checkpoint=False)
    # Record before first processing, as initialize_model does: the derived
    # buffer does not exist in the checkpoint-shaped restoration metadata.
    record_metadata_for_reloading(model)
    # The runner can create inference tensors, while reload RPCs run outside
    # inference mode. Exercise that transition through the real reload path.
    with torch.inference_mode():
        process_weights_after_loading(model, config, torch.device("cpu"))
    _assert_cache(layer, 1.0)
    captured = layer._merged_conv_weight
    pointer = captured.data_ptr()

    for value in (11.0, 21.0):
        initialize_layerwise_reload(model)
        # Deliberately load direct KDA parameters before child projections.
        # The deferred hook must wait until every child has been finalized.
        for name, parameter in list(model.named_parameters()):
            fill = value
            if ".k_conv1d." in name:
                fill += 1
            elif ".v_conv1d." in name:
                fill += 2
            parameter.weight_loader(parameter, torch.full(parameter.shape, fill))
        finalize_layerwise_reload(model, config)
        _assert_cache(layer, value)
        assert layer._merged_conv_weight is captured
        assert captured.data_ptr() == pointer
