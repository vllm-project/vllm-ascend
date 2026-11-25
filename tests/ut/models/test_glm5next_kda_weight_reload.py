# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise fused checkpoint shards, deferred refits and graph-owned KDA weights."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor import custom_op, parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention, _fused_qkv_conv1d


@pytest.mark.parametrize(("tp_size", "tp_rank"), [(1, 0), (2, 0), (2, 1)])
def test_fused_conv_shards_and_deferred_reload_preserve_cache(monkeypatch, tp_size, tp_rank):
    monkeypatch.delitem(custom_op.op_registry_oot, "ColumnParallelLinear", raising=False)
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer._merged_conv_weight = None
    layer.A_log = torch.nn.Parameter(torch.zeros(1, 1, 2, 1))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(2))
    layer.conv1d = _fused_qkv_conv1d(4, 4, tp_size, tp_rank, "kda.conv1d")
    model = torch.nn.Module()
    model.add_module("kda", layer)
    config = SimpleNamespace(dtype=torch.bfloat16, quantization=None, word_embeddings_untied_by_checkpoint=False)
    layer.model_config = config

    def load_shards(value):
        expected = []
        for shard_id in range(3):
            weight = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4) / 16 + value + shard_id
            # Checkpoint loaders accept both [C, K] and [C, 1, K].
            source = weight.squeeze(1) if shard_id == 1 else weight
            layer.conv1d.weight.weight_loader(layer.conv1d.weight, source, shard_id)
            expected.append(weight.chunk(tp_size)[tp_rank])
        return torch.cat(expected).squeeze(1).T.to(torch.bfloat16).contiguous()

    expected = load_shards(1.0)
    record_metadata_for_reloading(model)
    with torch.inference_mode():
        process_weights_after_loading(model, config, torch.device("cpu"))
    torch.testing.assert_close(layer._merged_conv_weight, expected)
    captured = layer._merged_conv_weight
    assert captured is not None
    assert "_merged_conv_weight" not in dict(layer.named_buffers(recurse=False))
    assert "_merged_conv_weight" not in dict(layer.named_parameters(recurse=False))
    pointer = captured.data_ptr()
    parameter_pointer = layer.conv1d.weight.data_ptr()

    for value in (11.0, 21.0):
        initialize_layerwise_reload(model)
        # The deferred hook must wait for the fused child even when the
        # attention layer's own parameters have already been finalized.
        for param in (layer.A_log, layer.dt_bias):
            param.weight_loader(param, torch.full(param.shape, value))
        expected = load_shards(value)
        finalize_layerwise_reload(model, config)
        torch.testing.assert_close(layer._merged_conv_weight, expected)
        assert layer._merged_conv_weight is captured
        assert captured.data_ptr() == pointer
        assert layer.conv1d.weight.data_ptr() == parameter_pointer
