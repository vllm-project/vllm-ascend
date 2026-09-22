# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention
from vllm_ascend.models.glm5next.kda_projection import KDAFGProjection
from vllm_ascend.models.glm5next.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)


def make_layer(width, tp_rank=0, tp_size=1, loader_version=2):
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.head_dim = 128
    # Exercise the real merged loader without initializing distributed groups.
    projection = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    torch.nn.Module.__init__(projection)
    projection.output_sizes = [width * tp_size, width * tp_size]
    projection.tp_size = tp_size
    projection.tp_rank = tp_rank
    with (
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=tp_rank),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=tp_size),
    ):
        projection.weight = ModelWeightParameter(
            data=torch.randn(2 * width, 128),
            input_dim=1,
            output_dim=0,
            weight_loader=projection.weight_loader_v2 if loader_version == 2 else projection.weight_loader,
        )
    layer.fg_b_proj = KDAFGProjection.__new__(KDAFGProjection)
    torch.nn.Module.__init__(layer.fg_b_proj)
    layer.fg_b_proj.head_dim = 128
    layer.fg_b_proj._merged = projection
    return layer


def make_model(layer):
    model = Glm5NextModel.__new__(Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(is_moe=False, is_linear_attn=True, num_nextn_predict_layers=0)
    model.add_module("linear_attention", layer)
    return model


def reference(layer, inputs):
    f_weight, g_weight = layer.fg_b_proj._merged.weight.chunk(2)
    fa, ga = inputs.split(128, dim=-1)
    return torch.nn.functional.linear(fa, f_weight), torch.nn.functional.linear(ga, g_weight)


@pytest.mark.parametrize("num_tokens", [0, 1, 8, 64, 128, 129, 2048])
@pytest.mark.parametrize("width", [512, 1024])
def test_batched_projection_reads_strided_fused_input(num_tokens, width):
    torch.manual_seed(1024)
    layer = make_layer(width)
    projected = torch.randn(num_tokens, 3 * width + width // 128 + 256)
    fg_a = projected[:, -256:]

    actual = layer.fg_b_proj(fg_a)

    torch.testing.assert_close(actual, reference(layer, fg_a))
    assert "_fg_b_weight" not in layer.state_dict()
    assert not list(layer.buffers())


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("wrapper", [False, True])
@pytest.mark.parametrize("loader_version", [1, 2])
def test_checkpoint_reload_keeps_storage_and_shards_both_projections(tp_size, wrapper, loader_version):
    width = 16
    full_weight = torch.arange(width * tp_size * 128, dtype=torch.float32).reshape(-1, 128)
    for rank in range(tp_size):
        layer = make_layer(width, rank, tp_size, loader_version)
        model = make_model(layer)
        prefix = "linear_attention."
        if wrapper:
            causal = Glm5NextForCausalLM.__new__(Glm5NextForCausalLM)
            torch.nn.Module.__init__(causal)
            causal.model = model
            outer = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
            torch.nn.Module.__init__(outer)
            outer.language_model = causal
            model = outer
            prefix = "model.language_model.linear_attention."
        address = layer.fg_b_proj._merged.weight.data_ptr()
        for scale in (1, 2):
            # Separate chunks, including g before f, must not read stale peers.
            for name, sign in (("g_b_proj", -1), ("f_b_proj", 1)):
                checkpoint_name = "forget_gate." + name if wrapper else name
                loaded = model.load_weights([(prefix + checkpoint_name + ".weight", full_weight * sign * scale)])
                expected_name = "linear_attention.fg_b_proj._merged.weight"
                if wrapper:
                    expected_name = "language_model.model." + expected_name
                assert loaded == {expected_name}
            expected = full_weight[rank * width : (rank + 1) * width] * scale
            torch.testing.assert_close(layer.fg_b_proj._merged.weight, torch.cat((expected, -expected)))
            assert layer.fg_b_proj._merged.weight.data_ptr() == address


@torch.no_grad()
@pytest.mark.parametrize("tokens", [4, 128, 129])
def test_kernel_format_reload_changes_bmm_and_gemm_without_refresh(tokens):
    layer = make_layer(16)
    model = make_model(layer)
    runner = SimpleNamespace(
        get_model=lambda: model,
        lora_config=None,
        model_config=SimpleNamespace(quantization=None),
        reset_lora_state=lambda: None,
        reset_encoder_cache=lambda: None,
        reset_mm_cache=lambda: None,
    )
    inputs = torch.ones(tokens, 256)
    layer.fg_b_proj(inputs)
    address = layer.fg_b_proj._merged.weight.data_ptr()
    for f_scale, g_scale in ((2, 3), (5, -1)):
        weights = torch.cat((torch.full((16, 128), f_scale), torch.full((16, 128), g_scale)))
        GPUModelRunner.reload_weights(
            runner,
            weights_iterator=[("linear_attention.fg_b_proj._merged.weight", weights)],
            is_checkpoint_format=False,
        )
        actual = layer.fg_b_proj(inputs)
        torch.testing.assert_close(actual[0], torch.full((tokens, 16), float(128 * f_scale)))
        torch.testing.assert_close(actual[1], torch.full((tokens, 16), float(128 * g_scale)))
        assert layer.fg_b_proj._merged.weight.data_ptr() == address


def test_load_state_dict_updates_projection_without_refresh():
    layer = make_layer(16)
    inputs = torch.ones(4, 256)
    layer.fg_b_proj(inputs)
    address = layer.fg_b_proj._merged.weight.data_ptr()
    layer.load_state_dict({"fg_b_proj._merged.weight": torch.full((32, 128), 2.0)})
    torch.testing.assert_close(layer.fg_b_proj(inputs), (torch.full((4, 16), 256.0),) * 2)
    assert layer.fg_b_proj._merged.weight.data_ptr() == address


@torch.no_grad()
def test_unmerged_checkpoint_loading_keeps_original_parameter_names():
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.head_dim = 128
    layer.fg_b_proj = KDAFGProjection.__new__(KDAFGProjection)
    torch.nn.Module.__init__(layer.fg_b_proj)
    layer.fg_b_proj.head_dim = 128
    layer.fg_b_proj._merged = None
    with (
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
    ):
        layer.fg_b_proj._f = ColumnParallelLinear(128, 16, bias=False, disable_tp=True)
        layer.fg_b_proj._g = ColumnParallelLinear(128, 16, bias=False, disable_tp=True)
    model = make_model(layer)
    addresses = [p.data_ptr() for p in layer.parameters()]
    for scale in (2, 3):
        loaded = model.load_weights(
            [
                (f"linear_attention.{name}.weight", torch.full((16, 128), float(scale)))
                for name in ("f_b_proj", "g_b_proj")
            ]
        )
        assert loaded == {"linear_attention.fg_b_proj._f.weight", "linear_attention.fg_b_proj._g.weight"}
        with patch("torch.ops.vllm.unquantized_gemm", side_effect=torch.nn.functional.linear):
            torch.testing.assert_close(
                layer.fg_b_proj(torch.ones(4, 256)), (torch.full((4, 16), float(128 * scale)),) * 2
            )
        assert [p.data_ptr() for p in layer.parameters()] == addresses


@torch.no_grad()
def test_layerwise_checkpoint_reload_keeps_captured_weight_storage():
    layer = make_layer(16)
    model = make_model(layer)
    record_metadata_for_reloading(model)
    inputs = torch.ones(4, 256)
    layer.fg_b_proj(inputs)
    address = layer.fg_b_proj._merged.weight.data_ptr()
    for scale in (2, 3):
        initialize_layerwise_reload(model)
        for name in ("g_b_proj", "f_b_proj"):
            model.load_weights([("linear_attention." + name + ".weight", torch.full((16, 128), float(scale)))])
        finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.float32))
        torch.testing.assert_close(layer.fg_b_proj(inputs), (torch.full((4, 16), float(128 * scale)),) * 2)
        assert layer.fg_b_proj._merged.weight.data_ptr() == address


@torch.no_grad()
@pytest.mark.parametrize("width", [16, 8192])
def test_projection_constructor_and_layerwise_reload(width):
    with (
        patch("vllm_ascend.ops.linear.get_parallel_op", return_value=(None, 0, 1)),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
    ):
        projection = KDAFGProjection(128, width, width)
    model = torch.nn.Module()
    model.add_module("fg", projection)
    record_metadata_for_reloading(model)
    addresses = [param.data_ptr() for param in model.parameters()]
    for scale in (2, 3):
        initialize_layerwise_reload(model)
        loaded = projection.load_weights(
            [
                ("g_b_proj.weight", torch.full((width, 128), -float(scale))),
                ("f_b_proj.weight", torch.full((width, 128), float(scale))),
            ]
        )
        assert loaded == set(dict(projection.named_parameters()))
        finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.float32))
        assert [param.data_ptr() for param in model.parameters()] == addresses
        with patch("torch.ops.vllm.unquantized_gemm", side_effect=torch.nn.functional.linear):
            actual = projection(torch.ones(4, 256))
        torch.testing.assert_close(
            actual, (torch.full((4, width), 128.0 * scale), torch.full((4, width), -128.0 * scale))
        )
