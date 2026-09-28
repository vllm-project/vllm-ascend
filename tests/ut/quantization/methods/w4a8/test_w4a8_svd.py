from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_ascend.quantization.methods.w4a8.w4a8_svd import (
    AscendW4A8SVDFusedMoEMethod,
    AscendW4A8SVDMoEScheme,
)


def test_factor_parameters_load_by_global_expert_and_split_gate_up():
    method = object.__new__(AscendW4A8SVDFusedMoEMethod)
    method.rank = 32
    layer = torch.nn.Module()
    layer.expert_map = torch.tensor([-1, 0, 1])
    method.create_weights(layer, 2, 64, 32, torch.bfloat16)

    weight = torch.randint(-128, 128, (16, 64), dtype=torch.int8)
    param = layer.w13_right_weight
    assert param.weight_loader(param, weight, shard_id="w3", expert_id=2, return_success=True)
    assert not param.weight_loader(param, weight, shard_id="w3", expert_id=0, return_success=True)
    torch.testing.assert_close(param[1, 1], weight.T.contiguous().view(torch.int32))


def test_process_weights_splits_loading_tensors_without_dense_reconstruction():
    method = object.__new__(AscendW4A8SVDFusedMoEMethod)
    method.rank = 32
    layer = torch.nn.Module()
    layer.expert_map = None
    layer.ep_rank = 0
    method.create_weights(layer, 2, 64, 32, torch.bfloat16)
    bytes_before = sum(parameter.numel() * parameter.element_size() for parameter in layer.parameters())

    scheme = object.__new__(AscendW4A8SVDMoEScheme)
    scheme.process_weights_after_loading(layer)
    factors = scheme.get_low_rank_weights(layer)

    assert not hasattr(layer, "w13_left_weight")
    assert len(factors) == 3
    assert factors[0].left is layer.w1_left_weight
    assert factors[1].left is layer.w3_left_weight
    assert factors[2].right is layer.w2_right_weight
    assert bytes_before == sum(parameter.numel() * parameter.element_size() for parameter in layer.parameters())


def test_low_rank_scheme_disables_eplb():
    scheme = object.__new__(AscendW4A8SVDMoEScheme)
    assert scheme.supports_eplb is False


def test_upstream_expert_loader_recognizes_factor_checkpoint_names():
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.3.mlp.experts"
    layer.ckpt_gate_proj_name = "gate_proj"
    layer.ckpt_up_proj_name = "up_proj"
    layer.ckpt_down_proj_name = "down_proj"
    layer.lora_base_layer_prefix = ""
    layer.moe_config = MagicMock(num_experts=2, num_logical_experts=2)
    layer.expert_map_manager = MagicMock(num_fused_shared_experts=0)
    layer.register_buffer("_expert_map", torch.tensor([-1, 0]))
    method = object.__new__(AscendW4A8SVDFusedMoEMethod)
    method.rank = 32
    method.create_weights(layer, 1, 64, 32, torch.bfloat16)
    weights = [
        ("0.gate_proj.left_weight", torch.ones(16, 32, dtype=torch.int8)),
        ("1.gate_proj.left_weight", torch.ones(16, 32, dtype=torch.int8)),
        ("1.up_proj.right_scale", torch.ones(32, 1)),
        ("1.down_proj.left_bias", torch.ones(64)),
    ]

    loaded = list(layer.load_weights(weights))

    assert loaded == ["w13_left_weight", "w13_right_scale", "w2_left_bias"]
    torch.testing.assert_close(layer.w2_left_bias[0], weights[-1][1])
    expected = weights[1][1].T.contiguous().view(torch.int32)
    torch.testing.assert_close(layer.w13_left_weight[0, 0], expected)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_factor_loader_rejects_invalid_scales(scale):
    method = object.__new__(AscendW4A8SVDFusedMoEMethod)
    method.rank = 32
    layer = torch.nn.Module()
    method.create_weights(layer, 1, 64, 32, torch.bfloat16)
    param = layer.w2_left_scale
    with pytest.raises(ValueError, match="finite positive"):
        param.weight_loader(param, torch.full((64, 1), scale), shard_id="w2", expert_id=0)


def test_create_weights_rejects_unaligned_dimensions():
    method = object.__new__(AscendW4A8SVDFusedMoEMethod)
    method.rank = 32
    with pytest.raises(ValueError, match="aligned to 32"):
        method.create_weights(torch.nn.Module(), 1, 65, 32, torch.bfloat16)


@pytest.mark.parametrize("rank", [None, True, 32.0, "32", 0, -32, 16, 96])
def test_create_weights_rejects_invalid_rank(rank):
    method = object.__new__(AscendW4A8SVDFusedMoEMethod)
    method.rank = rank
    with pytest.raises(ValueError, match="rank"):
        method.create_weights(torch.nn.Module(), 1, 64, 32, torch.bfloat16)


@pytest.mark.parametrize("case", ["supported", "model", "activation", "bias", "lora", "dtype", "tp", "eplb"])
def test_runtime_configuration_is_validated_before_loading(case):
    hf_config = SimpleNamespace(model_type="deepseek_v3", hidden_size=64, moe_intermediate_size=32)
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(enable_expert_parallel=True, tensor_parallel_size=2, enable_eplb=False),
        lora_config=None,
    )
    moe_config = SimpleNamespace(activation="silu", has_bias=False)
    config = dict(format_version=1, weight_bits=4, activation_bits=8, group_size=0, rank=32)
    if case == "model":
        hf_config.model_type = "qwen3_moe"
    elif case == "activation":
        moe_config.activation = "gelu"
    elif case == "bias":
        moe_config.has_bias = True
    elif case == "lora":
        runtime.lora_config = object()
    elif case == "dtype":
        runtime.model_config.dtype = torch.float16
    elif case == "tp":
        runtime.parallel_config.enable_expert_parallel = False
    elif case == "eplb":
        runtime.parallel_config.enable_eplb = True
    with (
        patch("vllm_ascend.quantization.methods.w4a8.w4a8_svd.get_current_vllm_config", return_value=runtime),
        patch("vllm_ascend.quantization.methods.w4a8.w4a8_svd.AscendW4A8SVDMoEScheme") as scheme,
        patch("vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod.__init__", return_value=None) as base_init,
    ):
        scheme.return_value.dynamic_eplb = False
        if case == "supported":
            assert AscendW4A8SVDFusedMoEMethod(moe_config, config).rank == 32
            base_init.assert_called_once()
        else:
            with pytest.raises(ValueError):
                AscendW4A8SVDFusedMoEMethod(moe_config, config)
            base_init.assert_not_called()


@pytest.mark.parametrize("shard", ["w1", "invalid"])
def test_factor_loader_rejects_incorrect_down_projection_shard(shard):
    method = object.__new__(AscendW4A8SVDFusedMoEMethod)
    method.rank = 32
    layer = torch.nn.Module()
    method.create_weights(layer, 1, 64, 32, torch.bfloat16)
    param = layer.w2_left_bias
    with pytest.raises(ValueError, match="shard"):
        param.weight_loader(param, torch.ones(64), shard_id=shard, expert_id=0)


def test_factor_loader_rejects_nonfinite_compensation_bias():
    method = object.__new__(AscendW4A8SVDFusedMoEMethod)
    method.rank = 32
    layer = torch.nn.Module()
    method.create_weights(layer, 1, 64, 32, torch.bfloat16)
    param = layer.w2_left_bias
    with pytest.raises(ValueError, match="finite FP32"):
        param.weight_loader(param, torch.full((64,), float("nan")), shard_id="w2", expert_id=0)
