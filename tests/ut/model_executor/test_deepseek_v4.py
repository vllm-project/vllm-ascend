from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v4 import model as deepseek_v4


class StubModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def test_routed_moe_receives_configured_swiglu_limit(monkeypatch):
    fused_moe_kwargs = {}

    class StubFusedMoEFactory(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            fused_moe_kwargs.update(kwargs)

    monkeypatch.setattr(deepseek_v4, "FusedMoEFactory", StubFusedMoEFactory)
    monkeypatch.setattr(deepseek_v4, "ReplicatedLinear", StubModule)
    monkeypatch.setattr(deepseek_v4, "DeepseekV2MLP", StubModule)
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        deepseek_v4,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 1), rank_in_group=0),
    )
    monkeypatch.setattr(deepseek_v4, "get_ascend_config", lambda: SimpleNamespace(mix_placement=False))
    monkeypatch.setattr(deepseek_v4.rocm_aiter_ops, "is_fused_moe_enabled", lambda: False)
    monkeypatch.setattr(deepseek_v4.rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: False)

    config = SimpleNamespace(
        hidden_act="silu",
        hidden_size=16,
        moe_intermediate_size=8,
        n_group=1,
        n_routed_experts=8,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_experts_per_tok=2,
        num_hash_layers=0,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        swiglu_limit=10.0,
        topk_group=1,
    )
    parallel_config = SimpleNamespace(
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        use_sequence_parallel_moe=False,
    )

    deepseek_v4.DeepseekV4MoE(config, parallel_config, prefix="model.layers.1.mlp")

    assert fused_moe_kwargs["swiglu_limit"] == config.swiglu_limit


@pytest.mark.parametrize("device", list(AscendDeviceType))
@pytest.mark.parametrize("custom_ops", [False, True])
def test_rms_norm_cast_respects_hardware_support(monkeypatch, device, custom_ops):
    hidden_states = torch.randn(2, 8, dtype=torch.bfloat16)
    normalized = torch.randn_like(hidden_states)
    layernorm = Mock(return_value=normalized, weight=torch.ones(8), variance_epsilon=1e-6)
    layer = SimpleNamespace(post_attention_layernorm=layernorm)
    fused_outputs = (torch.empty_like(hidden_states), torch.empty_like(hidden_states, dtype=torch.float32))
    fused_op = Mock(return_value=fused_outputs)
    monkeypatch.setattr(deepseek_v4, "get_current_hardware_profile", lambda: get_hardware_profile(device))
    monkeypatch.setattr(deepseek_v4, "enable_custom_op", lambda: custom_ops)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_rms_norm_cast", fused_op, raising=False)

    result = deepseek_v4.DeepseekV4DecoderLayer.rms_norm_cast(layer, hidden_states)

    if device == AscendDeviceType.A3 and custom_ops:
        assert result is fused_outputs
        fused_op.assert_called_once_with(hidden_states, layernorm.weight, layernorm.variance_epsilon)
        layernorm.assert_not_called()
    else:
        fused_op.assert_not_called()
        layernorm.assert_called_once_with(hidden_states)
        assert result[0] is normalized
        assert result[1].dtype == torch.float32
        torch.testing.assert_close(result[1], normalized.float(), rtol=0, atol=0)
