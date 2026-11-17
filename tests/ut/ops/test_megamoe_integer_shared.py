# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMulWithClamp

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.fused_moe import fused_moe, moe_utils, shared_experts
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MegamoeSharedExpertFusion
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.quantization.quant_type import QuantType


@pytest.fixture(autouse=True)
def default_vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _linear(shape):
    scheme = SimpleNamespace(quant_type=QuantType.W8A8, act_quant_type=torch.int8)
    return SimpleNamespace(
        weight=torch.arange(shape[0] * shape[1], dtype=torch.int8).reshape(shape),
        weight_scale_fp32=torch.full((shape[1],), 0.5),
        weight_offset=torch.zeros(shape[1]),
        bias=None,
        quant_method=SimpleNamespace(quant_method=scheme, process_weights_after_loading=Mock()),
    )


@pytest.fixture
def collector(monkeypatch):
    monkeypatch.setattr(
        shared_experts, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3)
    )
    monkeypatch.setattr(shared_experts.torch_npu, "npu_format_cast", lambda weight, _: weight)
    instance = AscendSharedExperts.__new__(AscendSharedExperts)
    instance.moe_config = SimpleNamespace(
        hidden_dim=4,
        intermediate_size_per_partition=2,
        tp_group=SimpleNamespace(world_size=1),
        activation="silu",
        swiglu_limit=10.0,
    )
    instance.quant_type = QuantType.W4A8
    instance.weights_replicated = False
    instance._megamoe_shared_fusion_disabled = False
    instance._megamoe_shared_parts = {}
    instance._megamoe_shared_fusion = MegamoeSharedExpertFusion()
    instance._megamoe_shared_quant_type = QuantType.NONE
    instance.megamoe_shared_weights_ready = False
    instance.layer = SimpleNamespace(
        gate_up_proj=_linear((4, 4)),
        down_proj=_linear((2, 4)),
        act_fn=SiluAndMulWithClamp(10.0),
    )
    return instance


def test_integer_shared_bitcasts_scales_and_keeps_standalone_weights(collector):
    w1, w2 = collector.layer.gate_up_proj, collector.layer.down_proj
    before = w1.weight.clone()
    collector._collect_megamoe_shared_weight("w1", w1)
    assert not collector.megamoe_shared_weights_ready
    assert collector.megamoe_shared_fusion().weights is None
    collector._collect_megamoe_shared_weight("w2", w2)
    fusion = collector.megamoe_shared_fusion()
    assert collector.megamoe_shared_weights_ready
    assert torch.equal(w1.weight, before)
    assert torch.equal(fusion.weights.w1[0], before)
    assert fusion.weights.w1[0].shape == (4, 4)
    assert fusion.weights.w2[0].shape == (2, 4)
    assert fusion.weights.w1_scale[0].dtype == torch.int64
    assert fusion.weights.w1_scale[0].tolist() == [0x3F000000] * 4
    assert w1.weight_scale_fp32.dtype == torch.float32
    assert torch.equal(w1.weight_scale_fp32, torch.full((4,), 0.5))
    _, quant_out, weight_type = moe_utils._get_cann_mega_moe_quant_settings(QuantType.W8A8)
    assert (fusion.quant_out_dtype, fusion.weight_type) == (quant_out, weight_type)
    assert fusion.weight_type != moe_utils._get_cann_mega_moe_quant_settings(QuantType.W4A8)[2]
    assert AscendW8A8DynamicLinearMethod.quant_type == QuantType.W8A8


@pytest.mark.parametrize(
    "invalid",
    [
        "offset",
        "missing_offset",
        "offset_shape",
        "shard",
        "dtype",
        "scale",
        "scale_dtype",
        "nan",
        "inf",
        "zero",
        "negative",
        "quant",
        "act",
    ],
)
def test_integer_shared_rejects_unsupported_payload_and_clears_readiness(collector, invalid):
    collector._collect_megamoe_shared_weight("w1", collector.layer.gate_up_proj)
    linear = collector.layer.down_proj
    collector._collect_megamoe_shared_weight("w2", linear)
    assert collector.megamoe_shared_weights_ready
    assert collector.megamoe_shared_fusion().weights is not None
    if invalid == "offset":
        linear.weight_offset[0] = 1
    elif invalid == "missing_offset":
        linear.weight_offset = None
    elif invalid == "offset_shape":
        linear.weight_offset = torch.zeros(1)
    elif invalid == "shard":
        linear.weight = linear.weight[:1]
    elif invalid == "dtype":
        linear.weight = linear.weight.float()
    elif invalid == "scale":
        linear.weight_scale_fp32 = torch.ones(1)
    elif invalid == "scale_dtype":
        linear.weight_scale_fp32 = linear.weight_scale_fp32.half()
    elif invalid == "nan":
        linear.weight_scale_fp32[0] = float("nan")
    elif invalid == "inf":
        linear.weight_scale_fp32[0] = float("inf")
    elif invalid == "zero":
        linear.weight_scale_fp32[0] = 0
    elif invalid == "negative":
        linear.weight_scale_fp32[0] = -0.5
    elif invalid == "quant":
        linear.quant_method.quant_method.quant_type = QuantType.NONE
    else:
        linear.quant_method.quant_method.act_quant_type = torch.float16
    collector._collect_megamoe_shared_weight("w2", linear)
    assert collector._megamoe_shared_fusion_disabled
    assert not collector.megamoe_shared_weights_ready
    assert collector.megamoe_shared_fusion().weights is None
    assert collector._megamoe_shared_parts == {}


@pytest.mark.parametrize("scenario", ["mode2", "unquantized", "sharded", "activation", "replicated", "valid"])
def test_integer_shared_setup_limits_scope(collector, monkeypatch, scenario):
    monkeypatch.setattr(shared_experts, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(shared_experts, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        shared_experts, "get_ascend_config", lambda: SimpleNamespace(megamoe_shared_expert_fusion=scenario != "mode2")
    )
    if scenario == "unquantized":
        collector.quant_type = QuantType.NONE
    elif scenario == "sharded":
        collector.moe_config.tp_group.world_size = 8
    elif scenario == "activation":
        collector.moe_config.swiglu_limit = None
    elif scenario == "replicated":
        collector.moe_config.tp_group.world_size = 8
        collector.weights_replicated = True
    original = collector.layer.gate_up_proj.quant_method.process_weights_after_loading
    collector._setup_megamoe_shared_fusion()
    wrapped = collector.layer.gate_up_proj.quant_method.process_weights_after_loading
    if scenario in ("valid", "replicated"):
        assert wrapped is not original
        wrapped(collector.layer.gate_up_proj)
        original.assert_called_once_with(collector.layer.gate_up_proj)
        assert "w1" in collector._megamoe_shared_parts
    else:
        assert wrapped is original
        assert not collector.megamoe_shared_weights_ready


@pytest.mark.parametrize("use_mega_moe", [False, True])
@pytest.mark.parametrize("comm_type", [MoECommType.FUSED_MC2, MoECommType.MC2])
@pytest.mark.parametrize("ready,disabled", [(False, False), (True, False), (True, True)])
def test_shared_forward_is_skipped_only_for_actual_megamoe_step(monkeypatch, use_mega_moe, comm_type, ready, disabled):
    monkeypatch.setattr(fused_moe, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=comm_type, use_mega_moe=use_mega_moe))
    runner = SimpleNamespace(
        ascend_shared_experts=SimpleNamespace(
            megamoe_shared_weights_ready=ready,
            _megamoe_shared_fusion_disabled=disabled,
        )
    )
    assert fused_moe.AscendMoERunner._megamoe_shared_expert_fused.fget(runner) == (
        ready and not disabled and use_mega_moe and comm_type is MoECommType.FUSED_MC2
    )


_INTEGER_SHARED_ABI = frozenset(
    {
        "shared_l1_weights",
        "shared_l2_weights",
        "shared_l1_weights_sf",
        "shared_l2_weights_sf",
        "shared_expert_quant_out_dtype",
        "shared_weight1_type",
        "shared_weight2_type",
    }
)


@pytest.mark.parametrize("missing_arg", sorted(_INTEGER_SHARED_ABI))
def test_integer_shared_abi_rejects_each_missing_descriptor(missing_arg):
    with pytest.raises(RuntimeError, match=missing_arg):
        FusedMC2CommImpl._validate_integer_shared_abi(_INTEGER_SHARED_ABI - {missing_arg})


def test_integer_shared_abi_accepts_complete_descriptor_set():
    FusedMC2CommImpl._validate_integer_shared_abi(_INTEGER_SHARED_ABI)


@pytest.mark.parametrize("sequence_parallel,replicated", [(False, False), (False, True), (True, False)])
def test_deepseek_shared_mlp_can_load_full_weights(monkeypatch, sequence_parallel, replicated):
    from vllm_ascend.models.deepseek_v41 import model

    gate_up = Mock(return_value=torch.nn.Identity())
    down = Mock(return_value=torch.nn.Identity())
    monkeypatch.setattr(model, "MergedColumnParallelLinear", gate_up)
    monkeypatch.setattr(model, "RowParallelLinear", down)
    model.DeepseekV41MLP(
        hidden_size=4,
        intermediate_size=2,
        hidden_act="silu",
        reduce_results=False,
        is_sequence_parallel=sequence_parallel,
        replicate_weights=replicated,
    )
    assert gate_up.call_args.kwargs["disable_tp"] == (sequence_parallel or replicated)
    assert down.call_args.kwargs["disable_tp"] == (sequence_parallel or replicated)
    assert not down.call_args.kwargs["reduce_results"]
