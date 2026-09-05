# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.activation import AscendSituAndMul, SituActivationConfig
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe.fused_moe import (
    AscendMoERunner,
    AscendUnquantizedFusedMoEMethod,
    make_eplb_placement_config,
    use_multistage_eplb_load,
)
from vllm_ascend.quantization.quant_type import QuantType


def _build_weight_layer():
    return SimpleNamespace(
        w13_weight=nn.Parameter(torch.randn(2, 3, 4)),
        w2_weight=nn.Parameter(torch.randn(2, 4, 3)),
    )


def _build_apply_layer():
    return SimpleNamespace(
        w13_weight=nn.Parameter(torch.randn(4, 3, 8)),
        w2_weight=nn.Parameter(torch.randn(4, 8, 3)),
        w13_bias=None,
        w2_bias=None,
        zero_expert_num=0,
        zero_expert_type=None,
        n_shared_experts=0,
        swiglu_limit=0.0,
    )


def _build_unquantized_method(*, dynamic_eplb: bool = False):
    method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
    method.dynamic_eplb = dynamic_eplb
    method.tid2eid = None
    method.moe = SimpleNamespace(has_bias=False)
    method._maybe_pad_weight = MagicMock(side_effect=lambda weight: weight)
    return method


def test_ascend_runner_prefers_runtime_situ_activation():
    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.routed_experts = SimpleNamespace(activation="upstream-activation")
    runner._ascend_runtime_activation = SituActivationConfig(
        beta=4.0,
        linear_beta=25.0,
    )

    assert runner.activation == SituActivationConfig(
        beta=4.0,
        linear_beta=25.0,
    )


def test_ascend_runner_uses_upstream_activation_by_default():
    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.routed_experts = SimpleNamespace(activation="upstream-activation")
    runner._ascend_runtime_activation = None

    assert runner.activation == "upstream-activation"


def test_fused_moe_factory_bridges_situ_activation(monkeypatch):
    from vllm_ascend.patch.platform import patch_fused_moe

    captured = {}

    def fake_fused_moe(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(patch_fused_moe, "_original_FusedMoE", fake_fused_moe)
    monkeypatch.setattr(
        patch_fused_moe,
        "get_ascend_config",
        lambda: SimpleNamespace(
            eplb_config=SimpleNamespace(
                dynamic_eplb=False,
                expert_map_path=None,
                num_redundant_experts=0,
            )
        ),
    )
    activation = SituActivationConfig(beta=4.0, linear_beta=25.0)

    patch_fused_moe._ascend_FusedMoE(
        activation=activation,
        runner_cls=object,
    )

    assert captured["activation"] == "silu"
    assert captured["runner_args"]["runtime_activation"] is activation


@pytest.mark.parametrize(
    ("dynamic_eplb", "policy_type", "collection_interval", "expected"),
    [
        (True, 2, 600, False),
        (True, 3, 600, True),
        (True, 3, 1, False),
        (False, 3, 600, False),
    ],
)
def test_use_multistage_eplb_load(dynamic_eplb, policy_type, collection_interval, expected):
    assert use_multistage_eplb_load(dynamic_eplb, policy_type, collection_interval) is expected


def test_make_eplb_placement_config_does_not_copy_source():
    source = SimpleNamespace(expert_map_path=None, dynamic_eplb=True, num_redundant_experts=0)

    placement_config = make_eplb_placement_config(source, num_redundant_experts=8)

    assert placement_config.expert_map_path is None
    assert placement_config.dynamic_eplb is True
    assert placement_config.num_redundant_experts == 8
    assert source.num_redundant_experts == 0


def test_ascend_unquantized_skips_upstream_modular_kernel_init():
    method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)

    assert method.maybe_make_prepare_finalize() is None


def test_process_weights_after_loading_uses_version_specific_layout(
    monkeypatch,
):
    method = _build_unquantized_method()
    layer = _build_weight_layer()
    w13_parameter = layer.w13_weight
    w2_parameter = layer.w2_weight
    w13_parameter.weight_loader = MagicMock()
    w2_parameter.weight_loader = MagicMock()
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()
    ascend_config = SimpleNamespace(enable_fused_mc2=False)

    monkeypatch.setattr(fused_moe_module, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(fused_moe_module, "maybe_trans_nz", lambda weight: weight)
    upstream_method_base = AscendUnquantizedFusedMoEMethod.__mro__[2]
    monkeypatch.setattr(
        upstream_method_base,
        "process_weights_after_loading",
        lambda self, layer: None,
        raising=False,
    )

    method.process_weights_after_loading(layer)

    torch.testing.assert_close(layer.w13_weight, original_w13.transpose(1, 2))
    torch.testing.assert_close(layer.w2_weight, original_w2.transpose(1, 2))
    assert layer.w13_weight.is_contiguous() is True
    assert layer.w2_weight.is_contiguous() is True
    assert layer.w13_weight is w13_parameter
    assert layer.w2_weight is w2_parameter
    assert layer.w13_weight.weight_loader is w13_parameter.weight_loader
    assert layer.w2_weight.weight_loader is w2_parameter.weight_loader


def test_ascend_runner_promotes_runtime_state_to_buffer():
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    state = torch.tensor([0, 1], dtype=torch.int32)
    runner.runtime_state = state

    runner._promote_attr_to_buffer("runtime_state")

    assert runner.runtime_state is state
    assert dict(runner.named_buffers())["runtime_state"] is state


@pytest.mark.parametrize("moe_comm_type", [MoECommType.ALLGATHER, MoECommType.FUSED_MC2])
def test_unquantized_apply_builds_current_fused_experts_input(monkeypatch, moe_comm_type):
    method = _build_unquantized_method()
    layer = _build_apply_layer()
    hidden_states = torch.randn(2, 3, dtype=torch.float16)
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    routed_out = torch.ones_like(hidden_states)
    moe_comm_method = MagicMock()
    moe_comm_method.fused_experts.return_value = routed_out

    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=moe_comm_type,
            moe_comm_method=moe_comm_method,
            use_megamoe=False,
        ),
    )
    monkeypatch.setattr(fused_moe_module, "get_moe_num_logical_experts", lambda *args, **kwargs: 4)
    monkeypatch.setattr(fused_moe_module, "get_forward_context", lambda: SimpleNamespace(input_ids=None))
    monkeypatch.setattr(fused_moe_module, "get_current_vllm_config", lambda: None)
    select_experts = MagicMock(return_value=(topk_weights, topk_ids))
    monkeypatch.setattr(fused_moe_module, "select_experts", select_experts)

    result = method.apply(
        layer=layer,
        x=hidden_states,
        use_grouped_topk=False,
        top_k=2,
        router_logits=torch.randn(2, 4),
        renormalize=True,
        num_experts=4,
        apply_router_weight_on_input=True,
        activation="gelu",
    )

    assert result is routed_out
    select_experts.assert_called_once()
    fused_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
    assert fused_input.hidden_states is hidden_states
    torch.testing.assert_close(fused_input.topk_weights, topk_weights.to(hidden_states.dtype))
    assert torch.equal(fused_input.topk_ids, topk_ids)
    assert fused_input.routing.apply_router_weight_on_input
    assert fused_input.activation == "gelu"
    assert fused_input.quant.quant_type == QuantType.NONE
    if moe_comm_type == MoECommType.FUSED_MC2:
        assert fused_input.weights.w1[0] is layer.w13_weight
        assert fused_input.weights.w2[0] is layer.w2_weight
    else:
        assert fused_input.weights.w1 is layer.w13_weight
        assert fused_input.weights.w2 is layer.w2_weight


@pytest.mark.parametrize(
    "moe_comm_type, flash_comm_v1_enabled, expected",
    [
        (MoECommType.ALLTOALL, False, True),
        (MoECommType.MC2, False, True),
        (MoECommType.FUSED_MC2, False, True),
        (MoECommType.ALLGATHER, False, False),
        (MoECommType.ALLGATHER, True, True),
    ],
)
def test_runner_reduction_contract(monkeypatch, moe_comm_type, flash_comm_v1_enabled, expected):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    shared_output = object()
    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_type=moe_comm_type, flash_comm_v1_enabled=flash_comm_v1_enabled),
    )

    assert runner.use_dp_chunking is False
    assert runner._fused_output_is_reduced is expected
    assert runner._maybe_reduce_shared_expert_output(shared_output) is shared_output


def test_flashcomm_shared_expert_io_uses_gather_and_reduce_for_all_moe(monkeypatch):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    hidden_states = torch.randn(2, 4)
    gathered = torch.randn(4, 4)
    shared_out = torch.randn(4, 4)
    reduced = torch.randn(2, 4)
    gather = MagicMock(return_value=gathered)
    pad_and_reduce = MagicMock(return_value=reduced)

    monkeypatch.setattr(fused_moe_module, "shared_expert_dp_enabled", lambda: False)
    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            flash_comm_v1_enabled=True,
            moe_comm_type=MoECommType.MC2,
        ),
    )
    monkeypatch.setattr(
        fused_moe_module.torch.ops.vllm,
        "maybe_all_gather_and_maybe_unpad",
        gather,
        raising=False,
    )
    monkeypatch.setattr(
        fused_moe_module.torch.ops.vllm,
        "pad_and_reduce",
        pad_and_reduce,
        raising=False,
    )

    assert runner._prepare_shared_expert_input(hidden_states) is gathered
    assert runner._finalize_shared_expert_output(shared_out) is reduced
    gather.assert_called_once_with(hidden_states, True)
    pad_and_reduce.assert_called_once_with(shared_out)


def test_shared_expert_dp_keeps_flashcomm_replica_path(monkeypatch):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    hidden_states = torch.randn(2, 4)
    shared_out = torch.randn(2, 4)
    gather = MagicMock()
    pad_and_reduce = MagicMock()

    monkeypatch.setattr(fused_moe_module, "shared_expert_dp_enabled", lambda: True)
    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            flash_comm_v1_enabled=True,
            moe_comm_type=MoECommType.MC2,
        ),
    )
    monkeypatch.setattr(
        fused_moe_module.torch.ops.vllm,
        "maybe_all_gather_and_maybe_unpad",
        gather,
        raising=False,
    )
    monkeypatch.setattr(
        fused_moe_module.torch.ops.vllm,
        "pad_and_reduce",
        pad_and_reduce,
        raising=False,
    )

    assert runner._prepare_shared_expert_input(hidden_states) is hidden_states
    assert runner._finalize_shared_expert_output(shared_out) is shared_out
    gather.assert_not_called()
    pad_and_reduce.assert_not_called()


@pytest.mark.parametrize("shared_expert_dp", [False, True])
def test_shared_expert_output_follows_dp_weight_layout(monkeypatch, shared_expert_dp):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    shared_out = torch.randn(2, 4)
    reduced = torch.randn(2, 4)
    all_reduce = MagicMock(return_value=reduced)

    monkeypatch.setattr(fused_moe_module, "shared_expert_dp_enabled", lambda: shared_expert_dp)
    monkeypatch.setattr(fused_moe_module, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            flash_comm_v1_enabled=False,
            moe_comm_type=MoECommType.MC2,
        ),
    )

    output = runner._finalize_shared_expert_output(shared_out)
    if shared_expert_dp:
        assert output is shared_out
        all_reduce.assert_not_called()
    else:
        assert output is reduced
        all_reduce.assert_called_once_with(shared_out)


def test_shared_expert_consistency_uses_shared_input_width(monkeypatch):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    runner.hidden_size = 4
    runner.moe_config = SimpleNamespace(in_dtype=torch.float32)
    integrated_out = torch.randn(10, 4)
    shared_experts = MagicMock(return_value=integrated_out)
    shared_experts.gate_up_proj.input_size = 8
    runner._shared_experts = shared_experts
    runner._shared_experts_part1 = MagicMock(return_value=torch.randn(10, 4))
    runner._shared_experts_part2 = MagicMock(return_value=integrated_out)
    test_input = torch.randn(10, 8)
    rand = MagicMock(return_value=(test_input + 1) / 2)
    monkeypatch.setattr(fused_moe_module.torch, "rand", rand)

    runner._validate_shared_expert_consistency()

    rand.assert_called_once_with(10, 8, device="npu", dtype=torch.float32)
    shared_experts.assert_called_once()
    torch.testing.assert_close(shared_experts.call_args.args[0], test_input)
    runner._shared_experts_part1.assert_called_once()
    torch.testing.assert_close(runner._shared_experts_part1.call_args.args[0], test_input)
    runner._shared_experts_part2.assert_called_once()
    torch.testing.assert_close(runner._shared_experts_part2.call_args.args[0], test_input)
    torch.testing.assert_close(
        runner._shared_experts_part2.call_args.args[1], runner._shared_experts_part1.return_value
    )


class _Projection(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2.0 + 1.0, None


class _Gate(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros((*hidden_states.shape[:-1], 1), dtype=hidden_states.dtype), None


@pytest.mark.parametrize("with_gate", [False, True])
def test_shared_experts_part2_applies_optional_gate(with_gate):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    runner._shared_experts = SimpleNamespace(
        act_fn=nn.Identity(),
        down_proj=_Projection(),
        expert_gate=_Gate() if with_gate else None,
    )
    hidden_states = torch.randn(3, 4)
    shared_gate_up = torch.randn(3, 4)

    output = runner._shared_experts_part2(hidden_states, shared_gate_up)

    expected = shared_gate_up * 2.0 + 1.0
    if with_gate:
        expected = expected * 0.5
    torch.testing.assert_close(output, expected)


def test_unquantized_shared_situ_uses_split_bf16_path(monkeypatch):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    gate_up = torch.randn(2, 8, dtype=torch.bfloat16)
    down_out = torch.randn(2, 4, dtype=torch.bfloat16)
    runner._shared_experts = SimpleNamespace(
        gate_up_proj=_Projection(),
        down_proj=_Projection(),
        act_fn=AscendSituAndMul(beta=4.0, linear_beta=25.0),
    )
    runner._shared_experts_part1 = MagicMock(return_value=gate_up)
    runner._shared_experts_part2 = MagicMock(return_value=down_out)
    runner.quant_type = QuantType.W4A8MXFP
    runner.multistream_overlap_shared_expert = False
    events = fused_moe_module.FusedMoEEvents(
        before_routed_experts=MagicMock(),
        before_dispatch=MagicMock(),
        before_gmm2=MagicMock(),
        before_combine=MagicMock(),
    )

    monkeypatch.setattr(fused_moe_module, "npu_stream_switch", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(fused_moe_module, "shared_expert_dp_enabled", lambda: True)
    monkeypatch.setattr(fused_moe_module, "shared_experts_calculation_stream", MagicMock())
    monkeypatch.setattr(
        fused_moe_module.torch.npu,
        "current_stream",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            flash_comm_v1_enabled=False,
            moe_comm_type=MoECommType.ALLGATHER,
        ),
    )

    output = runner._forward_shared_experts(hidden_states, events)

    assert output is down_out
    runner._shared_experts_part1.assert_called_once_with(hidden_states)
    runner._shared_experts_part2.assert_called_once_with(hidden_states, gate_up)


def test_w8a8_shared_situ_uses_dequant_situ_quant(monkeypatch):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    gate_up = torch.ones(2, 4, dtype=torch.int32)
    quantized_input = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2, dtype=torch.float32)
    quantized_situ = torch.ones(2, 2, dtype=torch.int8)
    situ_scale = torch.ones(2, dtype=torch.float32)
    down_out = torch.randn(2, 4, dtype=torch.bfloat16)
    gate_up_proj = MagicMock()
    gate_up_proj.weight = torch.ones(4, 4, dtype=torch.int8)
    gate_up_proj.weight_scale = torch.ones(4, dtype=torch.bfloat16)
    gate_up_proj.weight_scale_fp32 = torch.ones(4, dtype=torch.float32)
    down_proj = MagicMock()
    down_proj.weight = torch.ones(2, 4, dtype=torch.int8)
    down_proj.weight_scale = torch.ones(4, dtype=torch.bfloat16)
    runner._shared_experts = SimpleNamespace(
        gate_up_proj=gate_up_proj,
        down_proj=down_proj,
        act_fn=AscendSituAndMul(beta=4.0, linear_beta=25.0),
    )
    runner.quant_type = QuantType.W4A8
    runner.multistream_overlap_shared_expert = False
    stream = MagicMock()
    events = fused_moe_module.FusedMoEEvents(
        before_routed_experts=MagicMock(),
        before_dispatch=MagicMock(),
        before_gmm2=MagicMock(),
        before_combine=MagicMock(),
    )
    fast_dynamic_quant = MagicMock(return_value=(quantized_input, input_scale))
    fast_quant_matmul = MagicMock(side_effect=[gate_up, down_out])
    custom_ops = SimpleNamespace(
        dequant_situ_quant=MagicMock(return_value=(quantized_situ, situ_scale)),
    )

    monkeypatch.setattr(fused_moe_module, "npu_stream_switch", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(fused_moe_module, "shared_expert_dp_enabled", lambda: True)
    monkeypatch.setattr(fused_moe_module, "shared_experts_calculation_stream", MagicMock())
    monkeypatch.setattr(fused_moe_module.torch.npu, "current_stream", MagicMock(return_value=stream))
    monkeypatch.setattr(fused_moe_module.torch_npu, "npu_dynamic_quant", fast_dynamic_quant, raising=False)
    monkeypatch.setattr(fused_moe_module.torch_npu, "npu_quant_matmul", fast_quant_matmul)
    monkeypatch.setattr(fused_moe_module.torch.ops, "_C_ascend", custom_ops)
    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            flash_comm_v1_enabled=False,
            moe_comm_type=MoECommType.ALLGATHER,
        ),
    )

    output = runner._forward_shared_experts(hidden_states, events)

    torch.testing.assert_close(output, down_out)
    gate_up_proj.assert_not_called()
    down_proj.assert_not_called()
    fast_dynamic_quant.assert_called_once_with(hidden_states)
    assert fast_quant_matmul.call_count == 2
    situ_call = custom_ops.dequant_situ_quant.call_args.kwargs
    assert situ_call["x"] is gate_up
    assert situ_call["weight_scale"] is gate_up_proj.weight_scale_fp32
    assert situ_call["activation_scale"] is input_scale
    assert situ_call["group_index"] is None
    assert situ_call["beta"] == 4.0
    assert situ_call["linear_beta"] == 25.0


@pytest.mark.parametrize("quant_type", [QuantType.W4A8MXFP, QuantType.W8A8MXFP])
def test_mxfp_shared_situ_uses_situ_mx_quant(monkeypatch, quant_type):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    quantized_input = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
    input_scale = torch.ones(2, 1, dtype=torch.float32)
    gate_up = torch.randn(2, 4, dtype=torch.bfloat16)
    quantized_situ = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
    situ_scale = torch.ones(2, 1, dtype=torch.float32)
    down_out = torch.randn(2, 4, dtype=torch.bfloat16)
    gate_up_proj = MagicMock(return_value=(gate_up, None))
    gate_up_proj.weight_scale = torch.ones(1)
    down_proj = MagicMock(return_value=(down_out, None))
    down_proj.weight_scale = torch.ones(1)
    runner._shared_experts = SimpleNamespace(
        gate_up_proj=gate_up_proj,
        down_proj=down_proj,
        act_fn=AscendSituAndMul(beta=4.0, linear_beta=25.0),
    )
    runner.quant_type = quant_type
    runner.multistream_overlap_shared_expert = False
    events = fused_moe_module.FusedMoEEvents(
        before_routed_experts=MagicMock(),
        before_dispatch=MagicMock(),
        before_gmm2=MagicMock(),
        before_combine=MagicMock(),
    )
    custom_ops = SimpleNamespace(
        situ_mx_quant=MagicMock(return_value=(quantized_situ, situ_scale)),
    )

    monkeypatch.setattr(fused_moe_module, "npu_stream_switch", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(fused_moe_module, "shared_expert_dp_enabled", lambda: True)
    monkeypatch.setattr(fused_moe_module, "shared_experts_calculation_stream", MagicMock())
    monkeypatch.setattr(
        fused_moe_module.torch.npu,
        "current_stream",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        fused_moe_module.torch_npu,
        "npu_dynamic_mx_quant",
        MagicMock(return_value=(quantized_input, input_scale)),
        raising=False,
    )
    monkeypatch.setattr(fused_moe_module.torch.ops, "_C_ascend", custom_ops)
    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            flash_comm_v1_enabled=False,
            moe_comm_type=MoECommType.ALLGATHER,
        ),
    )

    output = runner._forward_shared_experts(hidden_states, events)

    assert output is down_out
    gate_up_proj.assert_called_once_with((quantized_input, input_scale))
    down_proj.assert_called_once_with((quantized_situ, situ_scale))
    situ_call = custom_ops.situ_mx_quant.call_args.kwargs
    assert situ_call["x"] is gate_up
    assert situ_call["beta"] == 4.0
    assert situ_call["linear_beta"] == 25.0
    assert situ_call["dst_type"] == 36


@pytest.mark.parametrize("has_shared_experts", [False, True])
def test_shared_forward_impl_returns_current_runner_contract(monkeypatch, has_shared_experts):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    runner._shared_experts = object() if has_shared_experts else None
    hidden_states = torch.randn(2, 4)
    shared_experts_input = torch.randn(2, 8)
    router_logits = torch.randn(2, 3)
    routed_out = torch.randn(2, 4)
    shared_out = torch.randn(2, 4)
    routed_result = SimpleNamespace(
        routed_out=routed_out,
        before_dispatch_evt=None,
        before_gmm2_evt=None,
        before_combine_evt=None,
        swiglu_limit=0.0,
    )
    runner.no_shared_forward_impl = MagicMock(return_value=routed_result)
    runner._forward_shared_experts = MagicMock(return_value=shared_out)
    current_stream = MagicMock()

    monkeypatch.setattr(AscendMoERunner, "is_internal_router", property(lambda _: False))
    monkeypatch.setattr(fused_moe_module.torch.npu, "current_stream", lambda: current_stream)

    result = runner.shared_forward_impl(hidden_states, router_logits, shared_experts_input)

    runner.no_shared_forward_impl.assert_called_once_with(
        hidden_states,
        router_logits,
        return_with_event=True,
    )
    if has_shared_experts:
        assert result[0] is shared_out
        assert result[1] is routed_out
        assert runner._forward_shared_experts.call_args.args[0] is shared_experts_input
    else:
        assert result is routed_out
        runner._forward_shared_experts.assert_not_called()


def test_forward_impl_preserves_original_input_for_shared_experts(monkeypatch):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    routed_input = torch.randn(2, 3584)
    shared_experts_input = torch.randn(2, 7168)
    router_logits = torch.randn(2, 3)
    expected = (torch.randn(2, 7168), torch.randn(2, 3584))
    runner.shared_forward_impl = MagicMock(return_value=expected)

    monkeypatch.setattr(AscendMoERunner, "shared_experts", property(lambda _: object()))
    monkeypatch.setattr(AscendMoERunner, "_sequence_parallel_context", lambda _: nullcontext())

    result = runner._forward_impl(routed_input, router_logits, shared_experts_input)

    assert result is expected
    runner.shared_forward_impl.assert_called_once_with(
        routed_input,
        router_logits,
        shared_experts_input,
    )
