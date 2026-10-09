from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.quant_moe import (
    QuantMoELoRAImpl,
    _apply_moe_activation,
    quant_apply_mlp_with_moe_lora,
    register_quant_moe_lora_impl,
    validate_quant_moe_lora_activation_input,
)
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.quantization.quant_type import QuantType

QUANT_MOE = "vllm_ascend.lora.quant_moe"


def _make_input(**overrides) -> MoEMlpComputeInput:
    values = dict(
        hidden_states=torch.randn(2, 4, dtype=torch.bfloat16),
        group_list=torch.tensor([1, 1], dtype=torch.int64),
        group_list_type=1,
        dynamic_scale=None,
        topk_scales=None,
        weights=MoEWeights(
            w1=[torch.ones(1, 4, 6, dtype=torch.int8)],
            w2=[torch.ones(1, 3, 4, dtype=torch.int8)],
            w1_scale=[torch.ones(1, 6)],
            w2_scale=[torch.ones(1, 4, dtype=torch.bfloat16)],
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8),
        fusion=True,
        activation="silu",
        expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
        topk_ids=torch.tensor([[0], [1]], dtype=torch.int32),
        lora_context=SimpleNamespace(use_ep=False),
    )
    values.update(overrides)
    return MoEMlpComputeInput(**values)


@pytest.mark.parametrize(
    ("comm_type", "mlp_input"),
    [
        (MoECommType.ALLGATHER, _make_input()),
        (
            MoECommType.ALLTOALL,
            _make_input(
                expanded_row_idx=None,
                topk_ids=None,
                lora_context=SimpleNamespace(use_ep=True),
            ),
        ),
    ],
)
def test_dynamic_int8_lora_injects_at_float_boundaries(comm_type, mlp_input) -> None:
    quantized_input = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2)
    gate_up_out = torch.randn(2, 6, dtype=torch.bfloat16)
    activated = torch.randn(2, 3, dtype=torch.bfloat16)
    quantized_activated = torch.ones(2, 3, dtype=torch.int8)
    activated_scale = torch.ones(2)
    down_out = torch.randn(2, 4, dtype=torch.bfloat16)
    routing = (torch.tensor([0, 1]), torch.tensor([0, 1]))
    event = object()
    stream = Mock(record_event=Mock(return_value=event))

    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch.object(
            DeviceOperator,
            "npu_dynamic_quant",
            side_effect=[(quantized_input, input_scale), (quantized_activated, activated_scale)],
        ) as dynamic_quant,
        patch(
            f"{QUANT_MOE}.torch_npu.npu_grouped_matmul",
            return_value=[gate_up_out],
            create=True,
        ) as gmm1,
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=activated),
        patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=down_out) as gmm2,
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather", return_value=routing) as recover_allgather,
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_all2all", return_value=routing) as recover_all2all,
        patch(f"{QUANT_MOE}.moe_lora_apply_w13") as apply_w13,
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as apply_w2,
        patch(f"{QUANT_MOE}.torch.npu.current_stream", return_value=stream),
    ):
        extra_ctx.moe_comm_type = comm_type
        output, output_event = quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    assert output is down_out
    assert output_event is event
    assert dynamic_quant.call_count == 2
    assert dynamic_quant.call_args_list[0].kwargs["hidden_states"] is mlp_input.hidden_states
    assert dynamic_quant.call_args_list[1].kwargs["hidden_states"] is activated
    assert gmm1.call_args.kwargs["x"][0] is quantized_input
    assert gmm2.call_args.kwargs["hidden_states"] is quantized_activated
    if comm_type == MoECommType.ALLGATHER:
        recover_allgather.assert_called_once_with(
            mlp_input.lora_context,
            mlp_input.expanded_row_idx,
            mlp_input.topk_ids,
        )
        recover_all2all.assert_not_called()
    else:
        recover_all2all.assert_called_once_with(
            mlp_input.lora_context,
            group_list=mlp_input.group_list,
        )
        recover_allgather.assert_not_called()
    apply_w13.assert_called_once_with(
        mlp_input.lora_context,
        gate_up_out=gate_up_out,
        hidden_states=mlp_input.hidden_states,
        lora_routing=routing,
    )
    apply_w2.assert_called_once_with(
        mlp_input.lora_context,
        down_out=down_out,
        silu_out=activated,
        lora_routing=routing,
    )


@pytest.mark.parametrize(
    ("comm_type", "mlp_input", "message"),
    [
        (MoECommType.FUSED_MC2, _make_input(), "AllGather TP"),
        (MoECommType.ALLGATHER, _make_input(dynamic_eplb=True), "dynamic EPLB"),
    ],
)
def test_dynamic_int8_lora_rejects_unsupported_modes(comm_type, mlp_input, message) -> None:
    with patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx:
        extra_ctx.moe_comm_type = comm_type
        with pytest.raises(NotImplementedError, match=message):
            quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)


def test_dynamic_int8_all2all_lora_handles_empty_ep_rank() -> None:
    mlp_input = _make_input(
        hidden_states=torch.empty(0, 4, dtype=torch.bfloat16),
        group_list=torch.zeros(2, dtype=torch.int64),
        expanded_row_idx=None,
        topk_ids=None,
        lora_context=SimpleNamespace(use_ep=True),
    )

    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch(f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant") as dynamic_quant,
    ):
        extra_ctx.moe_comm_type = MoECommType.ALLTOALL
        output, output_event = quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    assert output is mlp_input.hidden_states
    assert output_event is None
    dynamic_quant.assert_not_called()


def test_registered_backend_requires_float_input() -> None:
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    validate_quant_moe_lora_activation_input(
        quant_type=QuantType.W8A8,
        hidden_states=hidden_states,
        dynamic_scale=None,
    )
    with pytest.raises(NotImplementedError, match="unquantized activations"):
        validate_quant_moe_lora_activation_input(
            quant_type=QuantType.W8A8,
            hidden_states=hidden_states.to(torch.int8),
            dynamic_scale=torch.ones(2),
        )


def test_unregistered_quantized_moe_lora_fails_fast() -> None:
    with pytest.raises(NotImplementedError, match="no implementation registered"):
        validate_quant_moe_lora_activation_input(
            quant_type=QuantType.W4A8,
            hidden_states=torch.randn(2, 4),
            dynamic_scale=None,
        )


def test_duplicate_quant_moe_lora_registration_is_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register_quant_moe_lora_impl(QuantType.W8A8)
        def _duplicate(_mlp_compute_input):
            raise AssertionError("duplicate registration must fail before apply is stored")


def test_validate_skips_when_backend_has_no_activation_check() -> None:
    impl = QuantMoELoRAImpl(apply=lambda *_args, **_kwargs: None, validate_activation_input=None)
    with patch(f"{QUANT_MOE}._get_quant_moe_lora_impl", return_value=impl):
        validate_quant_moe_lora_activation_input(
            quant_type=QuantType.W8A8,
            hidden_states=torch.ones(1, 2, dtype=torch.int8),
            dynamic_scale=torch.ones(1),
        )


def test_dynamic_int8_lora_rejects_quantized_routed_activations() -> None:
    mlp_input = _make_input(
        hidden_states=torch.ones(2, 4, dtype=torch.int8),
        dynamic_scale=torch.ones(2),
    )
    with patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx:
        extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        with pytest.raises(AssertionError, match="BF16/FP16"):
            quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)


def test_dynamic_int8_lora_requires_allgather_routing_metadata() -> None:
    mlp_input = _make_input(expanded_row_idx=None, topk_ids=None)
    with patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx:
        extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        with pytest.raises(AssertionError, match="expanded_row_idx"):
            quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)


@pytest.mark.parametrize(
    ("weight_overrides", "message"),
    [
        ({"w1_scale_bias": torch.ones(1)}, "fused scale-bias"),
        ({"w1_offset": torch.ones(1)}, "antiquant offsets"),
        ({"w1_scale": None}, "weight scales"),
        (
            {
                "w1": [
                    torch.ones(1, 4, 6, dtype=torch.int8),
                    torch.ones(1, 4, 6, dtype=torch.int8),
                ]
            },
            "per-expert tensor lists",
        ),
    ],
)
def test_dynamic_int8_lora_rejects_unsupported_weight_layout(weight_overrides, message) -> None:
    weights_kwargs = dict(
        w1=[torch.ones(1, 4, 6, dtype=torch.int8)],
        w2=[torch.ones(1, 3, 4, dtype=torch.int8)],
        w1_scale=[torch.ones(1, 6)],
        w2_scale=[torch.ones(1, 4, dtype=torch.bfloat16)],
    )
    weights_kwargs.update(weight_overrides)
    mlp_input = _make_input(weights=MoEWeights(**weights_kwargs))
    with patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx:
        extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        with pytest.raises((NotImplementedError, AssertionError), match=message):
            quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)


def test_dynamic_int8_lora_applies_topk_scales_before_down_proj() -> None:
    activated = torch.ones(2, 3, dtype=torch.bfloat16)
    topk_scales = torch.tensor([[2.0], [3.0]], dtype=torch.bfloat16)
    mlp_input = _make_input(topk_scales=topk_scales)
    routing = (torch.tensor([0, 1]), torch.tensor([0, 1]))
    stream = Mock(record_event=Mock(return_value=object()))

    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch(
            f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant",
            side_effect=[
                (torch.ones(2, 4, dtype=torch.int8), torch.ones(2)),
                (torch.ones(2, 3, dtype=torch.int8), torch.ones(2)),
            ],
        ),
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[torch.randn(2, 6)], create=True),
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=activated),
        patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=torch.randn(2, 4)),
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather", return_value=routing),
        patch(f"{QUANT_MOE}.moe_lora_apply_w13"),
        patch(f"{QUANT_MOE}.moe_lora_apply_w2"),
        patch("torch.npu.current_stream", return_value=stream),
    ):
        extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    torch.testing.assert_close(activated, torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.bfloat16))


def test_apply_moe_activation_dispatches_known_kernels() -> None:
    gate_up = torch.randn(2, 6)
    with patch(f"{QUANT_MOE}.AscendSwigluOAIAndMul.swiglu_oai_forward", return_value="oai") as oai:
        assert _apply_moe_activation(gate_up, MoEActivation.SWIGLUOAI, 0.0, 1.0, 0.0) == "oai"
        oai.assert_called_once_with(gate_up)
    with patch(
        f"{QUANT_MOE}.torch_npu.npu_clipped_swiglu",
        return_value="uninterleave",
        create=True,
    ) as clipped:
        assert _apply_moe_activation(gate_up, "swigluoai_uninterleave", 4.0, 1.5, 0.2) == "uninterleave"
        clipped.assert_called_once()
    with patch(f"{QUANT_MOE}.AscendSwigluStepAndMul.swiglustep_forward", return_value="step") as step:
        assert _apply_moe_activation(gate_up, MoEActivation.SWIGLUSTEP, 0.0, 1.0, 0.0) == "step"
        assert step.call_args.kwargs["limit"] == 7.0

    gelu_out = _apply_moe_activation(torch.ones(2, 4), MoEActivation.GELU, 0.0, 1.0, 0.0)
    assert gelu_out.shape == (2, 2)
    tanh_out = _apply_moe_activation(torch.ones(2, 4), MoEActivation.GELU_TANH, 0.0, 1.0, 0.0)
    assert tanh_out.shape == (2, 2)

    clamped = torch.tensor([[10.0, 10.0, -10.0, 10.0]])
    with patch(f"{QUANT_MOE}.torch_npu.npu_swiglu", side_effect=lambda x: x, create=True) as swiglu:
        out = _apply_moe_activation(clamped, "silu", 2.0, 1.0, 0.0)
    swiglu.assert_called_once()
    torch.testing.assert_close(out, torch.tensor([[2.0, 2.0, -2.0, 2.0]]))


def test_mc2_comm_type_is_unsupported() -> None:
    with patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx:
        extra_ctx.moe_comm_type = MoECommType.MC2
        with pytest.raises(NotImplementedError, match="AllGather TP"):
            quant_apply_mlp_with_moe_lora(mlp_compute_input=_make_input())
