from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.quant_moe import (
    _add_composite_lora_gmm,
    _add_single_lora_gmm,
    _build_composite_lora_gmm_routing,
    _build_single_lora_gmm_routing,
    _can_use_composite_lora_gmm,
    _can_use_single_lora_gmm,
    _CompositeLoraGMMRouting,
    _SingleLoraGMMRouting,
    quant_apply_mlp_with_moe_lora,
    validate_quant_moe_lora_activation_input,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput, MoEQuantParams, MoEWeights
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
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch(
            f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant",
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
    ):
        extra_ctx.moe_comm_type = comm_type
        output, output_event = quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    assert output is down_out
    assert output_event is None
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


def test_dynamic_int8_uses_single_lora_gmm_without_recovering_routing() -> None:
    lora_context = SimpleNamespace(
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
    )
    topk_scales = torch.tensor([[0.25], [0.5]], dtype=torch.bfloat16)
    mlp_input = _make_input(lora_context=lora_context, topk_scales=topk_scales)
    quantized_input = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2)
    gate_up_out = torch.randn(2, 6, dtype=torch.bfloat16)
    activation_before_scale = torch.ones(2, 3, dtype=torch.bfloat16)
    quantized_activated = torch.ones(2, 3, dtype=torch.int8)
    activated_scale = torch.ones(2)
    down_out = torch.randn(2, 4, dtype=torch.bfloat16)
    single_routing = SimpleNamespace(slot="slot", enabled="enabled")
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch(
            f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant",
            side_effect=[(quantized_input, input_scale), (quantized_activated, activated_scale)],
        ),
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True),
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=activation_before_scale),
        patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=down_out),
        patch(f"{QUANT_MOE}._can_use_single_lora_gmm", return_value=True),
        patch(
            f"{QUANT_MOE}._build_single_lora_gmm_routing",
            return_value=single_routing,
        ) as build_routing,
        patch(f"{QUANT_MOE}._add_single_lora_gmm") as add_gmm,
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather") as recover,
        patch(f"{QUANT_MOE}.moe_lora_apply_w13") as apply_w13,
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as apply_w2,
        patch(f"{QUANT_MOE}.reset_lora_indices") as reset_indices,
    ):
        extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    assert add_gmm.call_count == 2
    build_routing.assert_called_once_with(
        lora_context,
        expanded_row_idx=mlp_input.expanded_row_idx,
        topk_ids=mlp_input.topk_ids,
    )
    assert add_gmm.call_args_list[0].args[0] is gate_up_out
    assert add_gmm.call_args_list[0].args[1] is mlp_input.hidden_states
    assert add_gmm.call_args_list[0].kwargs["routing"] is single_routing
    assert torch.equal(add_gmm.call_args_list[1].args[1], topk_scales.expand(-1, 3))
    assert add_gmm.call_args_list[1].kwargs["routing"] is single_routing
    recover.assert_not_called()
    apply_w13.assert_not_called()
    apply_w2.assert_not_called()
    reset_indices.assert_called_once_with(lora_context)


def test_dynamic_int8_uses_composite_gmm_without_recovering_routing() -> None:
    lora_context = SimpleNamespace(
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
    )
    mlp_input = _make_input(lora_context=lora_context)
    quantized_input = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2)
    gate_up_out = torch.randn(2, 6, dtype=torch.bfloat16)
    activated = torch.ones(2, 3, dtype=torch.bfloat16)
    quantized_activated = torch.ones(2, 3, dtype=torch.int8)
    activated_scale = torch.ones(2)
    down_out = torch.randn(2, 4, dtype=torch.bfloat16)
    composite_routing = SimpleNamespace(permutation="permutation")
    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch(
            f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant",
            side_effect=[(quantized_input, input_scale), (quantized_activated, activated_scale)],
        ),
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True),
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=activated),
        patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=down_out),
        patch(f"{QUANT_MOE}._can_use_single_lora_gmm", return_value=False),
        patch(f"{QUANT_MOE}._can_use_composite_lora_gmm", return_value=True),
        patch(
            f"{QUANT_MOE}._build_composite_lora_gmm_routing",
            return_value=composite_routing,
        ) as build_routing,
        patch(f"{QUANT_MOE}._add_composite_lora_gmm") as add_gmm,
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather") as recover,
        patch(f"{QUANT_MOE}.moe_lora_apply_w13") as apply_w13,
        patch(f"{QUANT_MOE}.moe_lora_apply_w2") as apply_w2,
        patch(f"{QUANT_MOE}.reset_lora_indices") as reset_indices,
    ):
        extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    build_routing.assert_called_once_with(
        lora_context,
        expanded_row_idx=mlp_input.expanded_row_idx,
        topk_ids=mlp_input.topk_ids,
        num_experts=mlp_input.group_list.numel(),
    )
    assert add_gmm.call_count == 2
    assert add_gmm.call_args_list[0].args[0] is gate_up_out
    assert add_gmm.call_args_list[0].args[1] is mlp_input.hidden_states
    assert add_gmm.call_args_list[0].kwargs["routing"] is composite_routing
    assert add_gmm.call_args_list[1].args[0] is down_out
    assert add_gmm.call_args_list[1].args[1] is activated
    assert add_gmm.call_args_list[1].kwargs["routing"] is composite_routing
    recover.assert_not_called()
    apply_w13.assert_not_called()
    apply_w2.assert_not_called()
    reset_indices.assert_called_once_with(lora_context)


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


def _make_gmm_lora_context(num_experts: int = 2):
    rank = 16
    max_loras = 3
    hidden_size = 4
    intermediate_size = 3
    return SimpleNamespace(
        punica_wrapper=SimpleNamespace(is_prefill=True, num_active_moe_loras=1, no_lora=False),
        adapter_enabled=torch.tensor([1, 0, 0, 0], dtype=torch.int32),
        fully_sharded=False,
        top_k=6,
        w13_lora_a_stacked=(
            torch.zeros(max_loras, num_experts, rank, hidden_size, dtype=torch.bfloat16),
            torch.zeros(max_loras, num_experts, rank, hidden_size, dtype=torch.bfloat16),
        ),
        w13_lora_b_stacked=(
            torch.zeros(max_loras, num_experts, intermediate_size, rank, dtype=torch.bfloat16),
            torch.zeros(max_loras, num_experts, intermediate_size, rank, dtype=torch.bfloat16),
        ),
        w2_lora_a_stacked=(torch.zeros(max_loras, num_experts, rank, intermediate_size, dtype=torch.bfloat16),),
        w2_lora_b_stacked=(torch.zeros(max_loras, num_experts, hidden_size, rank, dtype=torch.bfloat16),),
    )


def test_single_lora_gmm_checks_fixed_fast_path_shape() -> None:
    context = _make_gmm_lora_context()
    # V1 sets LoRAMapping.is_prefill=True to select SGMV on non-CUDA
    # platforms, so it is not a real scheduler-phase signal. GMM eligibility
    # must remain valid when a caller supplies the actual decode value.
    context.punica_wrapper.is_prefill = False
    group_list = torch.tensor([8, 8], dtype=torch.int64)
    hidden_states = torch.zeros(16, 4, dtype=torch.bfloat16)

    assert (
        _can_use_single_lora_gmm(
            context,
            hidden_states=hidden_states,
            group_list=group_list,
            group_list_type=1,
        )
        is True
    )
    assert (
        _can_use_single_lora_gmm(
            context,
            hidden_states=hidden_states[:15],
            group_list=group_list,
            group_list_type=1,
        )
        is False
    )
    context.fully_sharded = True
    assert (
        _can_use_single_lora_gmm(
            context,
            hidden_states=hidden_states,
            group_list=group_list,
            group_list_type=1,
        )
        is False
    )


def test_composite_lora_gmm_checks_mixed_requests_and_minimum_rows() -> None:
    context = _make_gmm_lora_context()
    context.punica_wrapper.num_active_moe_loras = 2
    context.punica_wrapper.is_prefill = False
    group_list = torch.tensor([24, 24], dtype=torch.int64)
    hidden_states = torch.zeros(48, 4, dtype=torch.bfloat16)

    assert _can_use_composite_lora_gmm(
        context,
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=1,
    )
    assert not _can_use_composite_lora_gmm(
        context,
        hidden_states=hidden_states[:47],
        group_list=group_list,
        group_list_type=1,
    )
    context.punica_wrapper.no_lora = True
    assert not _can_use_composite_lora_gmm(
        context,
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=1,
    )


def test_build_single_lora_gmm_routing_handles_base_rows() -> None:
    context = SimpleNamespace(
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([2, -1])),
        adapter_enabled=torch.tensor([0, 0, 1, 0], dtype=torch.int32),
        w13_lora_a_stacked=(torch.empty(3, 2, 16, 4),),
    )
    topk_ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    expanded_row_idx = torch.tensor([2, 0, 1, 3], dtype=torch.int32)

    routing = _build_single_lora_gmm_routing(
        context,
        expanded_row_idx=expanded_row_idx,
        topk_ids=topk_ids,
    )

    assert torch.equal(routing.slot, torch.tensor([2]))
    assert torch.equal(routing.enabled, torch.tensor([True, False, True, False]))


def test_build_composite_lora_gmm_routing_handles_multiple_slots_and_base() -> None:
    context = SimpleNamespace(
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([2, -1, 0])),
        adapter_enabled=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
        w13_lora_a_stacked=(torch.empty(3, 2, 16, 4),),
    )
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int32)
    expanded_row_idx = torch.tensor([4, 1, 0, 3, 5, 2], dtype=torch.int32)

    routing = _build_composite_lora_gmm_routing(
        context,
        expanded_row_idx=expanded_row_idx,
        topk_ids=topk_ids,
        num_experts=2,
    )

    dispatched_groups = torch.tensor([0, 4, 0, 1, 5, 1])
    dispatched_enabled = torch.tensor([False, True, True, False, True, True])
    assert torch.equal(dispatched_groups.index_select(0, routing.permutation), torch.tensor([0, 0, 1, 1, 4, 5]))
    assert torch.equal(routing.group_list, torch.tensor([2, 2, 0, 0, 1, 1]))
    assert torch.equal(routing.enabled, dispatched_enabled.index_select(0, routing.permutation))


def test_composite_lora_gmm_reorders_masks_and_restores_rows() -> None:
    output = torch.zeros(3, 2)
    inputs = torch.tensor([[1.0], [2.0], [3.0]])
    lora_a = (torch.zeros(3, 2, 16, 1),)
    lora_b = (torch.zeros(3, 2, 2, 16),)
    routing = _CompositeLoraGMMRouting(
        permutation=torch.tensor([2, 0, 1]),
        group_list=torch.tensor([1, 1, 1, 0, 0, 0]),
        enabled=torch.tensor([True, False, True]),
    )
    shrink = torch.zeros(3, 16)
    delta = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])

    with patch(
        f"{QUANT_MOE}._grouped_lora_matmul",
        side_effect=[shrink, delta],
    ) as grouped_matmul:
        _add_composite_lora_gmm(
            output,
            inputs,
            lora_a,
            lora_b,
            routing=routing,
        )

    assert torch.equal(grouped_matmul.call_args_list[0].args[0], torch.tensor([[3.0], [0.0], [2.0]]))
    assert grouped_matmul.call_args_list[0].args[1].shape == (6, 16, 1)
    assert grouped_matmul.call_args_list[1].args[1].shape == (6, 2, 16)
    assert torch.equal(output, torch.tensor([[0.0, 0.0], [30.0, 31.0], [10.0, 11.0]]))


@pytest.mark.parametrize("slot", [0, 2])
def test_single_lora_gmm_selects_device_slot_and_adds_each_output_slice(slot: int) -> None:
    output = torch.zeros(2, 5, dtype=torch.bfloat16)
    inputs = torch.ones(2, 4, dtype=torch.bfloat16)
    lora_a = tuple(
        torch.stack([torch.full(shape, value, dtype=torch.bfloat16) for value in range(3)])
        for shape in ((2, 16, 4), (2, 16, 4))
    )
    lora_b = tuple(
        torch.stack([torch.full(shape, value, dtype=torch.bfloat16) for value in range(3)])
        for shape in ((2, 2, 16), (2, 3, 16))
    )
    group_list = torch.tensor([1, 1], dtype=torch.int64)
    shrink = torch.zeros(2, 16, dtype=torch.bfloat16)
    routing = _SingleLoraGMMRouting(
        slot=torch.tensor([slot], dtype=torch.long),
        enabled=torch.tensor([True, False]),
    )

    with patch(
        f"{QUANT_MOE}._grouped_lora_matmul",
        side_effect=[
            shrink,
            torch.ones(2, 2, dtype=torch.bfloat16),
            shrink,
            torch.full((2, 3), 2, dtype=torch.bfloat16),
        ],
    ) as grouped_matmul:
        _add_single_lora_gmm(
            output,
            inputs,
            lora_a,
            lora_b,
            routing=routing,
            group_list=group_list,
        )

    assert grouped_matmul.call_count == 4
    assert torch.equal(
        grouped_matmul.call_args_list[0].args[0],
        torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bfloat16),
    )
    for call in grouped_matmul.call_args_list:
        assert torch.all(call.args[1] == slot)
    assert torch.equal(output[0, :2], torch.ones(2, dtype=torch.bfloat16))
    assert torch.equal(output[0, 2:], torch.full((3,), 2, dtype=torch.bfloat16))
    assert torch.equal(output[1], torch.zeros(5, dtype=torch.bfloat16))
