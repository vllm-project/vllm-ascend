from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.quant_moe import (
    _can_use_moe_lora_aux_stream,
    _execute_moe_lora_in_parallel,
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
    event = object()
    stream = Mock(record_event=Mock(return_value=event))

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
        patch("torch.npu.current_stream", return_value=stream),
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


def test_dynamic_int8_allgather_lora_overlaps_into_separate_deltas() -> None:
    events = tuple(object() for _ in range(4))
    aux_stream = object()
    lora_context = SimpleNamespace(
        use_ep=False,
        fully_sharded=False,
        aux_stream=aux_stream,
        events=events,
        w13_lora_b_stacked=(torch.empty(1, 2, 3, 1), torch.empty(1, 2, 3, 1)),
        w2_lora_b_stacked=(torch.empty(1, 2, 4, 1),),
    )
    mlp_input = _make_input(lora_context=lora_context)
    quantized_input = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2)
    gate_up_out = torch.zeros(2, 6, dtype=torch.bfloat16)
    activated = torch.randn(2, 3, dtype=torch.bfloat16)
    quantized_activated = torch.ones(2, 3, dtype=torch.int8)
    activated_scale = torch.ones(2)
    down_out = torch.zeros(2, 4, dtype=torch.bfloat16)
    routing = (torch.tensor([0, 1]), torch.tensor([0, 1]))
    before_gmm2_event = object()
    stream = Mock(record_event=Mock(return_value=before_gmm2_event))

    def execute(base_fn, lora_fn, *_):
        base_result = base_fn()
        lora_fn()
        return base_result, None

    def apply_w13_delta(*_, gate_up_out, **__):
        assert torch.count_nonzero(gate_up_out) == 0
        gate_up_out.add_(2)

    def apply_w2_delta(*_, down_out, **__):
        assert torch.count_nonzero(down_out) == 0
        down_out.add_(3)

    with (
        patch(f"{QUANT_MOE}._EXTRA_CTX") as extra_ctx,
        patch(
            f"{QUANT_MOE}.DeviceOperator.npu_dynamic_quant",
            side_effect=[(quantized_input, input_scale), (quantized_activated, activated_scale)],
        ),
        patch(f"{QUANT_MOE}.torch_npu.npu_grouped_matmul", return_value=[gate_up_out], create=True),
        patch(f"{QUANT_MOE}._apply_moe_activation", return_value=activated) as activation,
        patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", return_value=down_out),
        patch(f"{QUANT_MOE}._recover_moe_lora_routing_allgather", return_value=routing),
        patch(f"{QUANT_MOE}._can_use_moe_lora_aux_stream", return_value=True),
        patch(f"{QUANT_MOE}._execute_moe_lora_in_parallel", side_effect=execute) as parallel,
        patch(f"{QUANT_MOE}.moe_lora_apply_w13", side_effect=apply_w13_delta) as apply_w13,
        patch(f"{QUANT_MOE}.moe_lora_apply_w2", side_effect=apply_w2_delta) as apply_w2,
        patch("torch.npu.current_stream", return_value=stream),
    ):
        extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        output, output_event = quant_apply_mlp_with_moe_lora(mlp_compute_input=mlp_input)

    assert output is down_out
    assert output_event is before_gmm2_event
    assert torch.all(output == 3)
    assert parallel.call_count == 2
    assert parallel.call_args_list[0].args[2:] == (events[0], events[1], aux_stream)
    assert parallel.call_args_list[1].args[2:] == (events[2], events[3], aux_stream)
    assert apply_w13.call_args.kwargs["gate_up_out"] is not gate_up_out
    assert apply_w2.call_args.kwargs["down_out"] is not down_out
    assert apply_w13.call_args.kwargs["gate_up_out"].data_ptr() == apply_w2.call_args.kwargs["down_out"].data_ptr()
    assert torch.all(activation.call_args.args[0] == 2)


def test_moe_lora_aux_stream_gating() -> None:
    context = SimpleNamespace(
        use_ep=False,
        fully_sharded=False,
        aux_stream=object(),
        events=tuple(object() for _ in range(4)),
    )
    assert _can_use_moe_lora_aux_stream(context, MoECommType.ALLGATHER)
    assert not _can_use_moe_lora_aux_stream(context, MoECommType.ALLTOALL)

    context.fully_sharded = True
    assert not _can_use_moe_lora_aux_stream(context, MoECommType.ALLGATHER)
    context.fully_sharded = False

    # MoE is an opaque custom op. Its real implementation is invoked during
    # ACLGraph capture, where the pre-created stream/event fork and join must
    # remain active so both streams are recorded into the graph.
    with (
        patch("torch.compiler.is_compiling", return_value=True),
        patch("torch.npu.is_current_stream_capturing", return_value=True),
    ):
        assert _can_use_moe_lora_aux_stream(context, MoECommType.ALLGATHER)


def test_execute_moe_lora_in_parallel_orders_npu_events() -> None:
    main_stream = Mock()
    aux_stream = Mock()
    start_event = Mock()
    done_event = Mock()
    base_result = torch.tensor([1])
    calls = []

    def base_fn():
        calls.append("base")
        return base_result

    def lora_fn():
        calls.append("lora")

    with (
        patch("torch.npu.current_stream", return_value=main_stream),
        patch(f"{QUANT_MOE}.npu_stream_switch", return_value=nullcontext()),
    ):
        result, lora_result = _execute_moe_lora_in_parallel(
            base_fn,
            lora_fn,
            start_event,
            done_event,
            aux_stream,
        )

    assert result is base_result
    assert lora_result is None
    assert calls == ["base", "lora"]
    start_event.record.assert_called_once_with(main_stream)
    aux_stream.wait_event.assert_called_once_with(start_event)
    done_event.record.assert_called_once_with(aux_stream)
    main_stream.wait_event.assert_called_once_with(done_event)


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
def test_execute_moe_lora_in_parallel_supports_aclgraph_replay() -> None:
    static_input = torch.ones(16, device="npu", dtype=torch.float32)
    aux_stream = torch.npu.Stream()
    events = tuple(torch.npu.Event() for _ in range(2))

    def run_parallel() -> torch.Tensor:
        lora_delta = torch.empty_like(static_input)

        def base_fn() -> torch.Tensor:
            return static_input * 2

        def lora_fn() -> None:
            lora_delta.copy_(static_input * 3)

        base_output, _ = _execute_moe_lora_in_parallel(
            base_fn,
            lora_fn,
            events[0],
            events[1],
            aux_stream,
        )
        return base_output.add_(lora_delta)

    # Initialize lazy stream/event resources before capture, matching model
    # warmup followed by ACLGraph capture in the runner.
    run_parallel()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run_parallel()

    static_input.fill_(2)
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(output.cpu(), torch.full((16,), 10.0))


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
