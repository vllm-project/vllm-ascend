# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from vllm_ascend.ops.fused_moe import shared_experts as shared_experts_module
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import PreparedSharedExpertInput, RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.shared_experts import (
    AscendSharedExperts,
    PreparedSharedExpertExecution,
    SharedExpertMLPPath,
    SharedExpertParallelMode,
)


class IdentityOnlyEvent:
    def __eq__(self, other):
        raise AssertionError("Events must be compared by identity")


@pytest.fixture
def ready_event_executor(monkeypatch):
    executor = AscendSharedExperts.__new__(AscendSharedExperts)
    executor.multistream_overlap = True
    executor.parallel_mode = lambda: SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
    executor._select_mlp_path = lambda: SharedExpertMLPPath.LINEAR_WRAPPER
    states = torch.randn(2, 4)
    output = torch.randn(2, 4)
    order = []
    stream = SimpleNamespace(wait_event=MagicMock(side_effect=lambda _: order.append("wait")))
    monkeypatch.setattr(shared_experts_module, "npu_stream_switch", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(shared_experts_module, "shared_experts_calculation_stream", lambda: stream)
    monkeypatch.setattr(shared_experts_module.torch.npu, "current_stream", lambda: stream)
    monkeypatch.setattr(shared_experts_module, "_EXTRA_CTX", SimpleNamespace(num_tokens=2))

    def prepare(x):
        order.append("prepare")
        if executor._select_mlp_path() in (SharedExpertMLPPath.A8_INT_FUSED, SharedExpertMLPPath.A8_MXFP_FUSED):
            return x, torch.ones(2)
        return x

    def gate_up(*args, **kwargs):
        order.append("gate_up")
        return states

    executor._prepare_gate_up_input = prepare
    executor.part1 = gate_up
    executor.layer = SimpleNamespace(
        gate_up_proj=MagicMock(side_effect=lambda *args: (gate_up(), None)),
    )
    monkeypatch.setattr(shared_experts_module.torch_npu, "npu_quant_matmul", gate_up, raising=False)
    executor._finish_shared_mlp = MagicMock(return_value=output)
    executor._pad_and_reduce_scatter = MagicMock(return_value=output)
    executor.wait_for_output = MagicMock()
    return SimpleNamespace(executor=executor, states=states, output=output, stream=stream, order=order)


@pytest.mark.parametrize("path", list(SharedExpertMLPPath))
@pytest.mark.parametrize("split_submission", [False, True])
@pytest.mark.parametrize("router_event_kind", ["same", "distinct", "missing"])
def test_shared_ready_wait_is_deduplicated_per_preparation(
    ready_event_executor, path, split_submission, router_event_kind
):
    case = ready_event_executor
    executor = case.executor
    executor._select_mlp_path = lambda: path

    # Repeat on the same executor: a previous invocation must never suppress
    # this invocation's first wait, even if its input event object is reused.
    input_event = IdentityOnlyEvent()
    for _ in range(2):
        case.order.clear()
        case.stream.wait_event.reset_mock()
        router_event = input_event if router_event_kind == "same" else IdentityOnlyEvent()
        if router_event_kind == "missing":
            router_event = None
        finalize_event = IdentityOnlyEvent()
        milestones = RoutedMoEMilestones(
            shared_input_ready=input_event,
            router_output_ready=router_event,
            routed_finalize_done=finalize_event,
        )
        prepared = PreparedSharedExpertInput(case.states, is_gathered=True)

        def run(prepared=prepared, router_event=router_event, milestones=milestones):
            if split_submission:
                execution = executor.prepare_for_router_overlap(prepared, input_event)
                assert execution.input_ready_event is input_event
                submitted = executor.enqueue_gate_up_after_router(execution, router_event)
                return executor.forward(prepared, milestones, defer_output_wait=True, submitted_gate_up=submitted)
            return executor.forward(prepared, milestones, defer_output_wait=True)

        if router_event_kind == "missing":
            with pytest.raises(RuntimeError, match="Missing router_output_ready"):
                run()
            assert case.order == ["wait", "prepare"]
            continue

        assert run() is case.output
        expected = [call(input_event)]
        expected_order = ["wait", "prepare"]
        if router_event_kind == "distinct":
            expected.append(call(router_event))
            expected_order.append("wait")
        expected.extend([call(finalize_event)])
        expected_order.extend(["gate_up", "wait"])
        assert case.stream.wait_event.call_args_list == expected
        assert case.order == expected_order
        executor.wait_for_output.assert_not_called()


@pytest.mark.parametrize("path", list(SharedExpertMLPPath))
def test_shared_ready_wait_with_overlap_disabled(ready_event_executor, path):
    case = ready_event_executor
    case.executor.multistream_overlap = False
    case.executor._select_mlp_path = lambda: path
    result = case.executor.forward(PreparedSharedExpertInput(case.states, is_gathered=True), RoutedMoEMilestones())
    assert result is case.output
    assert case.order == ["prepare", "gate_up"]
    case.stream.wait_event.assert_not_called()


def test_gate_up_without_preparation_provenance_keeps_router_wait(ready_event_executor):
    case = ready_event_executor
    event = IdentityOnlyEvent()
    prepared = PreparedSharedExpertExecution(case.states, case.states, None)
    case.executor.enqueue_gate_up_after_router(prepared, event)
    case.stream.wait_event.assert_called_once_with(event)
    assert case.order == ["wait", "gate_up"]
