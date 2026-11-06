# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend._310p.fused_moe.shared_experts import (
    SubmittedSharedExpertWork310,
    configure_shared_experts_310,
)
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.shared_experts import (
    AscendSharedExperts,
    PreparedSharedExpertExecution,
    SharedExpertMLPPath,
    SharedExpertParallelMode,
)


def build_executor(with_gate=True, device="cpu", dtype=torch.float32):
    """Use tuple-returning linear wrappers, like registered vLLM layers."""
    torch.manual_seed(310)
    weights = [torch.randn(*shape, device=device, dtype=dtype) * 0.1 for shape in ((16, 8), (8, 8), (1, 8))]
    executor = AscendSharedExperts.__new__(AscendSharedExperts)
    executor.layer = SimpleNamespace(
        gate_up_proj=lambda x: (F.linear(x, weights[0]), None),
        down_proj=lambda x: (F.linear(x, weights[1]), None),
        expert_gate=(lambda x: (F.linear(x, weights[2]), None)) if with_gate else None,
        act_fn=lambda x: F.silu(x[..., :8]) * x[..., 8:],
    )
    executor.lora_context = None
    executor.multistream_overlap = True
    executor.parallel_mode = lambda: SharedExpertParallelMode.TENSOR_PARALLEL
    configure_shared_experts_310(executor)
    return executor


@pytest.mark.parametrize("batch", [1, 2, 16])
@pytest.mark.parametrize("with_gate", [False, True])
def test_early_work_matches_serial_and_preserves_join(batch, with_gate):
    executor = build_executor(with_gate)
    x = torch.randn(batch, 8)
    expected = executor.part2(x, executor.apply_activation(executor.part1(x)))
    prepared = PreparedSharedExpertExecution(x, x, None)
    router, combine = object(), object()
    milestones = RoutedMoEMilestones(routed_combine_start=combine)
    with ExitStack() as stack:
        for module in ("vllm_ascend.ops.fused_moe.shared_experts", "vllm_ascend._310p.fused_moe.shared_experts"):
            stack.enter_context(patch(module + ".npu_stream_switch", side_effect=lambda *a, **kw: nullcontext()))
            stack.enter_context(patch(module + ".shared_experts_calculation_stream", return_value=object()))
        wait = stack.enter_context(patch.object(executor, "_wait_for_milestone"))
        activation = stack.enter_context(patch.object(executor, "apply_activation", wraps=executor.apply_activation))
        gate = Mock(wraps=executor.layer.expert_gate) if with_gate else None
        executor.layer.expert_gate = gate
        submitted = executor.enqueue_gate_up_after_router(prepared, router)
        assert isinstance(submitted, SubmittedSharedExpertWork310)
        activation.assert_called_once()
        if gate is not None:
            gate.assert_called_once_with(x)
        wait.assert_called_once_with(router, "router_output_ready")
        actual = executor._finish_linear_wrapped_mlp(submitted, milestones)
        assert wait.call_args.args == (combine, "routed_combine_start")
        # Finishing must reuse activation/gate and must not evaluate them twice.
        activation.assert_called_once()
        if gate is not None:
            gate.assert_called_once()
        torch.testing.assert_close(actual, expected)
        other_x = x + 0.5
        other = executor.enqueue_gate_up_after_router(PreparedSharedExpertExecution(other_x, other_x, None), router)
        other_actual = executor._finish_linear_wrapped_mlp(other, milestones)
        other_expected = executor.part2(other_x, executor.apply_activation(executor.part1(other_x)))
        torch.testing.assert_close(other_actual, other_expected)


@pytest.mark.parametrize("fallback", ["disabled", "lora", "sp", "sedp"])
def test_unsupported_modes_keep_common_execution(fallback):
    executor = build_executor()
    if fallback == "disabled":
        executor.multistream_overlap = False
    if fallback == "sp":
        executor.parallel_mode = lambda: SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
    if fallback == "sedp":
        executor.parallel_mode = lambda: SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY
    prepared, submitted, milestones = object(), object(), object()
    with (
        patch("vllm_ascend._310p.fused_moe.shared_experts.has_lora", return_value=fallback == "lora"),
        patch.object(AscendSharedExperts, "enqueue_gate_up_after_router", return_value=submitted) as enqueue,
        patch.object(AscendSharedExperts, "_finish_linear_wrapped_mlp", return_value="common") as finish,
    ):
        result = executor.enqueue_gate_up_after_router(prepared, None)
        assert result is submitted
        assert executor._finish_linear_wrapped_mlp(result, milestones) == "common"
        enqueue.assert_called_once_with(executor, prepared, None)
        finish.assert_called_once_with(executor, submitted, milestones)


def test_configuration_preserves_executor_and_does_not_change_common_class():
    original_selector = AscendSharedExperts._select_mlp_path
    executor = build_executor()
    executor.layer.process_weights_after_loading = Mock()
    hook = executor.layer.process_weights_after_loading
    configure_shared_experts_310(executor)
    assert executor._select_mlp_path() is SharedExpertMLPPath.LINEAR_WRAPPER
    assert executor.layer.process_weights_after_loading is hook
    assert AscendSharedExperts._select_mlp_path is original_selector
