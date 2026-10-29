# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from dataclasses import dataclass
from types import MethodType

import torch

from vllm_ascend.lora.fused_moe import has_lora
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.shared_experts import (
    AscendSharedExperts,
    LinearInput,
    PreparedSharedExpertExecution,
    SharedExpertMLPPath,
    SharedExpertParallelMode,
    SubmittedSharedExpertGateUp,
)
from vllm_ascend.utils import npu_stream_switch, shared_experts_calculation_stream


@dataclass(frozen=True)
class SubmittedSharedExpertWork310(SubmittedSharedExpertGateUp):
    """Per-forward work queued before routed TopK/dispatch, never cached."""

    down_input: LinearInput
    output_gate: torch.Tensor | None


def _select_mlp_path_310(self: AscendSharedExperts) -> SharedExpertMLPPath:
    # 310P schemes own input scales/offsets, quantization bias, and NZ
    # layouts. The standard A8 fused path bypasses these contracts and
    # requires scale fields that the compatibility schemes do not create.
    return SharedExpertMLPPath.LINEAR_WRAPPER


def _enqueue_gate_up_after_router_310(
    self: AscendSharedExperts,
    prepared_execution: PreparedSharedExpertExecution,
    router_output_ready: torch.npu.Event | None,
) -> SubmittedSharedExpertGateUp:
    submitted = AscendSharedExperts.enqueue_gate_up_after_router(self, prepared_execution, router_output_ready)
    if (
        not self.multistream_overlap
        or has_lora(self.lora_context)
        or self.parallel_mode() is not SharedExpertParallelMode.TENSOR_PARALLEL
    ):
        return submitted

    # Gate-Up has already waited for router_output_ready on this same stream.
    # Queue activation now, including in eager prefill: appending it only after
    # routed.forward_impl returns misses the host-side submission window.
    with npu_stream_switch(shared_experts_calculation_stream(), enabled=True):
        shared_act = self.apply_activation(submitted.gate_up)
        down_input = self._prepare_linear_input_for_overlap(self.layer.down_proj, shared_act)
        output_gate = None
        expert_gate = getattr(self.layer, "expert_gate", None)
        if expert_gate is not None:
            gate_out, _ = expert_gate(submitted.hidden_states)
            output_gate = torch.sigmoid(gate_out)

    return SubmittedSharedExpertWork310(
        hidden_states=submitted.hidden_states,
        gate_up=submitted.gate_up,
        gate_up_input_scale=submitted.gate_up_input_scale,
        local_dp_metadata=submitted.local_dp_metadata,
        path=submitted.path,
        down_input=down_input,
        output_gate=output_gate,
    )


def _finish_linear_wrapped_mlp_310(
    self: AscendSharedExperts,
    submitted_gate_up: SubmittedSharedExpertGateUp,
    milestones: RoutedMoEMilestones,
) -> torch.Tensor:
    if not isinstance(submitted_gate_up, SubmittedSharedExpertWork310):
        return AscendSharedExperts._finish_linear_wrapped_mlp(self, submitted_gate_up, milestones)

    # Preserve the conservative Down/GMM2 boundary and the common final join.
    self._wait_for_routed_stage(milestones, milestones.routed_combine_start, "routed_combine_start")
    shared_out, _ = self.layer.down_proj(submitted_gate_up.down_input)
    if submitted_gate_up.output_gate is not None:
        shared_out = submitted_gate_up.output_gate * shared_out
    return shared_out


def configure_shared_experts_310(shared_experts: AscendSharedExperts) -> None:
    """Adapt one executor without replacing it or its weight-validation hook.

    Binding the selector on the instance leaves non-310P executors and the
    common class unchanged. The hook created by the parent continues to
    validate the same executor with its 310P-compatible MLP path.
    """
    shared_experts._select_mlp_path = MethodType(_select_mlp_path_310, shared_experts)  # type: ignore[method-assign]
    shared_experts.enqueue_gate_up_after_router = MethodType(  # type: ignore[method-assign]
        _enqueue_gate_up_after_router_310, shared_experts
    )
    shared_experts._finish_linear_wrapped_mlp = MethodType(  # type: ignore[method-assign]
        _finish_linear_wrapped_mlp_310, shared_experts
    )
