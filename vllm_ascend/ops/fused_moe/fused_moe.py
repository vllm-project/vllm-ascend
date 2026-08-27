#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
import torch.nn.functional as F
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoERouter
from vllm.model_executor.layers.fused_moe.layer import MoERunner

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method, setup_moe_comm_method
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.shared_experts import (
    AscendSharedExperts,
    SharedExpertParallelMode,
)


class AscendMoERunner(MoERunner):  # type: ignore[no-redef]
    def __init__(
        self,
        layer_name,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: AscendRoutedExperts,
        enable_dbo=False,
        gate=None,
        shared_experts=None,
        shared_expert_gate=None,
        routed_input_transform=None,
        routed_output_transform=None,
        routed_scaling_factor=1,
    ):
        super().__init__(
            layer_name,
            moe_config,
            router,
            routed_experts,
            enable_dbo,
            gate,
            shared_experts,
            shared_expert_gate,
            routed_input_transform,
            routed_output_transform,
            routed_scaling_factor,
        )
        self._gate = gate
        self.hidden_size = moe_config.hidden_dim

        self.quant_type = routed_experts.quant_type
        self.routed_experts.router = router

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        if self.moe_config.ep_size > 1:
            self.moe_config.ep_group = get_ep_group()
            self.moe_config.mc2_group = get_mc2_group()

        self.ascend_shared_experts = None
        if shared_experts is not None:
            routed_experts.return_with_event = True
            self.ascend_shared_experts = AscendSharedExperts(
                shared_experts,
                self.moe_config,
                self.quant_type,
                self._quant_method,
            )

        setup_moe_comm_method(self.moe_config)
        alltoall_comm = get_moe_comm_method(MoECommType.ALLTOALL)
        if alltoall_comm is not None:
            expert_ids_per_ep_rank = getattr(alltoall_comm.token_dispatcher, "expert_ids_per_ep_rank", None)
            if expert_ids_per_ep_rank is not None:
                self.routed_experts.register_buffer(
                    "expert_ids_per_ep_rank",
                    expert_ids_per_ep_rank,
                    persistent=False,
                )

    @property
    def is_internal_router(self) -> bool:
        # main (cdc4824a21): vllm#51838 removed the gate branch in
        # DeepseekV2MoE.forward, always passing router_logits=hidden_states.
        # The runner must recompute router_logits via the gate.
        return self.gate is not None

    @property
    def use_dp_chunking(self) -> bool:
        """Ascend uses its own forward_impl path, not the FlashInfer Cutlass
        chunked path. Always return False to stay on forward_impl."""
        return False

    @property
    def _fused_output_is_reduced(self) -> bool:
        # For MC2/ALLTOALL/FUSED_MC2 comm types, finalize() already includes
        # TP all-reduce for the routed output, and AscendSharedExperts.forward
        # handles it for the shared output. Signal this to the upstream
        # MoERunner.forward() so _maybe_reduce_final_output does not apply a
        # second TP all-reduce (which would double-count the contributions).
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        return moe_comm_type in {
            MoECommType.ALLTOALL,
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
        } or (moe_comm_type == MoECommType.ALLGATHER and self.moe_config.is_sequence_parallel)

    def _get_shared_expert_parallel_mode(self) -> SharedExpertParallelMode:
        shared_experts = getattr(self, "ascend_shared_experts", None)
        if shared_experts is None or not hasattr(shared_experts, "parallel_mode"):
            return SharedExpertParallelMode.TENSOR_PARALLEL
        return shared_experts.parallel_mode()

    def _reduce_shared_output_if_needed(
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool,
    ) -> torch.Tensor | None:
        if (
            shared_output is not None
            and fused_output_is_reduced
            and self._get_shared_expert_parallel_mode() is SharedExpertParallelMode.TENSOR_PARALLEL
        ):
            shared_output = tensor_model_parallel_all_reduce(shared_output)
        return shared_output

    @property
    def local_num_experts(self) -> int:
        """Number of physical experts managed by this EPLB layer."""
        return self.moe_config.num_local_experts

    @property
    def ep_rank(self) -> int:
        return self.moe_config.ep_rank

    # Shared-expert layout-specific communication is handled by
    # AscendSharedExperts, so only standard TP weights need a separate
    # all-reduce when routed output has already been reduced.
    def _maybe_reduce_shared_expert_output(  # type: ignore[misc]
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool | None = None,
    ) -> torch.Tensor | None:
        if fused_output_is_reduced is None:
            fused_output_is_reduced = self._fused_output_is_reduced
        return self._reduce_shared_output_if_needed(
            shared_output,
            fused_output_is_reduced,
        )

    def _maybe_reduce_routed_output_before_transform(
        self,
        fused_output: torch.Tensor,
        fused_output_is_reduced: bool,
    ) -> tuple[torch.Tensor, bool]:
        fused_output, fused_output_is_reduced = super()._maybe_reduce_routed_output_before_transform(
            fused_output,
            fused_output_is_reduced,
        )

        if (
            self._get_shared_expert_parallel_mode() is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY
            and not fused_output_is_reduced
        ):
            fused_output = tensor_model_parallel_all_reduce(fused_output)
            fused_output_is_reduced = True
        return fused_output, fused_output_is_reduced

    # Ascend already handles reduction in its own dispatch path, so
    # the upstream kwarg is accepted for interface alignment only.
    def _maybe_reduce_final_output(  # type: ignore[misc]
        self,
        states: torch.Tensor,
        trunc_size: int | None,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        if output_is_reduced is None:
            output_is_reduced = self._fused_output_is_reduced
        if not output_is_reduced and not self.moe_config.is_sequence_parallel:
            # Use the normal TP collective when the upstream reduction
            # contract requires it. Sequence-parallel outputs are token
            # shards, so reducing them position-wise would corrupt the result.
            states = tensor_model_parallel_all_reduce(states)
        if trunc_size is not None and trunc_size > 0:
            return states[..., :trunc_size]
        return states

    def set_lora_context(self, lora_context):
        self.routed_experts._ascend_moe_lora_context = lora_context
        if self.ascend_shared_experts is not None:
            self.ascend_shared_experts.set_lora_context(lora_context)

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with self._sequence_parallel_context():
            shared_hidden_states = shared_experts_input if shared_experts_input is not None else hidden_states
            if self.ascend_shared_experts is None:
                if self.is_internal_router:
                    gate = self.gate
                    assert gate is not None
                    hidden_states_fp32 = hidden_states.float()
                    router_logits = F.linear(
                        hidden_states_fp32,
                        gate.weight_fp32 if hasattr(gate, "weight_fp32") else gate.weight.to(torch.float32),
                    )
                return self.routed_experts.forward_impl(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                )
            shared_expert_input, shared_input_all_gather_done = (
                self.ascend_shared_experts.prepare_input_before_routed_experts(shared_hidden_states)
            )
            if self.is_internal_router:
                gate = self.gate
                assert gate is not None
                # NOTE(Angazenn): To make this cast explicitly, the hbm usage might
                # increase with extra hidden states. We also assume that all gate
                # linear is unquantized so that we the weight is pre-casted in
                # process_weights_after_loading of AscendUnquantizedLinearMethod.
                hidden_states_fp32 = shared_hidden_states.float()
                before_routed_experts = torch.npu.current_stream().record_event()
                # main (cdc4824a21): is_internal_router only checks self.gate,
                # weight_fp32 may be absent, fall back to gate.weight.
                router_logits = F.linear(
                    hidden_states_fp32,
                    gate.weight_fp32 if hasattr(gate, "weight_fp32") else gate.weight.to(torch.float32),
                )
                after_routed_experts = torch.npu.current_stream().record_event()
            else:
                before_routed_experts = torch.npu.current_stream().record_event()
                after_routed_experts = None

            if shared_input_all_gather_done is not None:
                torch.npu.current_stream().wait_event(shared_input_all_gather_done)
            routed_out, fused_moe_events = self.routed_experts.forward_impl(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
            fused_moe_events.before_routed_experts = before_routed_experts
            fused_moe_events.after_routed_experts = after_routed_experts
            if shared_input_all_gather_done is not None:
                fused_moe_events.after_routed_finalize = torch.npu.current_stream().record_event()

            shared_out = self.ascend_shared_experts.forward(
                shared_expert_input,
                fused_moe_events,
                input_is_gathered=shared_input_all_gather_done is not None,
            )
            return shared_out, routed_out
