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
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,  # noqa: F401
    MoERunner,
)

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts, FusedMoEEvents
from vllm_ascend.utils import vllm_version_is


class AscendMoERunner(MoERunner):  # type: ignore[no-redef]
    def __init__(
        self,
        layer_name,
        moe_config,
        router,
        routed_experts,
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

        self.quant_type = self.routed_experts.quant_type

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

    @property
    def is_internal_router(self) -> bool:
        gate = self.gate
        return gate is not None and hasattr(gate, "weight_fp32")

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
        } or (moe_comm_type == MoECommType.ALLGATHER and _EXTRA_CTX.flash_comm_v1_enabled)

    @property
    def local_num_experts(self) -> int:
        """Number of physical experts managed by this EPLB layer."""
        return self.moe_config.num_local_experts

    @property
    def ep_rank(self) -> int:
        return self.moe_config.ep_rank

    # main2main compat: `fused_output_is_reduced` was added to upstream
    # _maybe_reduce_shared_expert_output() in vllm main after 0.26.0.
    # Ascend already reduces shared expert output in
    # _forward_shared_experts, so the parameter is accepted for
    # interface alignment only.
    # Remove the version gate once 0.26.0 support is dropped.
    if vllm_version_is("0.26.0"):

        def _maybe_reduce_shared_expert_output(
            self,
            shared_output: torch.Tensor | None,
        ) -> torch.Tensor | None:
            # _forward_shared_experts already handles shared expert TP
            # all-reduce for MC2/ALLTOALL/FUSED_MC2. For AllGather the
            # reduction is done via _maybe_reduce_final_output on the
            # combined (shared + routed) output.
            return shared_output
    else:

        def _maybe_reduce_shared_expert_output(  # type: ignore[misc]
            self,
            shared_output: torch.Tensor | None,
            fused_output_is_reduced: bool | None = None,
        ) -> torch.Tensor | None:
            # Ascend handles shared expert reduction in
            # _forward_shared_experts; the upstream kwarg is unused here.
            _ = fused_output_is_reduced
            return shared_output

    # main2main compat: `output_is_reduced` was added to upstream
    # _maybe_reduce_final_output() in vllm main after 0.26.0.
    # Ascend already handles reduction in its own dispatch path, so
    # the upstream kwarg is accepted for interface alignment only.
    # Remove the version gate once 0.26.0 support is dropped.
    if vllm_version_is("0.26.0"):

        def _maybe_reduce_final_output(
            self,
            states: torch.Tensor,
            trunc_size: int,
        ) -> torch.Tensor:
            # Sequence-parallel MoE returns a token-sharded result after EP
            # reduce-scatter.  Qwen3Moe gathers those token shards across TP in
            # the model, so a TP all-reduce here would sum different token
            # segments and corrupt the sequence.
            if not self.moe_config.is_sequence_parallel:
                states = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(states)
            return states[..., :trunc_size]
    else:

        def _maybe_reduce_final_output(  # type: ignore[misc]
            self,
            states: torch.Tensor,
            trunc_size: int | None,
            output_is_reduced: bool | None = None,
        ) -> torch.Tensor:
            _ = output_is_reduced
            states = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(states)
            if trunc_size is not None and trunc_size > 0:
                return states[..., :trunc_size]
            return states

    def set_lora_context(self, lora_context):
        self.routed_experts._ascend_moe_lora_context = lora_context

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with self._sequence_parallel_context():
            if self.ascend_shared_experts is None:
                return self.routed_experts.forward_impl(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )
            if self.is_internal_router:
                gate = self.gate
                assert gate is not None
                # NOTE(Angazenn): To make this cast explicitly, the hbm usage might
                # increase with extra hidden states. We also assume that all gate
                # linear is unquantized so that we the weight is pre-casted in
                # process_weights_after_loading of AscendUnquantizedLinearMethod.
                hidden_states_fp32 = hidden_states.float()
                before_routed_experts = torch.npu.current_stream().record_event()
                router_logits = F.linear(hidden_states_fp32, gate.weight_fp32)
                after_routed_experts = torch.npu.current_stream().record_event()
            else:
                before_routed_experts = torch.npu.current_stream().record_event()
                after_routed_experts = None

            fused_moe_results = self.routed_experts.forward_impl(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            routed_out = fused_moe_results.routed_out

            shared_out = self.ascend_shared_experts.forward(
                hidden_states,
                FusedMoEEvents(
                    after_routed_experts=after_routed_experts,
                    before_routed_experts=before_routed_experts,
                    before_dispatch=fused_moe_results.before_dispatch_evt,
                    before_gmm2=fused_moe_results.before_gmm2_evt,
                    before_combine=fused_moe_results.before_combine_evt,
                    swiglu_limit=fused_moe_results.swiglu_limit,
                    swiglu_alpha=getattr(fused_moe_results, "swiglu_alpha", 1.0),
                    swiglu_beta=getattr(fused_moe_results, "swiglu_beta", 0.0),
                ),
            )
            return shared_out, routed_out
