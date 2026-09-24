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
from __future__ import annotations

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
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _moe_forward_shared, _unpack
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import PreparedSharedExpertInput, RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method, setup_moe_comm_method
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.shared_experts import (
    AscendSharedExperts,
    SharedExpertParallelMode,
)


def _ascend_moe_forward_shared_sp_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: object,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer the local SP outputs of the shared-expert custom op.

    The explicit shared input contains all gathered tokens, while the real
    shared output has already been reduce-scattered back to the routed input's
    local token count. The upstream fake uses the shared input's token count,
    which is only correct before the early all-gather optimization.
    """
    del router_logits, input_ids, layer_name
    assert shared_experts_input is not None
    shared_out = shared_experts_input.new_empty(
        (*hidden_states.shape[:-1], shared_experts_input.shape[-1]),
    )
    if hidden_dim_unpadded > 0:
        fused_out = hidden_states.new_empty(
            (*hidden_states.shape[:-1], hidden_dim_unpadded),
        )
    else:
        fused_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


# Keep the gathered shared input as an explicit graph dependency, but expose
# the SP-local shared output shape to torch.compile/ACL graph tracing.
direct_register_custom_op(
    op_name="ascend_moe_forward_shared_sp",
    op_func=_moe_forward_shared,
    fake_impl=_ascend_moe_forward_shared_sp_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
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
            if self._can_overlap_sp_shared_with(self.routed_input_transform):
                self._forward_entry = torch.ops.vllm.ascend_moe_forward_shared_sp

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mirror of upstream ``MoERunner.forward`` (this class is instantiated
        through the OOT PluggableLayer dispatch of ``MoERunner``).

        Upstream inlines the shared+routed output combine, which is not
        exposed as an overridable hook. On NPU the combine goes through the
        fused multi-tensor add kernel (``torch._foreach_add``), whose kernel
        launch count is independent of the tensor-list length and which is
        the NPU-validated fast path for the MoE combine. Every other step,
        including the runtime-aware reduction hooks (#16550), is inherited
        unchanged.

        Future plan: once the aclnnAdd_AddAiCore_Add kernel is available via
        the superkernel integration, upstream's plain add becomes the fast
        path and this override can be dropped.
        """
        # Apply transform for routed experts (e.g., latent projection for
        # latent MoE). When the caller pre-applies the routed input transform
        # outside the runner (e.g. to overlap it on a separate stream), it
        # passes the already-transformed routed input as ``hidden_states`` and
        # the original hidden states as ``shared_experts_input``; skip the
        # transform in that case so shared experts still see the original input.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(hidden_states)

        # Record before `_maybe_pad_hidden_states` pads activations to match
        # `moe_config.hidden_dim`, e.g. after `align_trtllm_fp4_moe_hidden_dim_for_fi`
        # so routed output can be trimmed before
        # shared+routed add / latent up proj if needed.
        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded if self._quant_method.has_unpadded_output else 0,
        )

        #
        # Note: there are two all-reduce points below. They are mutually
        # exclusive, controlled by _fused_output_is_reduced
        #  - When True: the combine kernel already reduced fused_output,
        #    so we reduce shared_output here to match, then skip the
        #    all-reduce in _maybe_reduce_final_output.
        #  - When False: neither output is reduced yet, so we combine
        #    them first and all-reduce the sum in _maybe_reduce_final_output.

        # Extract outputs from result
        shared_output, fused_output = _unpack(result)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        fused_output_is_reduced = self._fused_output_is_reduced

        # Latent routed output has to be reduced before output transform,
        # because the transform may include non-linear normalization.
        fused_output, fused_output_is_reduced = self._maybe_reduce_routed_output_before_transform(
            fused_output,
            fused_output_is_reduced,
        )

        # If routed output is already reduced, reduce shared to match.
        # See note above re: the two all-reduce points.
        shared_output = self._maybe_reduce_shared_expert_output(shared_output, fused_output_is_reduced)

        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(shared_output, fused_output)

        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

        if shared_output is not None:
            result = torch._foreach_add([shared_output], [fused_output])[0]
        else:
            result = fused_output

        result = self._maybe_reduce_final_output(result, og_hidden_dim_post_xform, fused_output_is_reduced)

        return self._maybe_add_zero_expert_output(result)

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

    def _can_overlap_sp_shared_with(self, routed_transform: object | None) -> bool:
        """Limit SP shared-expert overlap changes to routed transforms.

        Kimi-K3 is currently the only Ascend model that supplies these latent
        transforms. Keep the guard capability-based so every model without
        them stays on the original single-stream synchronization path.
        """
        shared_experts = getattr(self, "ascend_shared_experts", None)
        return (
            routed_transform is not None
            and shared_experts is not None
            and shared_experts.multistream_overlap
            and shared_experts.parallel_mode() is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
        )

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

    @property
    def local_num_experts(self) -> int:
        """Number of physical experts managed by this EPLB layer."""
        return self.moe_config.num_local_experts

    @property
    def ep_rank(self) -> int:
        return self.moe_config.ep_rank

    def _should_reduce_routed_before_combine(self) -> bool:
        """Static layer policy; communication-dependent decisions stay in the op."""
        # Shared DP already produces a complete output, so reduce routed first.
        if self._get_shared_expert_parallel_mode() is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
            return True
        # A routed transform must receive the complete TP result.
        if self.routed_output_transform is None or self.moe_config.is_sequence_parallel:
            return False
        return self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1

    # Shared-expert layout-specific communication is handled by
    # AscendSharedExperts, so only standard TP weights need a separate
    # all-reduce when routed output has already been reduced.
    def _maybe_reduce_shared_expert_output(  # type: ignore[misc]
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool | None = None,
    ) -> torch.Tensor | None:
        if (
            shared_output is None
            or self._get_shared_expert_parallel_mode() is not SharedExpertParallelMode.TENSOR_PARALLEL
        ):
            return shared_output
        # The upstream boolean can be specialized during tracing. Only the
        # model-static early-reduction policy may be tested outside the op.
        if self._should_reduce_routed_before_combine():
            return tensor_model_parallel_all_reduce(shared_output)
        return torch.ops.vllm.maybe_all_reduce_shared_expert(shared_output, self.layer_name)

    def _maybe_reduce_routed_output_before_transform(
        self,
        fused_output: torch.Tensor,
        fused_output_is_reduced: bool,
    ) -> tuple[torch.Tensor, bool]:
        if self._should_reduce_routed_before_combine():
            fused_output = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(fused_output, self.layer_name)
            return fused_output, True
        return fused_output, fused_output_is_reduced

    def _maybe_reduce_final_output(  # type: ignore[misc]
        self,
        states: torch.Tensor,
        trunc_size: int | None,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        # Do not branch on output_is_reduced: it can describe the tracing
        # batch rather than the batch replaying this graph. Early reduction
        # and sequence parallelism are static properties of the layer.
        if not self.moe_config.is_sequence_parallel and not self._should_reduce_routed_before_combine():
            states = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(states, self.layer_name)
        if trunc_size is not None and trunc_size > 0:
            return states[..., :trunc_size]
        return states

    def set_lora_context(self, lora_context):
        self.routed_experts._ascend_moe_lora_context = lora_context
        if self.ascend_shared_experts is not None:
            self.ascend_shared_experts.set_lora_context(lora_context)

    def apply_routed_input_transform(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Start the SP shared-input gather before the latent down projection."""
        prepared_input = PreparedSharedExpertInput(hidden_states)
        shared_experts = self.ascend_shared_experts
        if self._can_overlap_sp_shared_with(self.routed_input_transform):
            assert shared_experts is not None
            prepared_input = shared_experts.prepare_input_async(hidden_states)
        routed_input, shared_input = super().apply_routed_input_transform(hidden_states)
        if prepared_input.ready_event is not None:
            torch.npu.current_stream().wait_event(prepared_input.ready_event)
            shared_input = prepared_input.hidden_states
        return routed_input, shared_input

    def apply_routed_output_transform(self, fused_output: torch.Tensor) -> torch.Tensor:
        """Run the latent up projection before joining the SP shared output."""
        fused_output = super().apply_routed_output_transform(fused_output)
        shared_experts = self.ascend_shared_experts
        if self._can_overlap_sp_shared_with(self.routed_output_transform):
            assert shared_experts is not None
            shared_experts.wait_for_output()
        return fused_output

    def _compute_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.gate
        assert gate is not None
        # Models such as DeepSeek V4 opt into FP32 routing at weight loading.
        # Other gates must retain their own dtype and forward implementation.
        if hasattr(gate, "weight_fp32"):
            hidden_states_fp32 = router_logits if router_logits.dtype == torch.float32 else hidden_states.float()
            return F.linear(hidden_states_fp32, gate.weight_fp32)
        gate_out = gate(hidden_states)
        return gate_out[0] if isinstance(gate_out, tuple) else gate_out

    def _prepare_router_and_milestones(
        self,
        shared_hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.npu.Event, torch.npu.Event]:
        if self.is_internal_router:
            shared_input_ready = torch.npu.current_stream().record_event()
            router_logits = self._compute_router_logits(shared_hidden_states, router_logits)
            router_output_ready = torch.npu.current_stream().record_event()
        else:
            shared_input_ready = torch.npu.current_stream().record_event()
            router_output_ready = shared_input_ready
        return router_logits, shared_input_ready, router_output_ready

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
                    router_logits = self._compute_router_logits(hidden_states, router_logits)
                return self.routed_experts.forward_impl(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                )
            shared_input_is_gathered = self._can_overlap_sp_shared_with(self.routed_input_transform)
            defer_shared_output_wait = self._can_overlap_sp_shared_with(self.routed_output_transform)
            prepared_shared_input = (
                PreparedSharedExpertInput(shared_hidden_states, is_gathered=True)
                if shared_input_is_gathered
                else self.ascend_shared_experts.prepare_input_before_routed(shared_hidden_states)
            )
            router_logits, shared_input_ready, router_output_ready = self._prepare_router_and_milestones(
                shared_hidden_states,
                router_logits,
            )
            routed_out, milestones = self.routed_experts.forward_impl(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
            assert isinstance(milestones, RoutedMoEMilestones)
            milestones.shared_input_ready = shared_input_ready
            milestones.router_output_ready = router_output_ready
            if prepared_shared_input.is_gathered:
                milestones.routed_finalize_done = torch.npu.current_stream().record_event()

            shared_out = self.ascend_shared_experts.forward(
                prepared_shared_input,
                milestones,
                defer_output_wait=defer_shared_output_wait,
            )
            return shared_out, routed_out
