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

from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoERouter
from vllm.model_executor.layers.fused_moe.layer import MoERunner
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    _layer_name_type,
    _moe_forward_shared,
    _resolve_layer_name,
    _unpack,
    get_layer_from_name,
)
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.compilation.kimi_moe_superkernel import moe_superkernel_scope
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import PreparedSharedExpertInput, RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.kimi_decode_schedule import ROUTER_CORE_LIMIT, KimiDecodeSchedule
from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method, setup_moe_comm_method
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.shared_experts import (
    AscendSharedExperts,
    PreparedSharedExpertExecution,
    SharedExpertParallelMode,
    SubmittedSharedExpertGateUp,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import npu_stream_switch


def _ascend_kimi_moe_fake(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    shared_experts_input: torch.Tensor | None = None,
    router_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    if shared_experts_input is not None:
        return shared_experts_input.new_empty((*hidden_states.shape[:-1], shared_experts_input.shape[-1]))
    return torch.empty_like(hidden_states)


def _ascend_kimi_moe(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    shared_experts_input: torch.Tensor | None = None,
    router_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    # Resolve once, like the upstream MoE op. Keep runtime phase selection and
    # all stream orchestration opaque to Dynamo, but capturable by ACLGraph.
    runner = get_layer_from_name(_resolve_layer_name(layer_name))
    enabled = (
        get_ascend_config().ascend_compilation_config.enable_super_kernel
        and not _EXTRA_CTX.is_draft_model
        and runner._can_run_kimi_decode(hidden_states if shared_experts_input is None else shared_experts_input)
    )
    runner._kimi_superkernel_active = enabled
    runner._kimi_topk = None
    runner._kimi_shared_output_done = None
    runner._kimi_shared_input_ready = None
    try:
        with moe_superkernel_scope(f"{runner.layer_name}_1.moe" if enabled else None, enabled):
            return runner._forward_standard(
                hidden_states,
                hidden_states if router_logits is None else router_logits,
                input_ids,
                shared_experts_input,
            )
    finally:
        # Do not retain a prefill-sized FP32 router tensor on every layer.
        runner._kimi_router_logits = None
        runner._kimi_router_done = None
        runner._kimi_superkernel_active = False
        runner._kimi_topk = None
        runner._kimi_shared_output_done = None
        runner._kimi_shared_input_ready = None


direct_register_custom_op(
    op_name="ascend_kimi_moe",
    op_func=_ascend_kimi_moe,
    fake_impl=_ascend_kimi_moe_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
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

        # Submit routed work before the complete shared forward for latent
        # W4A8 sequence-parallel shared experts, as in release/v0.27.1rc.
        self._release_shared_schedule = (
            self.quant_type is QuantType.W4A8
            and self._can_overlap_sp_shared_with(self.routed_input_transform)
            and self._can_overlap_sp_shared_with(self.routed_output_transform)
        )

        setup_moe_comm_method(self.moe_config)
        # Communication objects are shared across MoE layers. Keep the bound
        # activation with its owning layer when a later layer replaces them.
        self.routed_experts.mega_moe_activation_kwargs = getattr(
            get_moe_comm_method(MoECommType.FUSED_MC2, self.moe_config), "mega_moe_activation_kwargs", None
        )
        alltoall_comm = get_moe_comm_method(MoECommType.ALLTOALL, self.moe_config)
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
        if getattr(self, "_kimi_gate", None) is not None:
            return torch.ops.vllm.ascend_kimi_moe(
                hidden_states,
                input_ids,
                self._encode_layer_name(),
                shared_experts_input,
                None if router_logits is hidden_states else router_logits,
            )
        return self._forward_standard(hidden_states, router_logits, input_ids, shared_experts_input)

    def _forward_standard(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mirror of upstream ``MoERunner.forward`` (this class is instantiated
        through the OOT PluggableLayer dispatch of ``MoERunner``).

        Shared and routed experts use a plain tensor Add after the
        routed-output transform, independently of the shared scheduling policy.
        The runtime-aware reduction hooks (#16550) are unchanged.
        """
        # Apply transform for routed experts (e.g., latent projection for
        # latent MoE). When the caller pre-applies the routed input transform
        # outside the runner (e.g. to overlap it on a separate stream), it
        # passes the already-transformed routed input as ``hidden_states`` and
        # the original hidden states as ``shared_experts_input``; skip the
        # transform in that case so shared experts still see the original input.
        if shared_experts_input is None:
            if getattr(self, "_kimi_gate", None) is not None and router_logits is not hidden_states:
                hidden_states, shared_experts_input = self.apply_routed_input_transform(
                    hidden_states,
                    router_logits=router_logits,
                )
            else:
                hidden_states, shared_experts_input = self.apply_routed_input_transform(hidden_states)
        elif getattr(self, "_kimi_gate", None) is not None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                shared_experts_input,
                routed_input=hidden_states,
                router_logits=None if router_logits is hidden_states else router_logits,
            )

        # Record before `_maybe_pad_hidden_states` pads activations to match
        # `moe_config.hidden_dim`, e.g. after `align_trtllm_fp4_moe_hidden_dim_for_fi`
        # so routed output can be trimmed before
        # shared+routed add / latent up proj if needed.
        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )

        if getattr(self, "_kimi_gate", None) is not None:
            # Already inside ascend_kimi_moe. Resolving the upstream op again
            # would advance the static-kernel MoE layer cursor twice.
            result = self._forward_impl(hidden_states, router_logits, shared_experts_input, input_ids)
        else:
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

        # Shared join and final residual Add are outside named MoE fusion.
        with moe_superkernel_scope(None, getattr(self, "_kimi_superkernel_active", False)):
            if shared_output is not None:
                result = shared_output + fused_output
            else:
                result = fused_output

            result = self._maybe_reduce_final_output(result, og_hidden_dim_post_xform, fused_output_is_reduced)

            return self._maybe_add_zero_expert_output(result)

    @property
    def is_internal_router(self) -> bool:
        # main (cdc4824a21): vllm#51838 removed the gate branch in
        # DeepseekV2MoE.forward, always passing router_logits=hidden_states.
        # The runner must recompute router_logits via the gate.
        return getattr(self, "_kimi_gate", None) is not None or self.gate is not None

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
        *,
        routed_input: torch.Tensor | None = None,
        router_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Start the SP shared-input gather before the latent down projection."""
        self._kimi_shared_input_ready = None
        prepared_input = PreparedSharedExpertInput(hidden_states)
        shared_experts = self.ascend_shared_experts
        if self._can_overlap_sp_shared_with(self.routed_input_transform):
            assert shared_experts is not None
            with moe_superkernel_scope(None, getattr(self, "_kimi_superkernel_active", False)):
                prepared_input = shared_experts.prepare_input_async(
                    hidden_states, comm_stream=getattr(self, "_kimi_shared_comm_stream", None)
                )
        kimi_gate = getattr(self, "_kimi_gate", None)
        if kimi_gate is not None and router_logits is not None:
            self._kimi_decode_active = self._can_run_kimi_decode(hidden_states)
            self._kimi_router_logits = router_logits
            self._kimi_router_done = None
        elif kimi_gate is not None:
            self._kimi_decode_active = self._can_run_kimi_decode(hidden_states)
            stream = getattr(self, "_kimi_router_stream", None)
            ready = torch.npu.current_stream().record_event() if stream is not None else None
            if stream is not None:
                hidden_states.record_stream(stream)
            with npu_stream_switch(stream, enabled=stream is not None):
                if ready is not None:
                    torch.npu.current_stream().wait_event(ready)
                scope = (
                    torch_npu.npu.npugraph_ex.scope.limit_core_num(*ROUTER_CORE_LIMIT)
                    if self._kimi_decode_active
                    else nullcontext()
                )
                with scope:
                    self._kimi_router_logits = self._compute_router_logits(hidden_states, hidden_states)
                self._kimi_router_done = torch.npu.current_stream().record_event() if stream is not None else None
                if self._kimi_decode_active:
                    self._kimi_topk = self._prepare_kimi_decode_topk(hidden_states, self._kimi_router_logits)
            # Quantization overlaps Router GEMM; the projection waits only for
            # GEMM, not TopK. Preserve the registered linear's padding/scales.
            if self._kimi_decode_active and routed_input is None:
                scheme = self.routed_input_transform.quant_method.quant_method
                prepared = scheme.prepare_input_for_overlap(self.routed_input_transform, hidden_states)
                if self._kimi_router_done is not None:
                    torch.npu.current_stream().wait_event(self._kimi_router_done)
                routed_input = self.routed_input_transform(prepared if prepared is not None else hidden_states)[0]
            elif self._kimi_router_done is not None:
                torch.npu.current_stream().wait_event(self._kimi_router_done)
        if routed_input is None:
            routed_input, shared_input = super().apply_routed_input_transform(hidden_states)
        else:
            shared_input = hidden_states
        if prepared_input.ready_event is not None:
            if kimi_gate is not None and self._kimi_decode_active:
                # Decode Dispatch does not consume the gathered activations.
                # Hand readiness to the shared stream instead of blocking MC2.
                self._kimi_shared_input_ready = prepared_input.ready_event
            else:
                torch.npu.current_stream().wait_event(prepared_input.ready_event)
            shared_input = prepared_input.hidden_states
        return routed_input, shared_input

    def _prepare_kimi_decode_topk(self, hidden_states, logits):
        """Select on the Router stream without reading the latent activations.

        Only the exact non-custom router admitted by _can_run_kimi_decode is
        allowed here. Match MC2's SP padding, retaining its actual prepare for
        hidden states, masks, scales and finalize bookkeeping.
        """
        comm = get_moe_comm_method(MoECommType.MC2, self.moe_config)
        pad = max(0, _EXTRA_CTX.padded_num_tokens // comm.prepare_finalize.tp_size - hidden_states.shape[0])
        if pad:
            logits = F.pad(logits, (0, 0, 0, pad))
        metadata = hidden_states.new_empty((hidden_states.shape[0] + pad, self.moe_config.hidden_dim))
        weights, ids = self.routed_experts._select_experts(metadata, logits, _EXTRA_CTX.in_profile_run, None)
        return weights, ids, torch.npu.current_stream().record_event()

    def apply_routed_output_transform(self, fused_output: torch.Tensor) -> torch.Tensor:
        """Run the latent up projection before joining the SP shared output."""
        fused_output = super().apply_routed_output_transform(fused_output)
        shared_experts = self.ascend_shared_experts
        if self._can_overlap_sp_shared_with(self.routed_output_transform):
            assert shared_experts is not None
            with moe_superkernel_scope(None, getattr(self, "_kimi_superkernel_active", False)):
                output_done = getattr(self, "_kimi_shared_output_done", None)
                if output_done is not None:
                    torch.npu.current_stream().wait_event(output_done)
                else:
                    shared_experts.wait_for_output()
        return fused_output

    def _prepare_router_input(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare only the input needed by a pre-cast FP32 router gate."""
        if getattr(self, "_kimi_gate", None) is not None:
            return hidden_states
        gate = getattr(self, "_kimi_gate", None) or self.gate
        assert gate is not None
        if not hasattr(gate, "weight_fp32"):
            return hidden_states
        return router_logits if router_logits.dtype == torch.float32 else hidden_states.float()

    def _apply_router_linear(self, router_input: torch.Tensor) -> torch.Tensor:
        """Apply the internal router gate to an already prepared input."""
        kimi_gate = getattr(self, "_kimi_gate", None)
        if kimi_gate is not None:
            # Preserve the model's registered 0.30 gate, including its output
            # dtype and kernel selection; only its stream placement changes.
            result = kimi_gate(router_input)
            return result[0] if isinstance(result, tuple) else result
        gate = getattr(self, "_kimi_gate", None) or self.gate
        assert gate is not None
        # Models with pre-cast FP32 weights use the direct linear path; other
        # gates retain their registered forward and its dtype semantics.
        if hasattr(gate, "weight_fp32"):
            return F.linear(router_input, gate.weight_fp32)
        gate_out = gate(router_input)
        return gate_out[0] if isinstance(gate_out, tuple) else gate_out

    def _compute_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute router logits without extending a fresh FP32 cast's lifetime.

        Gates without a pre-cast weight retain their registered forward path
        and do not allocate an unused FP32 input.
        """
        router_input = self._prepare_router_input(hidden_states, router_logits)
        return self._apply_router_linear(router_input)

    def _prepare_shared_for_router_overlap(
        self,
        prepared_shared_input: PreparedSharedExpertInput,
        shared_input_ready: torch.npu.Event,
    ) -> PreparedSharedExpertExecution | None:
        shared_experts = self.ascend_shared_experts
        assert shared_experts is not None
        if not getattr(shared_experts, "multistream_overlap", False):
            return None
        return shared_experts.prepare_for_router_overlap(
            prepared_shared_input,
            shared_input_ready,
        )

    def _prepare_router_and_milestones(
        self,
        prepared_shared_input: PreparedSharedExpertInput,
        shared_hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.npu.Event,
        torch.npu.Event,
        SubmittedSharedExpertGateUp | None,
    ]:
        if getattr(self, "_release_shared_schedule", False):
            # The shared stream observes these events when its complete
            # forward is submitted after routed_experts.forward_impl.
            if self.is_internal_router:
                router_input = self._prepare_router_input(shared_hidden_states, router_logits)
                shared_input_ready = torch.npu.current_stream().record_event()
                router_logits = self._apply_router_linear(router_input)
                del router_input
                router_output_ready = torch.npu.current_stream().record_event()
            else:
                shared_input_ready = torch.npu.current_stream().record_event()
                router_output_ready = shared_input_ready
            return router_logits, shared_input_ready, router_output_ready, None

        if self.is_internal_router:
            router_input = self._prepare_router_input(shared_hidden_states, router_logits)
            # Release shared Vector/AIV preparation after any external FP32
            # router-input Cast. Gates using their registered forward path may
            # still prepare their input internally.
            shared_input_ready = torch.npu.current_stream().record_event()
            prepared_execution = self._prepare_shared_for_router_overlap(
                prepared_shared_input,
                shared_input_ready,
            )
            router_logits = self._apply_router_linear(router_input)
            del router_input
            router_output_ready = torch.npu.current_stream().record_event()
        else:
            shared_input_ready = torch.npu.current_stream().record_event()
            prepared_execution = self._prepare_shared_for_router_overlap(
                prepared_shared_input,
                shared_input_ready,
            )
            router_output_ready = shared_input_ready
        # Submit Gate-Up now, before routed_experts.forward_impl enqueues TopK
        # and dispatch. The auxiliary stream waits for router_output_ready, so
        # Gate-Up starts together with routed TopK and can cover dispatch.
        shared_experts = self.ascend_shared_experts
        assert shared_experts is not None
        submitted_gate_up = (
            shared_experts.enqueue_gate_up_after_router(
                prepared_execution,
                router_output_ready,
            )
            if prepared_execution is not None
            else None
        )
        return router_logits, shared_input_ready, router_output_ready, submitted_gate_up

    def _can_run_kimi_decode(self, hidden_states: torch.Tensor) -> bool:
        # Quant schemes import routed/shared MoE code; load these after module
        # initialization. Both runners use the same MC2/GMM callbacks; keep
        # the Decode, quantization and shared-expert requirements identical.
        from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
        from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
            AscendW4A8MXFPDynamicFusedMoEMethod,
            AscendW4A8MXFPDynamicLinearMethod,
        )
        from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod

        metadata = get_forward_context().attn_metadata
        metadata = list(metadata.values()) if isinstance(metadata, dict) else [metadata]
        shared = self.ascend_shared_experts
        if shared is None:
            return False
        linears = (shared.layer.gate_up_proj, shared.layer.down_proj)
        bf16 = all(linear.weight.dtype == torch.bfloat16 and not hasattr(linear, "weight_scale") for linear in linears)
        mxfp4 = all(
            type(getattr(getattr(linear, "quant_method", None), "quant_method", None))
            is AscendW4A8MXFPDynamicLinearMethod
            for linear in linears
        )
        routed_scheme = getattr(self.routed_experts.quant_method, "quant_method", None)
        down_scheme = getattr(getattr(self.routed_input_transform, "quant_method", None), "quant_method", None)
        router = self.routed_experts.router
        return (
            get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMOE_MXFP)
            and bool(metadata)
            and all(getattr(m, "num_prefills", None) == 0 for m in metadata)
            and _EXTRA_CTX.moe_comm_type is MoECommType.MC2
            and self.quant_type is QuantType.W4A8MXFP
            and type(routed_scheme) is AscendW4A8MXFPDynamicFusedMoEMethod
            and not routed_scheme.use_expert_weight_list
            and type(down_scheme) is AscendW8A8MXFP8DynamicLinearMethod
            and type(router) is AscendFusedTopKRouter
            and router.custom_routing_function is None
            and router.tid2eid is None
            and router.bias_vl is None
            and self._can_overlap_sp_shared_with(self.routed_input_transform)
            and self._can_overlap_sp_shared_with(self.routed_output_transform)
            and self.moe_config.is_sequence_parallel
            and hidden_states.dtype == torch.bfloat16
            and not self.routed_experts.dynamic_eplb
            and not _EXTRA_CTX.is_draft_model
            and getattr(self.routed_experts, "_ascend_moe_lora_context", None) is None
            and shared.situ_activation is not None
            and getattr(shared.layer, "expert_gate", None) is None
            and (bf16 or mxfp4)
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        kimi_gate = getattr(self, "_kimi_gate", None)
        if kimi_gate is not None:
            if self._kimi_router_done is not None:
                torch.npu.current_stream().wait_event(self._kimi_router_done)
            router_logits = self._kimi_router_logits
            if getattr(self, "_kimi_router_stream", None) is not None:
                router_logits.record_stream(torch.npu.current_stream())
        with self._sequence_parallel_context():
            shared_hidden_states = shared_experts_input if shared_experts_input is not None else hidden_states
            if self.ascend_shared_experts is None:
                if self.is_internal_router and kimi_gate is None:
                    router_logits = self._compute_router_logits(hidden_states, router_logits)
                return self.routed_experts.forward_impl(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                    router_stream=getattr(self, "_kimi_router_stream", None),
                )
            shared_input_is_gathered = self._can_overlap_sp_shared_with(self.routed_input_transform)
            defer_shared_output_wait = self._can_overlap_sp_shared_with(self.routed_output_transform)
            prepared_shared_input = (
                PreparedSharedExpertInput(
                    shared_hidden_states,
                    is_gathered=True,
                    ready_event=getattr(self, "_kimi_shared_input_ready", None),
                )
                if shared_input_is_gathered
                else self.ascend_shared_experts.prepare_input_before_routed(shared_hidden_states)
            )
            if kimi_gate is not None:
                ready = (
                    torch.npu.current_stream().record_event()
                    if self.ascend_shared_experts.multistream_overlap
                    else None
                )
                if self._kimi_decode_active:
                    schedule = KimiDecodeSchedule(
                        self.ascend_shared_experts,
                        prepared_shared_input,
                        ready,
                        enable_superkernel=getattr(self, "_kimi_superkernel_active", False),
                        comm_stream=getattr(self, "_kimi_shared_comm_stream", None),
                    )
                    routed_out, _ = self.routed_experts.forward_impl(
                        hidden_states=hidden_states,
                        router_logits=router_logits,
                        input_ids=input_ids,
                        shared_expert_schedule=schedule,
                        router_stream=getattr(self, "_kimi_router_stream", None),
                        precomputed_topk=getattr(self, "_kimi_topk", None),
                    )
                    shared_out = schedule.finalize()
                    self._kimi_shared_output_done = schedule.output_done
                    return shared_out, routed_out
                shared_input_ready = router_output_ready = ready
                prepared_execution = self._prepare_shared_for_router_overlap(prepared_shared_input, ready)
                submitted_gate_up = (
                    self.ascend_shared_experts.enqueue_gate_up_after_router(prepared_execution, ready)
                    if prepared_execution is not None
                    else None
                )
            else:
                router_logits, shared_input_ready, router_output_ready, submitted_gate_up = (
                    self._prepare_router_and_milestones(
                        prepared_shared_input,
                        shared_hidden_states,
                        router_logits,
                    )
                )
            routed_out, milestones = self.routed_experts.forward_impl(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
                router_stream=getattr(self, "_kimi_router_stream", None),
            )
            assert isinstance(milestones, RoutedMoEMilestones)
            milestones.shared_input_ready = shared_input_ready
            milestones.router_output_ready = router_output_ready
            if prepared_shared_input.is_gathered:
                milestones.routed_finalize_done = torch.npu.current_stream().record_event()

            if submitted_gate_up is None:
                shared_out = self.ascend_shared_experts.forward(
                    prepared_shared_input,
                    milestones,
                    defer_output_wait=defer_shared_output_wait,
                )
            else:
                shared_out = self.ascend_shared_experts.forward(
                    prepared_shared_input,
                    milestones,
                    defer_output_wait=defer_shared_output_wait,
                    submitted_gate_up=submitted_gate_up,
                )
            return shared_out, routed_out
