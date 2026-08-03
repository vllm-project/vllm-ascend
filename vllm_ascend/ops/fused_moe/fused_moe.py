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
from dataclasses import dataclass, field
from functools import wraps

import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group, tensor_model_parallel_all_reduce
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,  # noqa: F401
    MoERunner,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    npu_stream_switch,
    shared_expert_dp_enabled,
    shared_experts_calculation_stream,
)


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    after_routed_experts: torch.npu.Event | None = field(default=None)
    before_dispatch: torch.npu.Event | None = field(default=None)
    before_gmm2: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)
    swiglu_limit: float = 0.0
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0


def mock_false():
    return False


def mock_true():
    return True


class AscendMoERunner(MoERunner):  # type: ignore[no-redef]
    moe_counter = -1

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

        ascend_config = get_ascend_config()
        self._shared_experts = shared_experts

        self.multistream_overlap_shared_expert = (
            ascend_config.multistream_overlap_shared_expert and shared_experts is not None
        )

        setup_moe_comm_method(self.moe_config)
        if self.multistream_overlap_shared_expert:
            # Wrap the quant_method's process_weights_after_loading to validate that
            # splitting shared expert computation (gate_up projection + activation,
            # then down projection) yields identical results to integrated
            # computation after weight loading.
            original_process_weights = self._quant_method.process_weights_after_loading

            @wraps(original_process_weights)
            def wrapped_process_weights(*args, **kwargs):
                result = original_process_weights(*args, **kwargs)
                self._validate_shared_expert_consistency()
                return result

            self._quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

    def _validate_shared_expert_consistency(self):
        """Validate that split shared expert computation matches integrated computation."""
        test_input = (
            torch.rand(10, self.hidden_size, device="npu", dtype=self.moe_config.in_dtype) * 2 - 1
        )  # Random input for testing, scoped to [-1, 1]

        assert self._shared_experts is not None
        integrated_out = self._shared_experts(test_input)
        part1_out = self._shared_experts_part1(test_input)
        split_out = self._shared_experts_part2(test_input, part1_out)

        if not torch.allclose(integrated_out, split_out):
            diff = (integrated_out - split_out).abs()
            logger.error(
                "[fused_moe/layer] Shared expert split computation validation failed."
                " The split-path computation does not match the integrated-path result."
                " max_abs_diff=%s, integrated_sum=%s, integrated_norm=%s,"
                " split_sum=%s, split_norm=%s, hidden_size=%s, dtype=%s.",
                diff.max().item(),
                integrated_out.sum().item(),
                integrated_out.norm().item(),
                split_out.sum().item(),
                split_out.norm().item(),
                self.hidden_size,
                self.moe_config.in_dtype,
            )
            raise ValueError("FusedMoE shared experts split computation does not match the integrated computation.")
        logger.info_once(
            "[fused_moe/layer] Shared expert split computation validation passed."
            " Integrated and split-path results are consistent."
        )

    def _shared_experts_part1(self, hidden_states: torch.Tensor):
        shared_gate_up, _ = self._shared_experts.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def _shared_experts_part2(self, hidden_states: torch.Tensor, shared_gate_up: torch.Tensor):
        shared_act = self._shared_experts.act_fn(shared_gate_up)  # type: ignore
        shared_out, _ = self._shared_experts.down_proj(shared_act)  # type: ignore

        # Qwen3-Next specific gating mechanism
        assert self._shared_experts is not None
        if hasattr(self._shared_experts, "expert_gate") and self._shared_experts.expert_gate is not None:
            gate_out, _ = self._shared_experts.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

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
        # TP all-reduce for the routed output, and _forward_shared_experts
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

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # _forward_shared_experts already handles shared expert TP all-reduce
        # for MC2/ALLTOALL/FUSED_MC2. For AllGather the reduction is done
        # via _maybe_reduce_final_output on the combined (shared + routed)
        # output. Skip any additional reduction here.
        return shared_output

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

    def set_lora_context(self, lora_context):
        self.routed_experts._ascend_moe_lora_context = lora_context

    def _forward_shared_experts(self, hidden_states: torch.Tensor, fused_moe_evts: FusedMoEEvents):
        if self._shared_experts is None:
            return None

        def maybe_wait_event(evt: torch.npu.Event | None):
            if evt is not None:
                torch.npu.current_stream().wait_event(evt)

        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap_shared_expert):
            # Only used for int quantization
            has_quantized_shared = hasattr(self._shared_experts.gate_up_proj, "weight_scale") and hasattr(
                self._shared_experts.down_proj, "weight_scale"
            )
            if has_quantized_shared and self.quant_type in (QuantType.W8A8, QuantType.W4A8):
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.after_routed_experts)
                hidden_states = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self._shared_experts.gate_up_proj.weight,
                    self._shared_experts.gate_up_proj.weight_scale,
                    pertoken_scale=None,
                    bias=None,
                    output_dtype=torch.int32,
                )
                # Execute activation concurrently with gmm2.

                maybe_wait_event(fused_moe_evts.before_gmm2)
                quantized_x, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                    x=hidden_states,
                    weight_scale=self._shared_experts.gate_up_proj.weight_scale_fp32,
                    activation_scale=pertoken_scale,
                    bias=None,
                    quant_scale=None,
                    quant_offset=None,
                    group_index=None,
                    activate_left=True,
                    quant_mode=1,
                    swiglu_mode=1,
                    clamp_limit=fused_moe_evts.swiglu_limit,
                    glu_alpha=fused_moe_evts.swiglu_alpha,
                    glu_bias=fused_moe_evts.swiglu_beta,
                )
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self._shared_experts.down_proj.weight,
                    self._shared_experts.down_proj.weight_scale,
                    pertoken_scale=swiglu_out_scale,
                    bias=None,
                    output_dtype=original_dtype,
                )
            elif has_quantized_shared and self.quant_type == QuantType.W4A8MXFP:
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                    hidden_states, dst_type=torch.float8_e4m3fn
                )
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                hidden_states = self._shared_experts.gate_up_proj((quantized_x, pertoken_scale))[0]
                # Execute activation concurrently with gmm2.
                maybe_wait_event(fused_moe_evts.before_gmm2)
                quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                    hidden_states,
                    topk_weight=None,
                    group_index=None,
                    dst_type=torch.float8_e4m3fn,
                    quant_mode=2,
                    clamp_value=fused_moe_evts.swiglu_limit,
                )
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = self._shared_experts.down_proj((quantized_x, swiglu_out_scale))[0]
            else:
                # Ensure the shared experts wait for hidden_states to be ready.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                part1_out = self._shared_experts_part1(hidden_states)
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = self._shared_experts_part2(hidden_states, part1_out)

        # Make sure the default stream waits for the shared experts stream to
        # finish.
        if self.multistream_overlap_shared_expert:
            torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

        # NOTE: This is exactly the opposite of
        # `maybe_all_reduce_tensor_model_parallel`
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        if (
            moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
            and not shared_expert_dp_enabled()
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out

    def shared_forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):
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

        fused_moe_results = self.routed_experts.no_shared_forward_impl(
            hidden_states=hidden_states,
            router_logits=router_logits,
            return_with_event=True,
        )
        routed_out = fused_moe_results.routed_out

        if self._shared_experts is None:
            return routed_out

        shared_out = self._forward_shared_experts(
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

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with self._sequence_parallel_context():
            if self.shared_experts is None:
                return self.routed_experts.no_shared_forward_impl(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )
            else:
                return self.shared_forward_impl(hidden_states, router_logits)
