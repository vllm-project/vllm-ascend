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
from enum import Enum, auto
from functools import wraps

import torch
import torch.nn.functional as F
import torch_npu
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoEMethodBase

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.lora.fused_moe import has_lora
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_sp_by_pass,
    get_ascend_device_type,
    npu_stream_switch,
    shared_experts_calculation_stream,
)


@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    after_routed_experts: torch.npu.Event | None = field(default=None)
    before_dispatch: torch.npu.Event | None = field(default=None)
    before_gmm2: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)


class SharedExpertParallelMode(Enum):
    """Effective activation and weight layout for a shared-expert forward."""

    FLASHCOMM_OFF_SHARED_EXPERT_DP_OFF = auto()  # Full activations, TP-sharded weights.
    FLASHCOMM_ON_SHARED_EXPERT_DP_OFF = auto()  # Sharded activations, TP-sharded weights.
    FLASHCOMM_OFF_SHARED_EXPERT_DP_ON = auto()  # Full activations, replicated weights.
    FLASHCOMM_ON_SHARED_EXPERT_DP_ON = auto()  # Sharded activations, replicated weights.


class AscendSharedExperts:
    """Ascend-owned shared expert executor.

    Keep the original shared expert module registered on ``AscendMoERunner``
    for checkpoint compatibility while moving split/overlap execution details
    out of the runner.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        quant_type: QuantType,
        quant_method: FusedMoEMethodBase,
    ):
        self.layer = layer
        self.moe_config = moe_config
        self.hidden_size = moe_config.hidden_dim
        self.in_dtype = moe_config.in_dtype
        self.swiglu_limit = 0.0 if moe_config.swiglu_limit is None else moe_config.swiglu_limit
        self.swiglu_alpha = 1.0 if moe_config.swiglu_alpha is None else moe_config.swiglu_alpha
        self.swiglu_beta = 0.0 if moe_config.swiglu_beta is None else moe_config.swiglu_beta
        self.quant_type = quant_type
        self.lora_context = None
        ascend_config = get_ascend_config()
        self.multistream_overlap = ascend_config.multistream_overlap_shared_expert
        # Native MoE sequence parallelism and the SP compiler pass also build
        # shared-expert linears with replicated weights. Keep this weight-layout
        # state independent from the runtime FlashComm decision.
        self.weights_replicated = (
            ascend_config.enable_shared_expert_dp or moe_config.is_sequence_parallel or enable_sp_by_pass()
        )

        if self.multistream_overlap:
            # Wrap the quant_method's process_weights_after_loading to validate that
            # splitting shared expert computation (gate_up projection + activation,
            # then down projection) yields identical results to integrated
            # computation after weight loading.
            original_process_weights = quant_method.process_weights_after_loading

            @wraps(original_process_weights)
            def wrapped_process_weights(*args, **kwargs):
                result = original_process_weights(*args, **kwargs)
                self.validate_consistency()
                return result

            quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context

    def validate_consistency(self):
        """Validate that split shared expert computation matches integrated computation."""
        test_input = (
            torch.rand(10, self.hidden_size, device="npu", dtype=self.in_dtype) * 2 - 1
        )  # Random input for testing, scoped to [-1, 1]

        integrated_out = self.layer(test_input)
        part1_out = self.part1(test_input)
        split_out = self.part2(test_input, part1_out)

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
                self.in_dtype,
            )
            raise ValueError("FusedMoE shared experts split computation does not match the integrated computation.")
        logger.info_once(
            "[fused_moe/layer] Shared expert split computation validation passed."
            " Integrated and split-path results are consistent."
        )

    def part1(self, hidden_states: torch.Tensor):
        shared_gate_up, _ = self.layer.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def part2(self, hidden_states: torch.Tensor, shared_gate_up: torch.Tensor):
        shared_act = self.layer.act_fn(shared_gate_up)  # type: ignore
        shared_out, _ = self.layer.down_proj(shared_act)  # type: ignore

        # Qwen3-Next specific gating mechanism
        if hasattr(self.layer, "expert_gate") and self.layer.expert_gate is not None:
            gate_out, _ = self.layer.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

    def parallel_mode(self) -> SharedExpertParallelMode:
        """Resolve the effective activation/weight layout for this forward."""
        # EP rewrites FusedMoEParallelConfig.tp_size to 1 because each routed
        # expert is local. Shared-expert linears still span the physical TP
        # group, so their layout must be derived from that group instead.
        tp_size = self.moe_config.tp_group.world_size
        if tp_size <= 1:
            return SharedExpertParallelMode.FLASHCOMM_OFF_SHARED_EXPERT_DP_OFF

        activations_sharded = (
            _EXTRA_CTX.flash_comm_v1_enabled or self.moe_config.is_sequence_parallel or enable_sp_by_pass()
        )
        if activations_sharded and self.weights_replicated:
            return SharedExpertParallelMode.FLASHCOMM_ON_SHARED_EXPERT_DP_ON
        if activations_sharded:
            return SharedExpertParallelMode.FLASHCOMM_ON_SHARED_EXPERT_DP_OFF
        if self.weights_replicated:
            return SharedExpertParallelMode.FLASHCOMM_OFF_SHARED_EXPERT_DP_ON
        return SharedExpertParallelMode.FLASHCOMM_OFF_SHARED_EXPERT_DP_OFF

    def _prepare_local_dp_input(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        original_num_tokens = hidden_states.shape[0]
        # See parallel_mode(): moe_config.tp_size describes routed experts in
        # EP, while this token split follows the shared-expert TP group.
        tp_group = self.moe_config.tp_group
        tp_size = tp_group.world_size
        pad_size = (tp_size - original_num_tokens % tp_size) % tp_size
        if pad_size > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
        hidden_states = torch.tensor_split(
            hidden_states,
            tp_size,
            dim=0,
        )[tp_group.rank_in_group]
        return hidden_states, (original_num_tokens, pad_size)

    def _finalize_local_dp_output(
        self,
        shared_out: torch.Tensor,
        metadata: tuple[int, int],
    ) -> torch.Tensor:
        original_num_tokens, pad_size = metadata
        shared_out = self.moe_config.tp_group.all_gather(shared_out, dim=0)
        if pad_size > 0:
            shared_out = shared_out[:original_num_tokens]
        return shared_out

    @staticmethod
    def _prepare_flashcomm_tp_input(hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, True)

    def _finalize_flashcomm_tp_output(self, shared_out: torch.Tensor) -> torch.Tensor:
        # Multimodal draft-model inputs intentionally bypass the usual
        # maybe_pad_and_reduce path. This output is different: its input was
        # explicitly TP all-gathered above for TP-sharded weights, so reduce it
        # directly with the physical TP group.
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            shared_out = F.pad(shared_out, (0, 0, 0, pad_size))
        return self.moe_config.tp_group.reduce_scatter(shared_out, dim=0)

    def forward(self, hidden_states: torch.Tensor, fused_moe_evts: FusedMoEEvents):
        mode = self.parallel_mode()
        local_dp_metadata = None

        def maybe_wait_event(evt: torch.npu.Event | None):
            if evt is not None:
                torch.npu.current_stream().wait_event(evt)

        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap):
            if mode is SharedExpertParallelMode.FLASHCOMM_OFF_SHARED_EXPERT_DP_ON:
                # Full activations + replicated weights: shard tokens locally,
                # run the MLP, then gather its complete output.
                maybe_wait_event(fused_moe_evts.before_routed_experts)
                hidden_states, local_dp_metadata = self._prepare_local_dp_input(hidden_states)
            elif mode is SharedExpertParallelMode.FLASHCOMM_ON_SHARED_EXPERT_DP_OFF:
                # TP-sharded weights need the full activation slice. Gather the
                # FlashComm shard before the MLP and reduce-scatter afterwards.
                maybe_wait_event(fused_moe_evts.before_routed_experts)
                hidden_states = self._prepare_flashcomm_tp_input(hidden_states)
            elif mode not in {
                SharedExpertParallelMode.FLASHCOMM_ON_SHARED_EXPERT_DP_ON,
                SharedExpertParallelMode.FLASHCOMM_OFF_SHARED_EXPERT_DP_OFF,
            }:
                raise AssertionError(f"Unsupported shared expert mode: {mode}")

            # Only used for int quantization
            has_quantized_shared_without_lora = (
                not has_lora(self.lora_context)
                and hasattr(self.layer.gate_up_proj, "weight_scale")
                and hasattr(self.layer.down_proj, "weight_scale")
            )
            if has_quantized_shared_without_lora and self.quant_type in (QuantType.W8A8, QuantType.W4A8):
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.after_routed_experts)
                hidden_states = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self.layer.gate_up_proj.weight,
                    self.layer.gate_up_proj.weight_scale,
                    pertoken_scale=None,
                    bias=None,
                    output_dtype=torch.int32,
                )
                # Execute activation concurrently with gmm2.

                maybe_wait_event(fused_moe_evts.before_gmm2)
                quantized_x, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                    x=hidden_states,
                    weight_scale=self.layer.gate_up_proj.weight_scale_fp32,
                    activation_scale=pertoken_scale,
                    bias=None,
                    quant_scale=None,
                    quant_offset=None,
                    group_index=None,
                    activate_left=True,
                    quant_mode=1,
                    swiglu_mode=1,
                    clamp_limit=self.swiglu_limit,
                    **(
                        {}
                        if get_ascend_device_type() == AscendDeviceType.A5
                        else {"glu_alpha": self.swiglu_alpha, "glu_bias": self.swiglu_beta}
                    ),
                )
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self.layer.down_proj.weight,
                    self.layer.down_proj.weight_scale,
                    pertoken_scale=swiglu_out_scale,
                    bias=None,
                    output_dtype=original_dtype,
                )
            elif has_quantized_shared_without_lora and self.quant_type == QuantType.W4A8MXFP:
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                    hidden_states, dst_type=torch.float8_e4m3fn
                )
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                hidden_states = self.layer.gate_up_proj((quantized_x, pertoken_scale))[0]
                # Execute activation concurrently with gmm2.
                maybe_wait_event(fused_moe_evts.before_gmm2)
                quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                    hidden_states,
                    topk_weight=None,
                    group_index=None,
                    dst_type=torch.float8_e4m3fn,
                    quant_mode=2,
                    clamp_value=self.swiglu_limit,
                )
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = self.layer.down_proj((quantized_x, swiglu_out_scale))[0]
            else:
                # Ensure the shared experts wait for hidden_states to be ready.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                part1_out = self.part1(hidden_states)
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = self.part2(hidden_states, part1_out)

        # Make sure the default stream waits for the shared experts stream to
        # finish.
        if self.multistream_overlap:
            torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

        if mode is SharedExpertParallelMode.FLASHCOMM_OFF_SHARED_EXPERT_DP_ON:
            assert local_dp_metadata is not None
            shared_out = self._finalize_local_dp_output(shared_out, local_dp_metadata)
        elif mode is SharedExpertParallelMode.FLASHCOMM_ON_SHARED_EXPERT_DP_OFF:
            shared_out = self._finalize_flashcomm_tp_output(shared_out)
        elif mode not in {
            SharedExpertParallelMode.FLASHCOMM_ON_SHARED_EXPERT_DP_ON,
            SharedExpertParallelMode.FLASHCOMM_OFF_SHARED_EXPERT_DP_OFF,
        }:
            raise AssertionError(f"Unsupported shared expert mode: {mode}")
        return shared_out
