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
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from typing import TypeAlias

import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import tensor_model_parallel_all_gather, tensor_model_parallel_reduce_scatter
from vllm.logger import logger
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoEMethodBase

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.lora.fused_moe import has_lora
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import (
    PreparedSharedExpertInput,
    RoutedMoEMilestones,
)
from vllm_ascend.quantization.methods.base import PreparedLinearInput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import npu_stream_switch, shared_experts_calculation_stream

# CANN uses 36 to select FP8 E4M3FN output for situ_mx_quant.
SITU_MX_DST_TYPE_E4M3FN = 36


class SharedExpertParallelMode(Enum):
    """Effective activation and weight layout for a shared-expert forward."""

    TENSOR_PARALLEL = auto()  # Full activations, TP-sharded weights.
    SHARED_EXPERT_DATA_PARALLEL_ONLY = auto()  # Full activations, replicated weights (DP only).
    SEQUENCE_PARALLEL_ONLY = auto()  # Sharded activations, TP-sharded weights (SP only).
    SEQUENCE_PARALLEL_SEDP = auto()  # Sharded activations, replicated weights (SP + DP).


class SharedExpertMLPPath(Enum):
    """Shared MLP implementation selected for a quantization scenario."""

    A8_INT_FUSED = auto()  # W8A8/W4A8: explicit A8 quant + fused activation quant.
    A8_MXFP_FUSED = auto()  # W4A8MXFP: explicit MXFP8 activation pipeline.
    LINEAR_WRAPPER = auto()  # Dense, other quant schemes, or any active LoRA.


LinearInput: TypeAlias = torch.Tensor | PreparedLinearInput | tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class PreparedSharedExpertExecution:
    """Shared MLP input preparation submitted before the router matmul.

    ``gate_up_input`` may contain scheme-owned activation quantization that is
    already queued on the shared-expert stream. Gate-Up consumes it immediately
    after ``router_output_ready``; activation and Down are appended after the
    routed path exposes its stage milestones.
    """

    hidden_states: torch.Tensor
    gate_up_input: LinearInput
    local_dp_metadata: tuple[int, int] | None


@dataclass(frozen=True)
class SubmittedSharedExpertGateUp:
    """Gate-Up work already submitted on the shared-expert stream."""

    hidden_states: torch.Tensor
    gate_up: torch.Tensor
    gate_up_input_scale: torch.Tensor | None
    local_dp_metadata: tuple[int, int] | None
    path: SharedExpertMLPPath


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
        self.shared_expert_input_size = getattr(
            layer.gate_up_proj,
            "input_size",
            self.hidden_size,
        )
        self.in_dtype = moe_config.in_dtype
        self.swiglu_limit = 0.0 if moe_config.swiglu_limit is None else moe_config.swiglu_limit
        self.swiglu_alpha = 1.0 if moe_config.swiglu_alpha is None else moe_config.swiglu_alpha
        self.swiglu_beta = 0.0 if moe_config.swiglu_beta is None else moe_config.swiglu_beta
        self.is_sequence_parallel = moe_config.is_sequence_parallel
        self.situ_activation = layer.act_fn if isinstance(layer.act_fn, SituAndMul) else None
        self.quant_type = quant_type
        self.lora_context = None
        ascend_config = get_ascend_config()
        self.multistream_overlap = ascend_config.multistream_overlap_shared_expert
        self.weights_replicated = ascend_config.enable_shared_expert_dp

        if self.multistream_overlap:
            # Wrap the quant_method's process_weights_after_loading to validate that
            # splitting shared expert computation (gate_up projection, activation,
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
            torch.rand(
                10,
                self.shared_expert_input_size,
                device="npu",
                dtype=self.in_dtype,
            )
            * 2
            - 1
        )  # Random input for testing, scoped to [-1, 1]

        integrated_out = self.layer(test_input)
        part1_out = self.part1(test_input)
        shared_act = self.apply_activation(part1_out)
        split_out = self.part2(test_input, shared_act)

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
                self.shared_expert_input_size,
                self.in_dtype,
            )
            raise ValueError("FusedMoE shared experts split computation does not match the integrated computation.")
        logger.info_once(
            "[fused_moe/layer] Shared expert split computation validation passed."
            " Integrated and split-path results are consistent."
        )

    def part1(
        self,
        hidden_states: LinearInput,
    ):
        shared_gate_up, _ = self.layer.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def apply_activation(self, shared_gate_up: torch.Tensor):
        return self.layer.act_fn(shared_gate_up)  # type: ignore

    def part2(
        self,
        hidden_states: torch.Tensor,
        down_input: LinearInput,
    ):
        shared_out, _ = self.layer.down_proj(down_input)  # type: ignore

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
            return SharedExpertParallelMode.TENSOR_PARALLEL

        if self.moe_config.is_sequence_parallel:
            # SP has already sharded the token dimension before entering the
            # runner. Replicated weights (SP+DP) compute directly on the
            # shard; TP-sharded weights (SP-only) gather the shard first and
            # reduce-scatter the output back.
            if self.weights_replicated:
                return SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP
            return SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
        if self.weights_replicated:
            return SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY
        return SharedExpertParallelMode.TENSOR_PARALLEL

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

    def _gather_sp_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gather SP-sharded activations to the full sequence via TP all-gather + unpad.

        SP shards the token dimension across the TP group as [ceil(T/TP), H]
        per rank (``sequence_parallel_chunk`` pads T to a multiple of TP).
        TP-sharded shared-expert weights need the full activations, so
        all-gather the shards back to [T_padded, H] and drop the padding rows.
        """
        gathered = tensor_model_parallel_all_gather(hidden_states, dim=0)
        return gathered[: _EXTRA_CTX.num_tokens]

    def _pad_and_reduce_scatter(self, shared_out: torch.Tensor) -> torch.Tensor:
        """Pad the full output to a TP multiple, then reduce-scatter to the SP shard.

        Called on exit of the SP-only path (TP-sharded weights) to convert the
        full [T, H] output back to this rank's SP shard [ceil(T/TP), H]. The
        reduce-scatter also sums the TP-partial down-projection results
        (down_proj is built with reduce_results=False), replacing the usual
        TP all-reduce.
        """
        tp_size = self.moe_config.tp_group.world_size
        original_num_tokens = shared_out.shape[0]
        pad_size = (tp_size - original_num_tokens % tp_size) % tp_size
        if pad_size > 0:
            shared_out = F.pad(shared_out, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(shared_out, dim=0)

    def prepare_input_async(
        self,
        hidden_states: torch.Tensor,
    ) -> PreparedSharedExpertInput:
        """Start the SP-only input all-gather on the shared-expert stream.

        The caller must wait for ``ready_event`` before enqueueing routed-path
        collectives.  This method is intended to overlap the gather only with
        a routed input transform such as Kimi-K3's latent projection.
        """
        if not (self.multistream_overlap and self.parallel_mode() is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY):
            return PreparedSharedExpertInput(hidden_states=hidden_states)

        input_ready = torch.npu.current_stream().record_event()
        with npu_stream_switch(shared_experts_calculation_stream(), enabled=True):
            torch.npu.current_stream().wait_event(input_ready)
            # Keep TP padding across the custom-op boundary; the shared MLP
            # trims it before computation.
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            all_gather_done = torch.npu.current_stream().record_event()
        return PreparedSharedExpertInput(
            hidden_states=hidden_states,
            is_gathered=True,
            ready_event=all_gather_done,
        )

    def prepare_input_before_routed(
        self,
        hidden_states: torch.Tensor,
    ) -> PreparedSharedExpertInput:
        """Prepare SP-only input on the calling stream before routed MoE.

        The TP all-gather deliberately stays on the default/routed stream.
        Running it on the shared-expert stream while routed EP collectives run
        on the default stream can deadlock HCCL when ranks make different
        progress across the two process groups.
        """
        if not (self.multistream_overlap and self.parallel_mode() is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY):
            return PreparedSharedExpertInput(hidden_states=hidden_states)

        # Keep TP padding across the custom-op boundary; the shared MLP trims
        # it before computation. The caller records shared_input_ready after
        # this returns, handing the gathered tensor to the auxiliary stream.
        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
        return PreparedSharedExpertInput(
            hidden_states=hidden_states,
            is_gathered=True,
        )

    def _wait_for_milestone(
        self,
        event: torch.npu.Event | None,
        name: str,
    ) -> None:
        """Wait for a required routed milestone when multistream is enabled."""
        if not self.multistream_overlap:
            return
        if event is None:
            raise RuntimeError(f"Missing {name} event while shared-expert multistream is enabled.")
        torch.npu.current_stream().wait_event(event)

    def _wait_for_routed_stage(
        self,
        milestones: RoutedMoEMilestones,
        event: torch.npu.Event | None,
        name: str,
    ) -> None:
        """Wait when the routed implementation exposes stage boundaries.

        Fused routed implementations such as FusedMC2 execute their stages in
        one operator and therefore cannot provide intermediate events. Shared
        experts remain independent and run as one concurrent auxiliary-stream
        workload in that case.
        """
        if milestones.has_fine_grained_stage_events:
            self._wait_for_milestone(event, name)

    def _prepare_execution_input(
        self,
        prepared_input: PreparedSharedExpertInput,
        milestones: RoutedMoEMilestones,
        mode: SharedExpertParallelMode,
    ) -> tuple[torch.Tensor, tuple[int, int] | None]:
        self._wait_for_milestone(
            milestones.shared_input_ready,
            "shared_input_ready",
        )
        hidden_states = prepared_input.hidden_states
        local_dp_metadata = None
        if mode is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
            # Full activations + replicated weights: shard tokens locally,
            # run the MLP, then gather its complete output.
            hidden_states, local_dp_metadata = self._prepare_local_dp_input(hidden_states)
        elif mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY and not prepared_input.is_gathered:
            # TP-sharded weights require full activations. Multistream starts
            # this gather before the routed path; the serial path does it here.
            hidden_states = self._gather_sp_input(hidden_states)
        elif mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY:
            # An early gather keeps padding across the custom-op boundary so
            # tracing sees the explicit gathered dependency. The MLP consumes
            # only the original tokens.
            hidden_states = hidden_states[: _EXTRA_CTX.num_tokens]
        return hidden_states, local_dp_metadata

    def prepare_for_router_overlap(
        self,
        prepared_input: PreparedSharedExpertInput,
        shared_input_ready: torch.npu.Event,
    ) -> PreparedSharedExpertExecution | None:
        """Queue Gate-Up input preparation before the router matmul.

        Recording an event before the router is insufficient by itself: if the
        auxiliary-stream work is submitted only after routed MoE, device-side
        execution still starts late. This method is called immediately after
        ``shared_input_ready`` is recorded, so Vector/AIV activation
        preparation can run concurrently with the Cube-heavy router matmul.

        Unsupported wrappers return the unmodified tensor as ``gate_up_input``;
        their complete registered linear is still ordered after
        ``router_output_ready`` and is submitted before routed TopK/dispatch.
        """
        if not self.multistream_overlap:
            return None

        mode = self.parallel_mode()
        milestones = RoutedMoEMilestones(shared_input_ready=shared_input_ready)
        with npu_stream_switch(shared_experts_calculation_stream(), enabled=True):
            hidden_states, local_dp_metadata = self._prepare_execution_input(
                prepared_input,
                milestones,
                mode,
            )
            gate_up_input = self._prepare_gate_up_input(hidden_states)
        return PreparedSharedExpertExecution(
            hidden_states=hidden_states,
            gate_up_input=gate_up_input,
            local_dp_metadata=local_dp_metadata,
        )

    def _prepare_gate_up_input(self, hidden_states: torch.Tensor) -> LinearInput:
        """Prepare the selected Gate-Up path without running its matmul."""
        path = self._select_mlp_path()
        if path is SharedExpertMLPPath.A8_INT_FUSED:
            return torch_npu.npu_dynamic_quant(hidden_states)
        if path is SharedExpertMLPPath.A8_MXFP_FUSED:
            return torch_npu.npu_dynamic_mx_quant(
                hidden_states,
                dst_type=torch.float8_e4m3fn,
            )
        return self._prepare_linear_input_for_overlap(self.layer.gate_up_proj, hidden_states)

    def _prepare_linear_input_for_overlap(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
    ) -> LinearInput:
        """Run scheme-owned Vector/AIV preparation before a stage barrier."""
        if not self.multistream_overlap or has_lora(self.lora_context):
            return hidden_states

        linear_method = getattr(layer, "quant_method", None)
        quant_scheme = getattr(linear_method, "quant_method", linear_method)
        prepare_input = getattr(quant_scheme, "prepare_input_for_overlap", None)
        if prepare_input is None:
            return hidden_states

        prepared_input = prepare_input(layer, hidden_states)
        return hidden_states if prepared_input is None else prepared_input

    def enqueue_gate_up_after_router(
        self,
        prepared_execution: PreparedSharedExpertExecution,
        router_output_ready: torch.npu.Event | None,
    ) -> SubmittedSharedExpertGateUp:
        """Submit Gate-Up before routed TopK/dispatch are enqueued.

        Gate-Up remains ordered after the router through ``router_output_ready``
        but is placed on the auxiliary stream immediately, allowing its Cube
        work to cover routed TopK and dispatch instead of missing that window
        because of host-side submission order.
        """
        path = self._select_mlp_path()
        gate_up_input_scale = None
        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap):
            self._wait_for_milestone(router_output_ready, "router_output_ready")
            if path is SharedExpertMLPPath.A8_INT_FUSED:
                # W8A8/W4A8: INT8 activations with INT8 or INT4 weights.
                assert isinstance(prepared_execution.gate_up_input, tuple)
                quantized_x, gate_up_input_scale = prepared_execution.gate_up_input
                gate_up = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self.layer.gate_up_proj.weight,
                    self.layer.gate_up_proj.weight_scale,
                    pertoken_scale=None,
                    bias=None,
                    output_dtype=torch.int32,
                )
            elif path is SharedExpertMLPPath.A8_MXFP_FUSED:
                # W4A8MXFP only: MXFP8 activations with MXFP4 weights.
                # W8A8MXFP remains on the registered linear-wrapper path.
                assert isinstance(prepared_execution.gate_up_input, tuple)
                quantized_x, pertoken_scale = prepared_execution.gate_up_input
                gate_up = self.layer.gate_up_proj((quantized_x, pertoken_scale))[0]
            else:
                gate_up = self.part1(prepared_execution.gate_up_input)

        return SubmittedSharedExpertGateUp(
            hidden_states=prepared_execution.hidden_states,
            gate_up=gate_up,
            gate_up_input_scale=gate_up_input_scale,
            local_dp_metadata=prepared_execution.local_dp_metadata,
            path=path,
        )

    def _run_a8_int_mlp(
        self,
        submitted_gate_up: SubmittedSharedExpertGateUp,
        milestones: RoutedMoEMilestones,
    ) -> torch.Tensor:
        """Finish the fused W8A8/W4A8 INT-activation shared MLP.

        Both quantization modes use INT8 activations. W8A8 supplies INT8
        weights, while W4A8 supplies INT4 weights. Gate-Up has already been
        submitted before routed TopK/dispatch; this stage performs fused
        activation/requantization and the Down projection.
        """
        pertoken_scale = submitted_gate_up.gate_up_input_scale
        assert pertoken_scale is not None
        self._wait_for_routed_stage(
            milestones,
            milestones.shared_activation_overlap_start,
            "shared_activation_overlap_start",
        )
        if self.situ_activation is not None:
            quantized_x, swiglu_out_scale = torch.ops._C_ascend.dequant_situ_quant(
                x=submitted_gate_up.gate_up,
                weight_scale=self.layer.gate_up_proj.weight_scale_fp32,
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=None,
                beta=self.situ_activation.beta,
                linear_beta=self.situ_activation.linear_beta or 0.0,
                activate_left=True,
                quant_mode="dynamic",
            )
        else:
            quantized_x, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                x=submitted_gate_up.gate_up,
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
                    if not get_current_hardware_profile().supports(HardwareCapability.FUSED_SWIGLU_TUNING_ARGS)
                    else {"glu_alpha": self.swiglu_alpha, "glu_bias": self.swiglu_beta}
                ),
            )
        self._wait_for_routed_stage(
            milestones,
            milestones.routed_combine_start,
            "routed_combine_start",
        )
        return torch_npu.npu_quant_matmul(
            quantized_x,
            self.layer.down_proj.weight,
            self.layer.down_proj.weight_scale,
            pertoken_scale=swiglu_out_scale,
            bias=None,
            output_dtype=submitted_gate_up.hidden_states.dtype,
        )

    def _run_a8_mxfp_mlp(
        self,
        submitted_gate_up: SubmittedSharedExpertGateUp,
        milestones: RoutedMoEMilestones,
    ) -> torch.Tensor:
        """Finish the fused W4A8MXFP shared MLP.

        This path means MXFP4 weights with MXFP8 activations. W8A8MXFP does
        not enter here; it keeps the registered linear wrapper so its scheme-
        owned padding, scale algorithm, and output dtype remain intact.
        Gate-Up has already been submitted before routed TopK/dispatch.
        """
        self._wait_for_routed_stage(
            milestones,
            milestones.shared_activation_overlap_start,
            "shared_activation_overlap_start",
        )
        if self.situ_activation is not None:
            quantized_x, swiglu_out_scale = torch.ops._C_ascend.situ_mx_quant(
                x=submitted_gate_up.gate_up,
                beta=self.situ_activation.beta,
                linear_beta=self.situ_activation.linear_beta or 0.0,
                activate_left=True,
                dst_type=SITU_MX_DST_TYPE_E4M3FN,
            )
        else:
            quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                submitted_gate_up.gate_up,
                topk_weight=None,
                group_index=None,
                dst_type=torch.float8_e4m3fn,
                quant_mode=2,
                clamp_value=self.swiglu_limit,
            )
        self._wait_for_routed_stage(
            milestones,
            milestones.routed_combine_start,
            "routed_combine_start",
        )
        return self.layer.down_proj((quantized_x, swiglu_out_scale))[0]

    def _finish_linear_wrapped_mlp(
        self,
        submitted_gate_up: SubmittedSharedExpertGateUp,
        milestones: RoutedMoEMilestones,
    ) -> torch.Tensor:
        self._wait_for_routed_stage(
            milestones,
            milestones.shared_activation_overlap_start,
            "shared_activation_overlap_start",
        )
        shared_act = self.apply_activation(submitted_gate_up.gate_up)
        # MC2 and All2All prepare Down during routed GMM1/GMM2; AllGather
        # starts at GMM2. The Down matmul remains behind combine.
        down_input = self._prepare_linear_input_for_overlap(self.layer.down_proj, shared_act)
        self._wait_for_routed_stage(
            milestones,
            milestones.routed_combine_start,
            "routed_combine_start",
        )
        return self.part2(submitted_gate_up.hidden_states, down_input)

    def _finish_shared_mlp(
        self,
        submitted_gate_up: SubmittedSharedExpertGateUp,
        milestones: RoutedMoEMilestones,
    ) -> torch.Tensor:
        if submitted_gate_up.path is SharedExpertMLPPath.A8_INT_FUSED:
            return self._run_a8_int_mlp(submitted_gate_up, milestones)
        if submitted_gate_up.path is SharedExpertMLPPath.A8_MXFP_FUSED:
            return self._run_a8_mxfp_mlp(submitted_gate_up, milestones)
        return self._finish_linear_wrapped_mlp(submitted_gate_up, milestones)

    def _select_mlp_path(self) -> SharedExpertMLPPath:
        """Select a path without changing the quantization scheme's math.

        Only schemes with a proven split activation-quant pipeline bypass the
        registered linear wrappers.  W8A8MXFP, W8A8FP, W4A4MXFP and other
        schemes continue through those wrappers.  Active LoRA always needs the
        wrapper path so its adapter computation is preserved.
        """
        has_quantized_shared_without_lora = (
            not has_lora(self.lora_context)
            and hasattr(self.layer.gate_up_proj, "weight_scale")
            and hasattr(self.layer.down_proj, "weight_scale")
        )
        # W8A8 and W4A8 share an INT8-activation implementation; A8 names
        # the activation width and does not imply that weights are INT8.
        if has_quantized_shared_without_lora and self.quant_type in (QuantType.W8A8, QuantType.W4A8):
            return SharedExpertMLPPath.A8_INT_FUSED
        # This MXFP path currently represents W4A8MXFP only (MXFP4 weights
        # with MXFP8 activations). W8A8MXFP keeps its registered linear
        # wrapper and uses prepare_input_for_overlap when it is safe.
        if has_quantized_shared_without_lora and self.quant_type == QuantType.W4A8MXFP:
            return SharedExpertMLPPath.A8_MXFP_FUSED
        return SharedExpertMLPPath.LINEAR_WRAPPER

    def wait_for_output(self) -> None:
        """Join a deferred SP shared-output collective on the current stream."""
        # This join is intentionally stream-based. The shared forward executes
        # inside an opaque custom op for ACL Graph, so a Python-side Event
        # assigned inside that op is not available while its fake path traces.
        torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

    def forward(
        self,
        prepared_input: PreparedSharedExpertInput,
        milestones: RoutedMoEMilestones,
        defer_output_wait: bool = False,
        submitted_gate_up: SubmittedSharedExpertGateUp | None = None,
    ) -> torch.Tensor:
        mode = self.parallel_mode()
        if submitted_gate_up is None:
            with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap):
                hidden_states, local_dp_metadata = self._prepare_execution_input(
                    prepared_input,
                    milestones,
                    mode,
                )
                gate_up_input = self._prepare_gate_up_input(hidden_states)
            prepared_execution = PreparedSharedExpertExecution(
                hidden_states=hidden_states,
                gate_up_input=gate_up_input,
                local_dp_metadata=local_dp_metadata,
            )
            submitted_gate_up = self.enqueue_gate_up_after_router(
                prepared_execution,
                milestones.router_output_ready,
            )

        local_dp_metadata = submitted_gate_up.local_dp_metadata
        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap):
            shared_out = self._finish_shared_mlp(submitted_gate_up, milestones)
            if self.multistream_overlap and mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY:
                self._wait_for_milestone(
                    milestones.routed_finalize_done,
                    "routed_finalize_done",
                )
                shared_out = self._pad_and_reduce_scatter(shared_out)

        if self.multistream_overlap and (
            mode is not SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY or not defer_output_wait
        ):
            self.wait_for_output()

        if mode is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
            assert local_dp_metadata is not None
            shared_out = self._finalize_local_dp_output(shared_out, local_dp_metadata)
        elif mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY and not self.multistream_overlap:
            shared_out = self._pad_and_reduce_scatter(shared_out)
        return shared_out
