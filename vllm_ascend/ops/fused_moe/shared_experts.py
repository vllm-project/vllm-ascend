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
from enum import Enum, auto
from functools import wraps

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_gather, tensor_model_parallel_reduce_scatter
from vllm.logger import logger
from vllm.model_executor.layers.activation import SiluAndMul, SituAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoEMethodBase

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, use_cann_megamoe
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.lora.fused_moe import has_lora
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import (
    PreparedSharedExpertInput,
    RoutedMoEMilestones,
)
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
        moe_layer: torch.nn.Module | None = None,
    ):
        self.layer = layer
        self.moe_config = moe_config
        self.moe_layer = moe_layer
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

        # Shared-expert fusion into the A5 MegaMoe operator: disabled until the
        # setup below proves the configuration is supported, then armed once
        # both shared linears have been processed into the MegaMoe layout.
        self.megamoe_shared_weights_ready = False
        self._megamoe_shared_fusion_disabled = False
        self._megamoe_shared_parts: dict[str, torch.Tensor] = {}

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

        self._setup_megamoe_shared_fusion()

    def _setup_megamoe_shared_fusion(self) -> None:
        """Fuse the shared expert into the A5 MegaMoe operator when supported.

        When active, each shared linear's ``process_weights_after_loading`` is
        wrapped to rebuild the weight into the MegaMoe layout; the payload is
        attached to the routed-expert layer as
        ``ascend_megamoe_shared_weights`` and forwarded to the operator as
        ``shared_l1_weights``/``shared_l2_weights``. The runner then skips the
        separate shared-expert forward whenever the step dispatches to
        MegaMoe. Any unsupported configuration keeps the standalone
        shared-expert execution unchanged.
        """
        if self.moe_layer is None or not use_cann_megamoe(get_current_vllm_config()):
            return
        # Only enable_fused_mc2=3 opts into fusing the shared expert into the
        # MegaMoe operator; =2 keeps the separate shared-expert forward.
        if not get_ascend_config().megamoe_shared_expert_fusion:
            return
        if not get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMOE_MXFP):
            return
        if self.quant_type != QuantType.W8A8MXFP:
            # Only MXFP8 is implemented: its shared weights rebuild cheaply
            # from the post-processed linear buffers. Packed-FP4 schemes
            # (W4A8MXFP/W4A4MXFP) would need a dedicated NZ_C0_32 rebuild.
            logger.info_once(
                "MegaMoe shared-expert fusion is not implemented for quant type %s; "
                "keeping the separate shared-expert forward.",
                self.quant_type,
            )
            self._megamoe_shared_fusion_disabled = True
            return
        # Output gating (e.g. Qwen3-Next expert_gate) happens outside the
        # operator and cannot be fused.
        if getattr(self.layer, "expert_gate", None) is not None:
            logger.info_once(
                "MegaMoe shared-expert fusion is not supported with expert gating; "
                "keeping the separate shared-expert forward."
            )
            self._megamoe_shared_fusion_disabled = True
            return
        # The operator consumes no shared bias.
        if (
            getattr(self.layer.gate_up_proj, "bias", None) is not None
            or getattr(self.layer.down_proj, "bias", None) is not None
        ):
            logger.info_once(
                "MegaMoe shared-expert fusion is not supported with shared-expert bias; "
                "keeping the separate shared-expert forward."
            )
            self._megamoe_shared_fusion_disabled = True
            return
        # The operator applies the routed activation to the shared experts too,
        # so only the standard SiluAndMul pairing can be fused.
        routed_activation = str(getattr(self.moe_layer, "activation", "silu"))
        if (
            routed_activation != "silu"
            or isinstance(self.layer.act_fn, SituAndMul)
            or not isinstance(self.layer.act_fn, SiluAndMul)
        ):
            logger.info_once(
                "MegaMoe shared-expert fusion is not supported for shared activation %s "
                "with routed activation %s; keeping the separate shared-expert forward.",
                type(self.layer.act_fn).__name__,
                routed_activation,
            )
            self._megamoe_shared_fusion_disabled = True
            return
        # Sequence-parallel shared experts rely on split/gather layouts that
        # the fused operator does not reproduce.
        if self.is_sequence_parallel:
            self._megamoe_shared_fusion_disabled = True
            return

        for linear, slot in (
            (self.layer.gate_up_proj, "w1"),
            (self.layer.down_proj, "w2"),
        ):
            self._wrap_linear_for_megamoe_fusion(linear, slot)

    def _wrap_linear_for_megamoe_fusion(self, linear: torch.nn.Module, slot: str) -> None:
        """Rebuild the shared weight right after its linear is processed.

        The wrapper is attached to the linear's own quant-method instance and
        ignores calls for other layers, so a shared quant-method instance
        stays safe.
        """
        quant_method = getattr(linear, "quant_method", None)
        original = getattr(quant_method, "process_weights_after_loading", None)
        if original is None:
            self._disable_megamoe_shared_fusion(
                f"shared linear {type(linear).__name__} has no process_weights_after_loading"
            )
            return

        @wraps(original)
        def wrapped_megamoe_collect(*args, **kwargs):
            result = original(*args, **kwargs)
            if args and args[0] is linear:
                self._collect_megamoe_shared_weight(slot, linear)
            return result

        quant_method.process_weights_after_loading = wrapped_megamoe_collect  # type: ignore

    def _collect_megamoe_shared_weight(self, slot: str, linear: torch.nn.Module) -> None:
        """Rebuild one shared linear into the A5 MegaMoe layout.

        MegaMoe expects (out, in) tensors while the processed linear buffer is
        (in, out) NZ, so a transpose+contiguous materializes the correct ND
        layout; the per-group scale becomes (n, k//2, 2), matching the routed
        per-expert scale shape.
        """
        if self._megamoe_shared_fusion_disabled:
            return
        padding = vars(linear).get("mxfp8_tp_padding", (0, 0))
        if padding != (0, 0):
            self._disable_megamoe_shared_fusion(
                f"padded shared-expert weights {padding} do not match the routed "
                "intermediate size required by the MegaMoe layout"
            )
            return
        expected_weight_shape = (
            (
                2 * self.moe_config.intermediate_size_per_partition,
                self.moe_config.hidden_dim,
            )
            if slot == "w1"
            else (
                self.moe_config.hidden_dim,
                self.moe_config.intermediate_size_per_partition,
            )
        )
        weight = linear.weight.data.transpose(0, 1).contiguous()
        scale = linear.weight_scale.data.transpose(0, 1).contiguous()
        if tuple(weight.shape) != expected_weight_shape:
            self._disable_megamoe_shared_fusion(
                f"shared-expert weight shape {tuple(weight.shape)} does not match the "
                f"routed expert layout {expected_weight_shape}"
            )
            return
        self._megamoe_shared_parts[slot] = weight
        self._megamoe_shared_parts[slot + "_scale"] = scale
        self._maybe_finish_megamoe_shared_weights()

    def _maybe_finish_megamoe_shared_weights(self) -> None:
        if self._megamoe_shared_fusion_disabled or self.moe_layer is None:
            return
        needed = ("w1", "w2", "w1_scale", "w2_scale")
        if not all(name in self._megamoe_shared_parts for name in needed):
            return
        # Attach to the routed-expert layer so the quant methods can pick the
        # payload up in get_fused_mc2_weights. Re-attaching on every rebuild
        # keeps RL weight reloads pointing at the fresh tensors.
        self.moe_layer.ascend_megamoe_shared_weights = (
            [self._megamoe_shared_parts["w1"]],
            [self._megamoe_shared_parts["w2"]],
            [self._megamoe_shared_parts["w1_scale"]],
            [self._megamoe_shared_parts["w2_scale"]],
        )
        if not self.megamoe_shared_weights_ready:
            self.megamoe_shared_weights_ready = True
            logger.info_once("Fused the shared expert into the A5 MegaMoe operator.")

    def _disable_megamoe_shared_fusion(self, reason: str) -> None:
        if not self._megamoe_shared_fusion_disabled:
            logger.info_once(
                "Disabling MegaMoe shared-expert fusion: %s; keeping the separate shared-expert forward.",
                reason,
            )
        self._megamoe_shared_fusion_disabled = True
        self._megamoe_shared_parts.clear()

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

    def part1(self, hidden_states: torch.Tensor):
        shared_gate_up, _ = self.layer.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def apply_activation(self, shared_gate_up: torch.Tensor):
        return self.layer.act_fn(shared_gate_up)  # type: ignore

    def part2(self, hidden_states: torch.Tensor, shared_act: torch.Tensor):
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

    def _run_a8_int_mlp(
        self,
        hidden_states: torch.Tensor,
        milestones: RoutedMoEMilestones,
        down_projection_ready: torch.npu.Event | None,
        down_projection_milestone: str,
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        # Vector dynamic quant overlaps the Cube-heavy router gate.
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        self._wait_for_milestone(
            milestones.router_output_ready,
            "router_output_ready",
        )
        # Gate-up is Cube-heavy. From router_output_ready it overlaps the
        # routed AllGather prepare communication, or All2All preprocessing and
        # its forward exchange.
        gate_up = torch_npu.npu_quant_matmul(
            quantized_x,
            self.layer.gate_up_proj.weight,
            self.layer.gate_up_proj.weight_scale,
            pertoken_scale=None,
            bias=None,
            output_dtype=torch.int32,
        )

        self._wait_for_routed_stage(
            milestones,
            milestones.routed_gmm2_start,
            "routed_gmm2_start",
        )
        if self.situ_activation is not None:
            quantized_x, swiglu_out_scale = torch.ops._C_ascend.dequant_situ_quant(
                x=gate_up,
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
                x=gate_up,
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
            down_projection_ready,
            down_projection_milestone,
        )
        return torch_npu.npu_quant_matmul(
            quantized_x,
            self.layer.down_proj.weight,
            self.layer.down_proj.weight_scale,
            pertoken_scale=swiglu_out_scale,
            bias=None,
            output_dtype=original_dtype,
        )

    def _run_a8_mxfp_mlp(
        self,
        hidden_states: torch.Tensor,
        milestones: RoutedMoEMilestones,
        down_projection_ready: torch.npu.Event | None,
        down_projection_milestone: str,
    ) -> torch.Tensor:
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
            hidden_states,
            dst_type=torch.float8_e4m3fn,
        )
        self._wait_for_milestone(
            milestones.router_output_ready,
            "router_output_ready",
        )
        gate_up = self.layer.gate_up_proj((quantized_x, pertoken_scale))[0]

        self._wait_for_routed_stage(
            milestones,
            milestones.routed_gmm2_start,
            "routed_gmm2_start",
        )
        if self.situ_activation is not None:
            quantized_x, swiglu_out_scale = torch.ops._C_ascend.situ_mx_quant(
                x=gate_up,
                beta=self.situ_activation.beta,
                linear_beta=self.situ_activation.linear_beta or 0.0,
                activate_left=True,
                dst_type=SITU_MX_DST_TYPE_E4M3FN,
            )
        else:
            quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                gate_up,
                topk_weight=None,
                group_index=None,
                dst_type=torch.float8_e4m3fn,
                quant_mode=2,
                clamp_value=self.swiglu_limit,
            )
        self._wait_for_routed_stage(
            milestones,
            down_projection_ready,
            down_projection_milestone,
        )
        return self.layer.down_proj((quantized_x, swiglu_out_scale))[0]

    def _run_linear_wrapped_mlp(
        self,
        hidden_states: torch.Tensor,
        milestones: RoutedMoEMilestones,
        down_projection_ready: torch.npu.Event | None,
        down_projection_milestone: str,
    ) -> torch.Tensor:
        self._wait_for_milestone(
            milestones.router_output_ready,
            "router_output_ready",
        )
        gate_up = self.part1(hidden_states)

        self._wait_for_routed_stage(
            milestones,
            milestones.routed_gmm2_start,
            "routed_gmm2_start",
        )
        shared_act = self.apply_activation(gate_up)
        self._wait_for_routed_stage(
            milestones,
            down_projection_ready,
            down_projection_milestone,
        )
        return self.part2(hidden_states, shared_act)

    def _run_shared_mlp(
        self,
        hidden_states: torch.Tensor,
        milestones: RoutedMoEMilestones,
    ) -> torch.Tensor:
        # The shared Down projection can overlap the routed combine/finalize.
        # Only the later SP output collective must wait for routed finalize.
        down_projection_ready = milestones.routed_combine_start
        down_projection_milestone = "routed_combine_start"

        path = self._select_mlp_path()
        if path is SharedExpertMLPPath.A8_INT_FUSED:
            return self._run_a8_int_mlp(
                hidden_states,
                milestones,
                down_projection_ready,
                down_projection_milestone,
            )
        if path is SharedExpertMLPPath.A8_MXFP_FUSED:
            return self._run_a8_mxfp_mlp(
                hidden_states,
                milestones,
                down_projection_ready,
                down_projection_milestone,
            )
        return self._run_linear_wrapped_mlp(
            hidden_states,
            milestones,
            down_projection_ready,
            down_projection_milestone,
        )

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
        if has_quantized_shared_without_lora and self.quant_type in (QuantType.W8A8, QuantType.W4A8):
            return SharedExpertMLPPath.A8_INT_FUSED
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
    ) -> torch.Tensor:
        mode = self.parallel_mode()
        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap):
            hidden_states, local_dp_metadata = self._prepare_execution_input(
                prepared_input,
                milestones,
                mode,
            )
            shared_out = self._run_shared_mlp(hidden_states, milestones)
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
