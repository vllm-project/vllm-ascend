# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""DeepSeek V4.1 FP8 x FP4 MegaMoE on Ascend 950.

Like the CUDA adapter, load checkpoint weights once into the kernel layout,
stage local tokens and global expert IDs into symmetric memory, then run the
fused dispatch/GEMM/SwiGLU/combine operation, including the shared FP8 experts.
Only TP1, linear EP8 and serialized target/draft calls are supported by this
initial V4.1 integration.
"""

import math

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config


class DeepGemmV41Experts:
    """One layer's weights over process-group-owned symmetric scratch."""

    def __init__(self, layer):
        # Import in workers only; the dependency owns native NPU state.
        import deep_gemm

        config = get_current_vllm_config()
        text = config.model_config.hf_text_config
        # vLLM flattens HF text_config into DeepseekV41Config.
        if text.model_type != "deepseek_v41":
            raise ValueError("deepgemm_mega_moe supports DeepSeek V4.1 only")
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("V4.1 DeepGEMM MegaMoE requires TP1")
        group = get_ep_group()
        if group.world_size != 8:
            raise ValueError("V4.1 DeepGEMM MegaMoE requires EP8")
        if get_ascend_config().enable_fused_mc2:
            raise ValueError("Select DeepGEMM with enable_fused_mc2=0")
        if config.parallel_config.enable_eplb or layer.global_redundant_expert_num:
            raise ValueError("This V4.1 backend requires linear EP without EPLB replicas")
        if getattr(config.parallel_config, "expert_placement_strategy", "linear") != "linear":
            raise ValueError("V4.1 DeepGEMM requires linear expert placement")
        if getattr(config.parallel_config, "enable_dbo", False):
            raise ValueError("V4.1 DeepGEMM symmetric scratch does not support overlapping microbatches")
        if layer.apply_router_weight_on_input:
            raise ValueError("V4.1 MegaMoE expects output routing weights")
        moe = layer.moe_config
        if getattr(moe.activation, "value", moe.activation) != "silu":
            raise ValueError("V4.1 MegaMoE requires SwiGLU")

        self.kernel = deep_gemm.fp8_fp4_mega_moe
        self.hidden = layer.w2_weight.shape[1]
        self.intermediate = layer.w13_weight.shape[1] // 2
        self.num_experts = moe.num_experts
        self.topk = moe.experts_per_token
        self.activation_clamp = moe.swiglu_limit
        self.max_tokens = config.scheduler_config.max_num_batched_tokens
        self.routed_scaling_factor = float(getattr(layer, "_deepgemm_v41_routed_scaling_factor", 1.0))
        if not math.isfinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0:
            raise ValueError("V4.1 MegaMoE requires a positive finite routed scaling factor")
        self._prepare_shared_weights(layer, text, deep_gemm.transform_weights_for_mega_moe)

        # Byte-paired UE8M0 is packed along K, not the NZ-transposed layout
        # used by the existing CANN grouped matmul backend.
        (w13, s13), (w2, s2) = deep_gemm.transform_weights_for_mega_moe(
            (
                layer.w13_weight.data.view(torch.int8),
                layer.w13_weight_scale.data.view(torch.uint8).contiguous().view(torch.int16),
            ),
            (
                layer.w2_weight.data.view(torch.int8),
                layer.w2_weight_scale.data.view(torch.uint8).contiguous().view(torch.int16),
            ),
        )
        layer.w13_weight.data = w13
        layer.w13_weight_scale.data = s13
        layer.w2_weight.data = w2
        layer.w2_weight_scale.data = s2

        # The target loads before DSpark. All layers with the same shape use
        # one view; draft layouts reuse the target allocation through the
        # dependency's public base= API, which checks group and capacity.
        key = (self.num_experts, self.topk, self.hidden, self.intermediate, self.max_tokens, self.num_shared_experts)
        buffers = getattr(group, "_deepgemm_v41_buffers", None)
        if buffers is None:
            buffers = {}
            group._deepgemm_v41_buffers = buffers
        if key not in buffers:
            base = next(iter(buffers.values()), None)
            if base is None and self.num_experts != text.n_routed_experts:
                raise RuntimeError("Initialize the V4.1 target before its DSpark draft")
            buffers[key] = deep_gemm.SymmBuffer(
                group.device_group,
                self.num_experts,
                self.max_tokens,
                self.topk,
                self.hidden,
                self.intermediate,
                num_shared_experts=self.num_shared_experts,
                base=base,
            )
            logger.info(
                "DeepGEMM V4.1 MegaMoE: experts=%d topk=%d shared=%d H=%d I=%d capacity=%d bytes=%d shared_base=%s",
                self.num_experts,
                self.topk,
                self.num_shared_experts,
                self.hidden,
                self.intermediate,
                self.max_tokens,
                buffers[key].num_bytes,
                base is not None,
            )
        self.buffer = buffers[key]

    @staticmethod
    def _shared_weight_pair(linear):
        """Expand checkpoint block scales to per-output, per-32 UE8M0."""
        weight, scale = linear.weight.data, linear.weight_scale.data
        if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
            raise ValueError("V4.1 shared experts require raw ND FP8 E4M3 weights")
        rows, columns = weight.shape
        if scale.ndim != 2 or not all(scale.shape) or rows % scale.shape[0] or columns % scale.shape[1]:
            raise ValueError("Shared checkpoint scale shape does not divide the weight shape")
        row_repeat, block_k = rows // scale.shape[0], columns // scale.shape[1]
        if row_repeat not in (1, block_k) or block_k not in (32, 128) or columns % 64:
            raise ValueError("Shared experts require per-32 or square 32/128 checkpoint block scales")
        if scale.dtype == torch.float32:
            # Match the DS checkpoint conversion: FP32 powers of two to UE8M0.
            scale = ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        elif scale.dtype in (torch.uint8, getattr(torch, "float8_e8m0fnu", None)):
            scale = scale.view(torch.uint8)
        else:
            raise ValueError("Shared checkpoint scales must contain FP32 or UE8M0 bytes")
        scale = scale.repeat_interleave(row_repeat, dim=0).repeat_interleave(block_k // 32, dim=1)
        return weight.contiguous(), scale.contiguous().view(torch.int16)

    def _prepare_shared_weights(self, layer, text, transform):
        self.num_shared_experts = 0
        self.shared_l1_weights = None
        self.shared_l2_weights = None
        expected = getattr(text, "n_shared_experts", None)
        if expected is None:
            raise ValueError("V4.1 MegaMoE requires explicit n_shared_experts metadata")
        reference = getattr(layer, "_deepgemm_v41_shared_experts", None)
        shared = reference() if reference is not None else None
        if shared is None:
            if expected:
                raise RuntimeError(
                    "V4.1 shared experts are configured but their module reference is missing or expired"
                )
            return
        if getattr(shared, "_deepgemm_v41_weights_transformed", False):
            raise RuntimeError("Shared expert weights have already been converted for V4.1 MegaMoE")
        gate_up, down = shared.gate_up_proj, shared.down_proj
        if gate_up.weight.ndim != 2 or down.weight.ndim != 2:
            raise ValueError("Shared expert weights must remain in their checkpoint ND layout")
        shared_intermediate = down.weight.shape[1]
        count, remainder = divmod(shared_intermediate, self.intermediate)
        if remainder or count not in (1, 2) or count != expected:
            raise ValueError("Shared expert width must match n_shared_experts times the routed intermediate width")
        if gate_up.weight.shape != (2 * shared_intermediate, self.hidden) or down.weight.shape[0] != self.hidden:
            raise ValueError("Shared expert gate/up and down weight shapes are inconsistent")
        shared_l1, shared_l2 = transform(self._shared_weight_pair(gate_up), self._shared_weight_pair(down))
        gate_up.weight.data, gate_up.weight_scale.data = shared_l1
        down.weight.data, down.weight_scale.data = shared_l2
        self.shared_l1_weights = (gate_up.weight, gate_up.weight_scale)
        self.shared_l2_weights = (down.weight, down.weight_scale)
        self.num_shared_experts = count
        shared._deepgemm_v41_weights_transformed = True

    def apply(self, layer, x, topk_weights, topk_ids):
        from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

        if x.dtype != torch.bfloat16 or x.ndim != 2:
            raise ValueError("V4.1 MegaMoE expects local BF16 [tokens, hidden] input")
        count = x.shape[0]
        if count > self.max_tokens:
            raise ValueError(f"V4.1 MegaMoE token count {count} exceeds {self.max_tokens}")

        before = torch.npu.current_stream().record_event()
        quantized, scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        self.buffer.x[:count].copy_(quantized)
        self.buffer.x_sf[:count].copy_(scale.reshape(count, self.hidden // 32).view(torch.int16))
        self.buffer.topk_idx[:count].copy_(topk_ids)
        # DeepInfer scales routed weights before SwiGLU/requantization. Shared
        # experts have their own kernel path and must not receive this factor.
        routed_weights = (
            topk_weights * self.routed_scaling_factor if self.routed_scaling_factor != 1.0 else topk_weights
        )
        self.buffer.topk_weights[:count].copy_(routed_weights)
        out = torch.empty((count, self.hidden), dtype=torch.bfloat16, device=x.device)
        self.kernel(
            out,
            (layer.w13_weight, layer.w13_weight_scale),
            (layer.w2_weight, layer.w2_weight_scale),
            self.buffer,
            shared_l1_weights=self.shared_l1_weights,
            shared_l2_weights=self.shared_l2_weights,
            activation_clamp=self.activation_clamp,
        )
        after = torch.npu.current_stream().record_event()
        return FusedExpertsResult(
            routed_out=out,
            before_dispatch_evt=before,
            before_gmm2_evt=before,
            before_combine_evt=after,
            shared_activation_overlap_start_evt=before,
        )
