#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
"""
Graph fusion pass for QKV Split + RMSNorm + RoPE on Ascend NPU.

This pass uses PyTorch's Inductor pattern matcher to find subgraphs that
perform QKV split → per-head RMSNorm → RoPE, and replaces them with a
single fused Triton kernel call (torch.ops.vllm.qkv_rmsnorm_rope).

Supported patterns:
  1. Full RoPE (rotary_dim == head_dim): uses npu_apply_rotary_pos_emb
  2. Full RoPE with bias: same as (1) but with RMSNorm bias
  3. Partial RoPE (rotary_dim < head_dim): uses rope_forward_triton
  4. Partial RoPE with bias: same as (3) but with RMSNorm bias
"""
import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter
from vllm.attention.layer import Attention
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import Range
from vllm.logger import logger


# =============================================================================
# Pattern 1: Full RoPE via npu_apply_rotary_pos_emb (no bias)
# =============================================================================
class QKNormRopeFusionPattern:
    """Matches QKV split + RMSNorm + full RoPE (npu_apply_rotary_pos_emb)."""

    def __init__(self, vllm_config, head_dim, num_heads, num_kv_heads, eps=1e-6):
        self.vllm_config = vllm_config
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.device = vllm_config.device_config.device if vllm_config.device_config else None

    def get_inputs(self):
        T = 5
        qkv = torch.empty(T, self.q_size + 2 * self.kv_size, dtype=torch.bfloat16, device="npu")
        q_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        # cos/sin for full RoPE: [1, T, 1, head_dim] (duplicated HF format)
        cos = torch.empty(1, T, 1, self.head_dim, dtype=torch.bfloat16, device="npu")
        sin = torch.empty(1, T, 1, self.head_dim, dtype=torch.bfloat16, device="npu")
        return [qkv, q_weight, k_weight, cos, sin]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
                qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
        ):
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight, self.eps)

            # BUG FIX: original code used self.kv_hidden_size (undefined), should be self.head_dim
            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight, self.eps)

            q_flat = q_norm_out.view(q.shape)
            q_reshape = q_flat.contiguous().view(1, q_flat.shape[0], -1, self.head_dim)

            k_flat = k_norm_out.view(k.shape)
            k_reshape = k_flat.contiguous().view(1, k_flat.shape[0], -1, self.head_dim)

            q_rope, k_rope = torch.ops.npu.npu_apply_rotary_pos_emb(q_reshape, k_reshape, cos, sin)

            return q_rope, k_rope, v

        def replacement(
                qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
        ):
            results = torch.ops.vllm.qkv_rmsnorm_rope(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                rotary_dim=self.head_dim,  # full RoPE: rotary_dim == head_dim
                eps=self.eps,
                q_bias=None,
                k_bias=None,
                sin=sin,
                cos=cos,
            )
            return results

        pm.register_replacement(pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)


# =============================================================================
# Pattern 2: Full RoPE via npu_apply_rotary_pos_emb (with bias)
# =============================================================================
class QKNormRopeFusionPatternWithBias:
    """Matches QKV split + RMSNorm (with bias) + full RoPE."""

    def __init__(self, vllm_config, head_dim, num_heads, num_kv_heads, eps=1e-6):
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.vllm_config = vllm_config
        self.device = vllm_config.device_config.device if vllm_config.device_config else None

    def get_inputs(self):
        T = 5
        qkv = torch.empty(T, self.q_size + 2 * self.kv_size, dtype=torch.bfloat16, device="npu")
        q_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        q_bias = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_bias = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        cos = torch.empty(1, T, 1, self.head_dim, dtype=torch.bfloat16, device="npu")
        sin = torch.empty(1, T, 1, self.head_dim, dtype=torch.bfloat16, device="npu")
        return [qkv, q_weight, k_weight, q_bias, k_bias, cos, sin]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
                qkv: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                q_bias: torch.Tensor,
                k_bias: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
        ):
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight, self.eps)
            q_normed = q_norm_out + q_bias

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight, self.eps)
            k_normed = k_norm_out + k_bias

            q_flat = q_normed.view(q.shape)
            q_reshape = q_flat.contiguous().view(1, q_flat.shape[0], -1, self.head_dim)

            k_flat = k_normed.view(k.shape)
            k_reshape = k_flat.contiguous().view(1, k_flat.shape[0], -1, self.head_dim)

            q_rope, k_rope = torch.ops.npu.npu_apply_rotary_pos_emb(q_reshape, k_reshape, cos, sin)

            return q_rope, k_rope, v

        def replacement(
                qkv: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                q_bias: torch.Tensor,
                k_bias: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
        ):
            results = torch.ops.vllm.qkv_rmsnorm_rope(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                rotary_dim=self.head_dim,  # full RoPE
                eps=self.eps,
                q_bias=q_bias,
                k_bias=k_bias,
                cos=cos,
                sin=sin,
            )
            return results

        pm.register_replacement(pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)


# =============================================================================
# Pattern 3: Partial RoPE via rope_forward_triton (no bias)
# =============================================================================
class QKNormPartialRopeFusionPattern:
    """
    Matches QKV split + RMSNorm + partial RoPE (rope_forward_triton).

    This pattern handles models where rotary_dim < head_dim (e.g., Qwen3-Next
    with partial_rotary_factor=0.5). The RoPE is applied only to the first
    rotary_dim dimensions of each head.
    """

    def __init__(self, vllm_config, head_dim, num_heads, num_kv_heads, rotary_dim, eps=1e-6):
        self.vllm_config = vllm_config
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.device = vllm_config.device_config.device if vllm_config.device_config else None

    def get_inputs(self):
        T = 5
        qkv = torch.empty(T, self.q_size + 2 * self.kv_size, dtype=torch.bfloat16, device="npu")
        q_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        # cos/sin for partial RoPE: [1, T, 1, rotary_dim] (not head_dim!)
        # This matches the actual cos/sin shape from set_cos_and_sin() for
        # models with partial_rotary_factor.
        cos = torch.empty(1, T, 1, self.rotary_dim, dtype=torch.bfloat16, device="npu")
        sin = torch.empty(1, T, 1, self.rotary_dim, dtype=torch.bfloat16, device="npu")
        return [qkv, q_weight, k_weight, cos, sin]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
                qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
        ):
            # QKV split
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # Per-head RMSNorm on Q
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight, self.eps)

            # Per-head RMSNorm on K
            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight, self.eps)

            # Flatten back and reshape for rope_forward_triton
            q_flat = q_norm_out.view(q.shape)
            k_flat = k_norm_out.view(k.shape)

            # Reshape cos/sin for rope_forward_triton
            cos = cos.view(-1, self.rotary_dim)
            sin = sin.view(-1, self.rotary_dim)

            # Reshape to [tokens, heads, head_dim] for rope
            q = q_flat.contiguous().view(q_flat.shape[0], -1, self.head_dim)
            k = k_flat.contiguous().view(k_flat.shape[0], -1, self.head_dim)

            # Partial RoPE via triton kernel
            query, key = torch.ops.vllm.rope_forward_triton(
                q, k, cos, sin, rope_dim=self.rotary_dim, is_neox_style=True
            )

            return query.view(q.shape), key.view(k.shape), v

        def replacement(
                qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
        ):
            results = torch.ops.vllm.qkv_rmsnorm_rope(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                rotary_dim=self.rotary_dim,  # partial RoPE: rotary_dim < head_dim
                eps=self.eps,
                q_bias=None,
                k_bias=None,
                sin=sin,
                cos=cos,
            )
            return results

        pm.register_replacement(pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)


# =============================================================================
# Pattern 4: Partial RoPE via rope_forward_triton (with bias)
# =============================================================================
class QKNormPartialRopeFusionPatternWithBias:
    """Matches QKV split + RMSNorm (with bias) + partial RoPE."""

    def __init__(self, vllm_config, head_dim, num_heads, num_kv_heads, rotary_dim, eps=1e-6):
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.vllm_config = vllm_config
        self.device = vllm_config.device_config.device if vllm_config.device_config else None

    def get_inputs(self):
        T = 5
        qkv = torch.empty(T, self.q_size + 2 * self.kv_size, dtype=torch.bfloat16, device="npu")
        q_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_weight = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        q_bias = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_bias = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        cos = torch.empty(1, T, 1, self.rotary_dim, dtype=torch.bfloat16, device="npu")
        sin = torch.empty(1, T, 1, self.rotary_dim, dtype=torch.bfloat16, device="npu")
        return [qkv, q_weight, k_weight, q_bias, k_bias, cos, sin]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
                qkv: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                q_bias: torch.Tensor,
                k_bias: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
        ):
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight, self.eps)
            q_normed = q_norm_out + q_bias

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight, self.eps)
            k_normed = k_norm_out + k_bias

            q_flat = q_normed.view(q.shape)
            k_flat = k_normed.view(k.shape)

            cos = cos.view(-1, self.rotary_dim)
            sin = sin.view(-1, self.rotary_dim)

            q = q_flat.contiguous().view(q_flat.shape[0], -1, self.head_dim)
            k = k_flat.contiguous().view(k_flat.shape[0], -1, self.head_dim)

            query, key = torch.ops.vllm.rope_forward_triton(
                q, k, cos, sin, rope_dim=self.rotary_dim, is_neox_style=True
            )

            return query.view(q.shape), key.view(k.shape), v

        def replacement(
                qkv: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                q_bias: torch.Tensor,
                k_bias: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
        ):
            results = torch.ops.vllm.qkv_rmsnorm_rope(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                rotary_dim=self.rotary_dim,
                eps=self.eps,
                q_bias=q_bias,
                k_bias=k_bias,
                cos=cos,
                sin=sin,
            )
            return results

        pm.register_replacement(pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)


# =============================================================================
# Fusion Pass
# =============================================================================
class QKNormRopeFusionPass(VllmInductorPass):
    """
    Inductor pass that fuses QKV split + RMSNorm + RoPE into a single
    Triton kernel (torch.ops.vllm.qkv_rmsnorm_rope).

    Handles both full RoPE (via npu_apply_rotary_pos_emb) and partial RoPE
    (via rope_forward_triton) patterns, with and without RMSNorm bias.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="qknorm_rope_fusion_pass"
        )

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug(
                "QKNorm and Rope fusion not enabled: unsupported dtype %s", dtype
            )
            return

        # Get attention layer metadata (head_dim, num_heads, etc.)
        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
            vllm_config, Attention
        )
        if len(attn_layers) == 0:
            logger.debug(
                "QKNorm and Rope fusion enabled, but no Attention layers were discovered."
            )
            return

        layer = next(iter(attn_layers.values()))

        # Determine rotary_dim from model config (for partial RoPE support)
        model_config = vllm_config.model_config
        rotary_dim = layer.head_size  # default: full RoPE
        if hasattr(model_config.hf_text_config, "partial_rotary_factor"):
            factor = model_config.hf_text_config.partial_rotary_factor
            computed = int(layer.head_size * factor)
            if computed < layer.head_size:
                rotary_dim = computed
                logger.debug(
                    "Partial RoPE detected: head_dim=%d, rotary_dim=%d (factor=%.2f)",
                    layer.head_size, rotary_dim, factor,
                )

        for epsilon in [1e-6, 1e-5]:
            if layer.head_size != 128:
                logger.debug(
                    "QKNorm and Rope fusion not enabled: head_dim %d is not 128",
                    layer.head_size,
                )
                continue

            # Register full RoPE patterns (npu_apply_rotary_pos_emb)
            QKNormRopeFusionPattern(
                vllm_config=vllm_config,
                head_dim=layer.head_size,
                num_heads=layer.num_heads,
                num_kv_heads=layer.num_kv_heads,
                eps=epsilon,
            ).register(self.pattern_match_passes)

            QKNormRopeFusionPatternWithBias(
                vllm_config=vllm_config,
                head_dim=layer.head_size,
                num_heads=layer.num_heads,
                num_kv_heads=layer.num_kv_heads,
                eps=epsilon,
            ).register(self.pattern_match_passes)

            # Register partial RoPE patterns (rope_forward_triton)
            # only when the model actually uses partial RoPE
            if rotary_dim < layer.head_size:
                QKNormPartialRopeFusionPattern(
                    vllm_config=vllm_config,
                    head_dim=layer.head_size,
                    num_heads=layer.num_heads,
                    num_kv_heads=layer.num_kv_heads,
                    rotary_dim=rotary_dim,
                    eps=epsilon,
                ).register(self.pattern_match_passes)

                QKNormPartialRopeFusionPatternWithBias(
                    vllm_config=vllm_config,
                    head_dim=layer.head_size,
                    num_heads=layer.num_heads,
                    num_kv_heads=layer.num_kv_heads,
                    rotary_dim=rotary_dim,
                    eps=epsilon,
                ).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s QKNorm and Rope patterns", self.matched_count)
        logger.debug("Patterns registered for replacement:")
        pattern_idx = 0
        for pattern_entry in self.pattern_match_passes.patterns.values():
            for p in pattern_entry:
                p_str = PatternPrettyPrinter.run(p.pattern)
                logger.debug("Pattern %d: %s", pattern_idx, p_str)
                pattern_idx += 1
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
