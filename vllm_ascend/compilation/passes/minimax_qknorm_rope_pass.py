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
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import Range
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import logger
from vllm.model_executor.layers.attention import Attention

from vllm_ascend.compilation.passes.base_pattern import BasePattern


class MiniMaxCrossHeadQKNormRopePattern(BasePattern):
    """
    Matches FX graph pattern for MiniMax cross-head QKNorm + RoPE (TP=1 case).

    Pattern:
      q, k, v = split(qkv, [q_size, kv_size, kv_size])
      q = q.contiguous()
      k = k.contiguous()
      q_normed, q_inv = npu_rms_norm(q, q_weight, eps)
      k_normed, k_inv = npu_rms_norm(k, k_weight, eps)
      q, k = npu_rotary_embedding(pos, q_normed, k_normed, cache, ...)

    Replacement:
      q, k, v, qk_var = minimax_qkv_crosshead_norm_rope(...)
    """

    def __init__(self, vllm_config, head_dim, num_heads, num_kv_heads,
                 rotary_dim, eps=1e-6):
        super().__init__(vllm_config, eps)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.rotary_dim = rotary_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

    def get_inputs(self):
        T = 5
        max_position_embeddings = 16384
        qkv = torch.empty(T, self.q_size + 2 * self.kv_size,
                          dtype=torch.bfloat16, device="npu")
        q_weight = torch.empty(self.q_size, dtype=torch.bfloat16,
                               device="npu")
        k_weight = torch.empty(self.kv_size, dtype=torch.bfloat16,
                               device="npu")
        cos_sin_cache = torch.empty(max_position_embeddings, self.rotary_dim,
                                    dtype=torch.bfloat16, device="npu")
        positions = torch.ones(T, dtype=torch.int64, device="npu")
        return [qkv, q_weight, k_weight, cos_sin_cache, positions]

    def get_pattern(self):
        def pattern(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = q.contiguous()
            k = k.contiguous()

            # Cross-head RMSNorm (flat, not reshaped per-head)
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(
                q, q_weight, self.eps)
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(
                k, k_weight, self.eps)

            q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
                positions, q_norm_out, k_norm_out, cos_sin_cache,
                self.head_dim, self.rotary_dim, True)

            return q_rope, k_rope, v

        return pattern

    def get_replacement(self):
        def replacement(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            q, k, v, qk_var = \
                torch.ops.vllm.minimax_qkv_crosshead_norm_rope(
                    qkv=qkv,
                    cos_sin_cache=cos_sin_cache,
                    positions=positions,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    q_hidden_size=self.q_size,
                    kv_hidden_size=self.kv_size,
                    head_dim=self.head_dim,
                    eps=self.eps,
                    rotary_dim=self.rotary_dim,
                )
            return q, k, v

        return replacement


class MiniMaxCrossHeadQKNormRopeTPPattern(BasePattern):
    """
    Matches FX graph pattern for MiniMax cross-head QKNorm + RoPE with TP
    correction.

    Pattern:
      q, k, v = split(qkv, [q_size, kv_size, kv_size])
      q = q.contiguous()
      k = k.contiguous()
      q_normed, q_inv_rms = npu_rms_norm(q, q_weight, eps)
      k_normed, k_inv_rms = npu_rms_norm(k, k_weight, eps)
      # TP correction math:
      q_local_inv_rms = q_inv_rms.to(float32)
      q_local_var = (q_local_inv_rms.reciprocal().pow(2) - eps).clamp_min_(0)
      k_local_inv_rms = k_inv_rms.to(float32)
      k_local_var = (k_local_inv_rms.reciprocal().pow(2) - eps).clamp_min_(0)
      qk_var = cat([q_local_var, k_local_var])
      qk_var = all_reduce(qk_var) / tp_world
      q_global_var, k_global_var = qk_var.chunk(2)
      q_correction = rsqrt(q_global_var + eps) / rsqrt(q_local_var + eps)
      k_correction = rsqrt(k_global_var + eps) / rsqrt(k_local_var + eps)
      q = q_normed * q_correction.to(dtype)
      k = k_normed * k_correction.to(dtype)
      q, k = npu_rotary_embedding(pos, q, k, cache, ...)

    Replacement:
      q, k, v, qk_var = minimax_qkv_crosshead_norm_rope(...)
      # qk_var is packed [batch, 2] (col 0 = q_var, col 1 = k_var)
      # No cat needed -- directly all_reduce
      qk_reduced = all_reduce(qk_var) / tp_world
      q = q * (rsqrt(qk_reduced[:,0] + eps) / rsqrt(qk_var[:,0] + eps)).to(dtype)
      k = k * (rsqrt(qk_reduced[:,1] + eps) / rsqrt(qk_var[:,1] + eps)).to(dtype)
    """

    def __init__(self, vllm_config, head_dim, num_heads, num_kv_heads,
                 rotary_dim, tp_world, eps=1e-6):
        super().__init__(vllm_config, eps)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.rotary_dim = rotary_dim
        self.tp_world = tp_world
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

    def get_inputs(self):
        T = 5
        max_position_embeddings = 16384
        qkv = torch.empty(T, self.q_size + 2 * self.kv_size,
                          dtype=torch.bfloat16, device="npu")
        q_weight = torch.empty(self.q_size, dtype=torch.bfloat16,
                               device="npu")
        k_weight = torch.empty(self.kv_size, dtype=torch.bfloat16,
                               device="npu")
        cos_sin_cache = torch.empty(max_position_embeddings, self.rotary_dim,
                                    dtype=torch.bfloat16, device="npu")
        positions = torch.ones(T, dtype=torch.int64, device="npu")
        return [qkv, q_weight, k_weight, cos_sin_cache, positions]

    def get_pattern(self):
        tp_world = self.tp_world
        eps = self.eps

        def pattern(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = q.contiguous()
            k = k.contiguous()

            q_norm_out, q_inv_rms = torch.ops.npu.npu_rms_norm(
                q, q_weight, eps)
            k_norm_out, k_inv_rms = torch.ops.npu.npu_rms_norm(
                k, k_weight, eps)

            # TP correction
            q_local_inv_rms = q_inv_rms.to(torch.float32)
            q_local_var = (
                q_local_inv_rms.reciprocal().pow(2) - eps
            ).clamp_min_(0.0)

            k_local_inv_rms = k_inv_rms.to(torch.float32)
            k_local_var = (
                k_local_inv_rms.reciprocal().pow(2) - eps
            ).clamp_min_(0.0)

            qk_var = torch.cat([q_local_var, k_local_var], dim=-1)
            qk_var = tensor_model_parallel_all_reduce(qk_var) / tp_world
            q_global_var, k_global_var = qk_var.chunk(2, dim=-1)

            q_local_rstd = torch.rsqrt(q_local_var + eps)
            k_local_rstd = torch.rsqrt(k_local_var + eps)
            q_global_rstd = torch.rsqrt(q_global_var + eps)
            k_global_rstd = torch.rsqrt(k_global_var + eps)

            q_corrected = q_norm_out * (
                q_global_rstd / q_local_rstd).to(q_norm_out.dtype)
            k_corrected = k_norm_out * (
                k_global_rstd / k_local_rstd).to(k_norm_out.dtype)

            q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
                positions, q_corrected, k_corrected, cos_sin_cache,
                self.head_dim, self.rotary_dim, True)

            return q_rope, k_rope, v

        return pattern

    def get_replacement(self):
        tp_world = self.tp_world
        eps = self.eps

        def replacement(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            q, k, v, qk_var = \
                torch.ops.vllm.minimax_qkv_crosshead_norm_rope(
                    qkv=qkv,
                    cos_sin_cache=cos_sin_cache,
                    positions=positions,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    q_hidden_size=self.q_size,
                    kv_hidden_size=self.kv_size,
                    head_dim=self.head_dim,
                    eps=eps,
                    rotary_dim=self.rotary_dim,
                )

            # TP correction after fused kernel
            # (valid because RoPE is linear: RoPE(q*s) = s*RoPE(q))
            # qk_var is already packed [batch, 2], no cat needed
            # all_reduce is in-place: must clone before it overwrites local vars
            qk_reduced = tensor_model_parallel_all_reduce(qk_var.clone()) / tp_world
            q_local_var = qk_var[:, :1]
            k_local_var = qk_var[:, 1:]
            q_global_var = qk_reduced[:, :1]
            k_global_var = qk_reduced[:, 1:]

            q_correction = (
                torch.rsqrt(q_global_var + eps)
                / torch.rsqrt(q_local_var + eps)
            ).to(q.dtype)
            k_correction = (
                torch.rsqrt(k_global_var + eps)
                / torch.rsqrt(k_local_var + eps)
            ).to(k.dtype)

            q = q * q_correction
            k = k * k_correction

            return q, k, v

        return replacement

    def register(self, pm_pass):
        # Override to include tp_world in pattern_id for dedup
        import torch._inductor.pattern_matcher as pm
        import torchair

        from vllm_ascend.compilation.passes.base_pattern import (
            _registered_patterns,
        )

        pattern_id = (
            f"{self.__class__.__name__}_{self.eps}_{self.tp_world}"
        )
        if pattern_id in _registered_patterns:
            return

        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        pm.register_replacement(
            pattern_fn, replacement_fn, example_inputs,
            pm.fwd_only, pm_pass)

        torchair.register_replacement(
            search_fn=pattern_fn,
            replace_fn=replacement_fn,
            example_inputs=example_inputs,
            extra_check=self.get_extra_stream_scope_check(),
        )

        _registered_patterns.add(pattern_id)


class MiniMaxQKNormRopeFusionPass(VllmInductorPass):
    """
    Fuses MiniMax cross-head QKNorm + RoPE into a single Triton kernel.

    This pass targets MiniMax-M2/M2.5 models which use cross-head RMSNorm
    (normalizing across ALL heads jointly) as opposed to per-head RMSNorm
    used by models like Qwen3-Next.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="minimax_qknorm_rope_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16,):
            logger.debug(
                "MiniMax QKNorm+Rope fusion not enabled: "
                "unsupported dtype %s", dtype)
            return

        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
            vllm_config, Attention)
        if len(attn_layers) == 0:
            logger.debug(
                "MiniMax QKNorm+Rope fusion enabled, "
                "but no Attention layers found.")
            return

        layer = next(iter(attn_layers.values()))
        head_dim = layer.head_size
        num_heads = layer.num_heads
        num_kv_heads = layer.num_kv_heads

        # Get rotary_dim from model config
        hf_config = vllm_config.model_config.hf_config
        rotary_dim = getattr(hf_config, "rotary_dim", head_dim)

        # Get TP world size
        from vllm.distributed import get_tensor_model_parallel_world_size
        tp_world = get_tensor_model_parallel_world_size()

        for epsilon in [1e-6, 1e-5]:
            # TP=1 pattern (no correction needed)
            MiniMaxCrossHeadQKNormRopePattern(
                vllm_config=vllm_config,
                head_dim=head_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                rotary_dim=rotary_dim,
                eps=epsilon,
            ).register(self.pattern_match_passes)

            # TP>1 pattern (with correction)
            if tp_world > 1:
                MiniMaxCrossHeadQKNormRopeTPPattern(
                    vllm_config=vllm_config,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    rotary_dim=rotary_dim,
                    tp_world=tp_world,
                    eps=epsilon,
                ).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug(
            "Fused %s MiniMax CrossHead QKNorm+Rope patterns",
            self.matched_count)
        if self.matched_count > 0:
            logger.debug("Patterns registered for replacement:")
            pattern_idx = 0
            for pattern_entry in self.pattern_match_passes.patterns.values():
                for p in pattern_entry:
                    p_str = PatternPrettyPrinter.run(p.pattern)
                    logger.debug("Pattern %d: %s", pattern_idx, p_str)
                    pattern_idx += 1
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
