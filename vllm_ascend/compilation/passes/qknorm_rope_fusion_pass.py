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
import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import (PatternMatcherPass,
                                             PatternPrettyPrinter)
from vllm.attention.layer import Attention
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import Range
from vllm.logger import logger


class QKNormRopeFusionPattern:

    def __init__(self,
                 vllm_config,
                 head_dim,
                 num_heads,
                 num_kv_heads,
                 eps=1e-6):
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
        qkv = torch.empty(T,
                          self.q_size + 2 * self.kv_size,
                          dtype=torch.bfloat16,
                          device="npu")
        q_weight = torch.empty(self.head_dim,
                               dtype=torch.bfloat16,
                               device="npu")
        k_weight = torch.empty(self.head_dim,
                               dtype=torch.bfloat16,
                               device="npu")
        cos = torch.empty(1,
                          T,
                          1,
                          self.head_dim,
                          dtype=torch.bfloat16,
                          device="npu")
        sin = torch.empty(1,
                          T,
                          1,
                          self.head_dim,
                          dtype=torch.bfloat16,
                          device="npu")
        return [qkv, q_weight, k_weight, cos, sin]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(qkv: torch.Tensor, q_weight: torch.Tensor,
                    k_weight: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor):

            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)

            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                               self.head_dim)
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight,
                                                       self.eps)

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                               self.head_dim)
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight,
                                                       self.eps)

            q_flat = q_norm_out.view(q.shape)
            q_reshape = q_flat.contiguous().view(1, q_flat.shape[0], -1,
                                                 self.head_dim)

            k_flat = k_norm_out.view(k.shape)
            k_reshape = k_flat.contiguous().view(1, k_flat.shape[0], -1,
                                                 self.head_dim)

            q_rope, k_rope = torch.ops.npu.npu_apply_rotary_pos_emb(
                q_reshape, k_reshape, cos, sin)

            return q_rope, k_rope, v

        def replacement(qkv: torch.Tensor, q_weight: torch.Tensor,
                        k_weight: torch.Tensor, cos: torch.Tensor,
                        sin: torch.Tensor):
            results = torch.ops.vllm.qkv_rmsnorm_rope(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.eps,
                q_bias=None,
                k_bias=None,
                sin=sin,
                cos=cos)

            return results

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class QKNormRopeFusionPatternWithBias:

    def __init__(self,
                 vllm_config,
                 head_dim,
                 num_heads,
                 num_kv_heads,
                 eps=1e-6):
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
        qkv = torch.empty(T,
                          self.q_size + 2 * self.kv_size,
                          dtype=torch.bfloat16,
                          device="npu")
        q_weight = torch.empty(self.head_dim,
                               dtype=torch.bfloat16,
                               device="npu")
        k_weight = torch.empty(self.head_dim,
                               dtype=torch.bfloat16,
                               device="npu")
        q_bias = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        k_bias = torch.empty(self.head_dim, dtype=torch.bfloat16, device="npu")
        cos = torch.empty(1,
                          T,
                          1,
                          self.head_dim,
                          dtype=torch.bfloat16,
                          device="npu")
        sin = torch.empty(1,
                          T,
                          1,
                          self.head_dim,
                          dtype=torch.bfloat16,
                          device="npu")

        return [qkv, q_weight, k_weight, q_bias, k_bias, cos, sin]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(qkv: torch.Tensor, q_weight: torch.Tensor,
                    k_weight: torch.Tensor, q_bias: torch.Tensor,
                    k_bias: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor):
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)

            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                               self.head_dim)
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight,
                                                       self.eps)
            q_normed = q_norm_out + q_bias

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                               self.head_dim)
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight,
                                                       self.eps)
            k_normed = k_norm_out + k_bias

            q_flat = q_normed.view(q.shape)
            q_reshape = q_flat.contiguous().view(1, q_flat.shape[0], -1,
                                                 self.head_dim)

            k_flat = k_normed.view(k.shape)
            k_reshape = k_flat.contiguous().view(1, k_flat.shape[0], -1,
                                                 self.head_dim)

            q_rope, k_rope = torch.ops.npu.npu_apply_rotary_pos_emb(
                q_reshape, k_reshape, cos, sin)

            return q_rope, k_rope, v

        def replacement(qkv: torch.Tensor, q_weight: torch.Tensor,
                        k_weight: torch.Tensor, q_bias: torch.Tensor,
                        k_bias: torch.Tensor, cos: torch.Tensor,
                        sin: torch.Tensor):
            results = torch.ops.vllm.qkv_rmsnorm_rope(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.eps,
                q_bias=q_bias,
                k_bias=k_bias,
                cos=cos,
                sin=sin)
            return results

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class QKNormRopeFusionPass(VllmInductorPass):
    """
    A pass for fusing QKV split and RMSNorm operations into a single qk_rmsnorm operator.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="qknorm_rope_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug(
                "QKNorm and Rope fusion not enabled: unsupported dtype %s",
                dtype)
            return

        # use one attn layer to get meta (such as head_dim) for QKNormRopeFusionPattern
        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
            vllm_config, Attention)
        if len(attn_layers) == 0:
            logger.debug(
                "QKNorm and Rope fusion enabled, but no Attention layers were discovered."
            )
            return
        layer = next(iter(attn_layers.values()))
        for epsilon in [1e-6, 1e-5]:
            if layer.head_size != 128:
                logger.debug(
                    "QKNorm and Rope fusion not enabled: head_dim %d is not equal of 128",
                    layer.head_size)
                continue
            QKNormRopeFusionPattern(vllm_config=vllm_config,
                                    head_dim=layer.head_size,
                                    num_heads=layer.num_heads,
                                    num_kv_heads=layer.num_kv_heads,
                                    eps=epsilon).register(
                                        self.pattern_match_passes)

            QKNormRopeFusionPatternWithBias(vllm_config=vllm_config,
                                            head_dim=layer.head_size,
                                            num_heads=layer.num_heads,
                                            num_kv_heads=layer.num_kv_heads,
                                            eps=epsilon).register(
                                                self.pattern_match_passes)

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
