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
import logging

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.attention.layer import Attention
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import (VllmConfig, get_current_vllm_config,
                         get_layers_from_vllm_config)


class QKNormFusionPattern:

    def __init__(self, head_dim, num_heads, num_kv_heads, eps=1e-6):
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        vllm_config = get_current_vllm_config()
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
        return [qkv, q_weight, k_weight]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(qkv: torch.Tensor, q_weight: torch.Tensor,
                    k_weight: torch.Tensor):
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                               self.head_dim)
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight,
                                                       self.eps)
            q_flat = q_norm_out.view(q.shape)

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                               self.head_dim)
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight,
                                                       self.eps)
            k_flat = k_norm_out.view(k.shape)
            return q_flat, k_flat, v

        def replacement(qkv: torch.Tensor, q_weight: torch.Tensor,
                        k_weight: torch.Tensor):
            results = torch.ops.vllm.qk_rmsnorm(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.eps,
                q_bias=None,
                k_bias=None,
            )
            return results

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class QKNormFusionPatternWithBias:

    def __init__(self, head_dim, num_heads, num_kv_heads, eps=1e-6):
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        vllm_config = get_current_vllm_config()
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

        return [qkv, q_weight, k_weight, q_bias, k_bias]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(qkv: torch.Tensor, q_weight: torch.Tensor,
                    k_weight: torch.Tensor, q_bias: torch.Tensor,
                    k_bias: torch.Tensor):
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)

            q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight,
                                                       self.eps)
            q_normed = q_norm_out + q_bias
            q_flat = q_normed.view(q.shape)

            k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight,
                                                       self.eps)
            k_normed = k_norm_out + k_bias
            k_flat = k_normed.view(k.shape)

            return q_flat, k_flat, v

        def replacement(qkv: torch.Tensor, q_weight: torch.Tensor,
                        k_weight: torch.Tensor, q_bias: torch.Tensor,
                        k_bias: torch.Tensor):
            results = torch.ops.vllm.qk_rmsnorm(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.eps,
                q_bias=q_bias,
                k_bias=k_bias,
            )
            return results

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class QKNormFusionPass(VllmInductorPass):
    """
    A pass for fusing QKV split and RMSNorm operations into a single qk_rmsnorm operator.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="qknorm_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logging.info("QKNorm fusion not enabled: unsupported dtype %s",
                         dtype)
            return

        # use one attn layer to get meta (such as head_dim) for QkNormFusionPattern
        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
            vllm_config, Attention)
        if len(attn_layers) == 0:
            logging.info(
                "QK Norm fusion enabled, but no Attention layers were discovered."
            )
            return
        layer = next(iter(attn_layers.values()))
        for epsilon in [1e-6, 1e-5]:
            QKNormFusionPattern(head_dim=layer.head_size,
                                num_heads=layer.num_heads,
                                num_kv_heads=layer.num_kv_heads,
                                eps=epsilon).register(
                                    self.pattern_match_passes)

            QKNormFusionPatternWithBias(head_dim=layer.head_size,
                                        num_heads=layer.num_heads,
                                        num_kv_heads=layer.num_kv_heads,
                                        eps=epsilon).register(
                                            self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logging.debug("Fused %s QKNorm patterns", self.matched_count)
        self.end_and_log()

    def is_applicable(self, runtime_shape):
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
