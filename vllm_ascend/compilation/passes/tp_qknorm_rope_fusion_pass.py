#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

# Importing the module registers torch.ops.vllm.minimax_qk_norm_fusion, which
# the pattern below calls while being traced; make the registration explicit
# instead of relying on the model being loaded first.
import vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp  # noqa: F401
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import Range
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.model_executor.layers.attention import Attention

import vllm_ascend.ops.triton.linearnorm.split_qkv_tp_rmsnorm_rope  # noqa: F401
from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.utils import get_rope_dim


class TPQKNormRopeFusionPattern(BasePattern):
    """Fuse the TP qk-norm + rope subgraph (MiniMax-style full-channel norm)
    into one Triton kernel.

    Unfused (upstream eager under TP > 1 on NPU):
        q, k = torch.ops.vllm.minimax_qk_norm_fusion(
            qkv, q_weight, k_weight, q_size, kv_size, tp_rank, tp_world, eps, None)
        _, _, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = torch.ops.vllm.npu_rotary_embedding(
            positions, q, k, cos_sin_cache, head_dim, rotary_dim, True)

    Fused:
        q, k, v = torch.ops.vllm.qkv_tp_rmsnorm_rope(
            qkv, q_weight, k_weight, q_size, kv_size, head_dim, rotary_dim,
            eps, tp_world)

    The fused op reads its cos/sin rows from the runner-maintained per-step
    slices, so the replacement consumes neither cos_sin_cache nor positions.

    Compared with the per-head ``qkv_rmsnorm_rope`` fusion (QKNormRopeFusionPass),
    this variant norms over the full q/k channel dim with a TP-global variance
    (all-reduce inside the op) and applies neox-style partial rope, matching the
    semantics of ``MiniMaxText01RMSNormTP.forward_qkv`` + ``npu_rotary_embedding``.
    """

    def __init__(
        self,
        vllm_config,
        head_dim,
        num_heads,
        num_kv_heads,
        tp_rank,
        tp_world,
        eps=1e-6,
    ):
        super().__init__(vllm_config, eps)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.rope_dim = get_rope_dim(vllm_config)

    def get_inputs(self):
        T = 5
        max_position_embeddings = 16384
        qkv = torch.empty(T, self.q_size + 2 * self.kv_size, dtype=torch.bfloat16, device="npu")
        # The TP qk-norm runs over the full q/k channel dim (not per-head),
        # so the weights are [q_size] / [kv_size] rather than [head_dim].
        q_weight = torch.empty(self.q_size, dtype=torch.bfloat16, device="npu")
        k_weight = torch.empty(self.kv_size, dtype=torch.bfloat16, device="npu")
        cos_sin_cache = torch.empty(max_position_embeddings, self.rope_dim, dtype=torch.bfloat16, device="npu")
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
            # workspace is always None on NPU (Lamport fused AR+RMS kernel is
            # CUDA-only), so bake the same constant the traced graph carries.
            q, k = torch.ops.vllm.minimax_qk_norm_fusion(
                qkv,
                q_weight,
                k_weight,
                self.q_size,
                self.kv_size,
                self.tp_rank,
                self.tp_world,
                self.eps,
                None,
            )
            _, _, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
                positions, q, k, cos_sin_cache, self.head_dim, self.rope_dim, True
            )

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
            # cos_sin_cache / positions are unused: the fused op reads the
            # per-step cos/sin slices maintained by the model runner, so no
            # per-layer gather is needed.
            results = torch.ops.vllm.qkv_tp_rmsnorm_rope(
                input=qkv,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                rotary_dim=self.rope_dim,
                eps=self.eps,
                tp_world=self.tp_world,
            )

            return results

        return replacement


class TPQKNormRopeFusionPass(VllmInductorPass):
    """
    A pass for fusing the TP qk-norm + rope ops into the fused
    split_qkv_tp_rmsnorm_rope Triton kernel, so the fused op is captured by
    torch.compile / NPUGraph instead of an eager monkey-patch.

    Applicable to models whose attention uses MiniMaxText01RMSNormTP q/k norm
    (e.g. MiniMax-M2): under TP > 1 the compiled graph routes the norm through
    torch.ops.vllm.minimax_qk_norm_fusion, which this pass fuses with the
    adjacent split / npu_rotary_embedding ops.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="tp_qknorm_rope_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16,):
            logger.debug("TP QKNorm and Rope fusion not enabled: unsupported dtype %s", dtype)
            return

        # NOTE: the gate cannot discover MiniMaxText01RMSNormTP modules via
        # get_layers_from_vllm_config: upstream calls their norm through the
        # staticmethods forward_qk / forward_qkv, so the modules are never
        # invoked via __call__ and never enter static_forward_context. Gate on
        # model_type instead and extend the tuple as more models adopt the
        # layer.
        model_type = getattr(vllm_config.model_config.hf_text_config, "model_type", "")
        if model_type not in ("minimax_m2",):
            logger.debug("TP QKNorm and Rope fusion not enabled: model_type %s", model_type)
            return

        tp_world = get_tensor_model_parallel_world_size()
        if tp_world == 1:
            # Under TP=1 upstream routes q/k norm through forward_qk (plain
            # aten ops); there is no minimax_qk_norm_fusion op in the graph.
            logger.debug("TP QKNorm and Rope fusion not enabled: tp_world == 1")
            return

        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(vllm_config, Attention)
        if len(attn_layers) == 0:
            logger.debug("TP QKNorm and Rope fusion enabled, but no Attention layers were discovered.")
            return
        layer = next(iter(attn_layers.values()))

        TPQKNormRopeFusionPattern(
            vllm_config=vllm_config,
            head_dim=layer.head_size,
            num_heads=layer.num_heads,
            num_kv_heads=layer.num_kv_heads,
            tp_rank=get_tensor_model_parallel_rank(),
            tp_world=tp_world,
            eps=getattr(vllm_config.model_config.hf_text_config, "rms_norm_eps", 1e-6),
        ).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s TP QKNorm and Rope patterns", self.matched_count)
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
