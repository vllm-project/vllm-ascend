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
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern

# computation-communication tiling block is 512
ALLREDUCE_NORM_FUSE_THRESHOLD = 512


class MatmulAllReduceAddRMSNormPattern(BasePattern):
    """
    Fuse the decode critical path
        unquantized_gemm -> all_reduce -> copy/detach -> (1.0 + w) ->
        maybe_chunk_residual -> npu_add_rms_norm_bias
    (i.e. a RowParallelLinear o_proj/out_proj followed by the Gemma-style
    fused add+RMSNorm of the next half-layer) into the single
    matmul_allreduce_add_rmsnorm operator.

    The graph structure matched here follows the traced Qwen3.5-35B-A3B
    decode graph, where:
      * torch.ops.vllm.unquantized_gemm is traced with 2 args (bias omitted)
      * the o_proj result is copied into a preallocated buffer (aten.copy +
        aten.detach from the `output[:], _ = self.o_proj(...)` patch)
      * AscendGemmaRMSNorm computes gamma = weight + 1.0 inline
      * npu_add_rms_norm_bias is called with 3 args (x, residual, gamma)
    """

    def __init__(self, vllm_config, eps=1e-6):
        super().__init__(vllm_config, eps)
        device_group = get_tp_group().device_group
        backend = device_group._get_backend(torch.device("npu"))
        self.local_rank = torch.distributed.get_rank(group=device_group)
        self.tp_group_name = backend.get_hccl_comm_name(self.local_rank)
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_inputs(self):
        batch_size, seq_len = 2, 4
        hidden_size = 2048
        x = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        weight = torch.randn(hidden_size, hidden_size, device="npu")
        residual = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        rms_norm_weight = torch.randn(hidden_size, device="npu")
        out_buf = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        return [x, weight, residual, rms_norm_weight, out_buf]

    def get_pattern(self):
        def pattern(x, weight, residual, rms_norm_weight, out_buf):
            mm = torch.ops.vllm.unquantized_gemm(x, weight)
            all_reduce_ = tensor_model_parallel_all_reduce(mm)
            cp = torch.ops.aten.copy.default(out_buf, all_reduce_)
            dt = torch.ops.aten.detach.default(cp)
            gamma = rms_norm_weight + 1.0
            chunked_residual = torch.ops.vllm.maybe_chunk_residual(dt, residual)
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(dt, chunked_residual, gamma)
            return output[0], output[2]

        return pattern

    def get_replacement(self):
        def replacement(x, weight, residual, rms_norm_weight, out_buf):
            gamma = rms_norm_weight + 1.0
            out0, out1 = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(
                x,
                weight,
                residual,
                gamma,
                self.tp_group_name,
                self.tp_size,
                self.local_rank,
                self.eps,
                True,  # is_trans_b: weight is [out, in], op computes x @ weight^T
                False,  # is_gather_add_out
            )
            return out0, out1

        return replacement


class MatmulAllReduceAddRMSNormPass(VllmInductorPass):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="allreduce_rmsnorm_fusion_pass")

        MatmulAllReduceAddRMSNormPattern(vllm_config).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        pattern_idx = 0
        for pattern_entry in self.pattern_match_passes.patterns.values():
            for p in pattern_entry:
                p_str = PatternPrettyPrinter.run(p.pattern)
                logger.debug("Pattern %d: %s", pattern_idx, p_str)
                pattern_idx += 1
        logger.debug("Replaced %s patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """Applicable to all compile ranges, including decode."""
        return True
