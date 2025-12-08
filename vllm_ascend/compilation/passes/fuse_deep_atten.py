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
import re
import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.distributed.parallel_state import get_tp_group

import logging

class FuseDeepAttnPattern:

    def __init__(self, vllm_config, eps=1e-6):
        self.vllm_config = vllm_config
        self.eps = eps
        device_group = get_tp_group().device_group
        self.local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.tp_group_name = backend.get_hccl_comm_name(self.local_rank)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        x = torch.randn(2, 4, device="npu")
        weight = torch.randn(8, 4, device="npu")
        residual = torch.randn(2, 8, device="npu")
        rms_norm_weight = torch.randn(8, device="npu")
        return [x, weight, residual, rms_norm_weight]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(x, weight, residual, rms_norm_weight):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            tmp = torch.ops.npu.npu_linear(x, weight)
            all_reduce_ = torch.distributed.all_reduce(tmp)
            output = torch.ops.npu.npu_add_rms_norm(all_reduce_, residual,
                                                    rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]

            return out0, out1

        def replacement(x, weight, residual, rms_norm_weight):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            out0, out1 = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(x, weight, residual, rms_norm_weight,
                            self.tp_group_name, 0, 0, self.eps, True, True)
            return out0, out1

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class FuseDeepAttnPass(VllmInductorPass):
    """
    A pass for fusing AddRMSNorm and W8A8 quantization operations on Ascend.
    """


    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="fuse_deep_atten_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logging.info("Quant fusion not enabled: unsupported dtype %s",
                         dtype)
            return

        common_epsilons = [1e-5, 1e-6]
        for eps in common_epsilons:
            FuseDeepAttnPattern(vllm_config,
                                   eps=eps).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        logging.warning("=========before torch compile capture graph========")
        logging.warning(f"graph {graph.graph}")
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logging.warning("Replaced %s patterns", self.matched_count)
        self.end_and_log()

        logging.warning("=========after torch compile capture graph========")
        logging.warning(f"graph {graph.graph}")


    def is_applicable(self, runtime_shape):
        """
        Check if the pass is applicable for the current configuration.
        """
        return True