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

from torch import fx as fx
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig


class GraphRewritePassManager:
    """
    A pass manager for graph rewriting passes.
    It handles the configuration and execution of passes.
    The counterpart in vllm is PostGradPassManager. Since torch_npu does not
    support inductor and triton for now, we choose to adopt the graph rewriter on
    fx graph rather than the inductor pass manager.
    """

    def __init__(self):
        self.passes: list[VllmInductorPass] = []

    def __call__(self, graph: fx.Graph, **kwargs) -> fx.Graph:
        for pass_ in self.passes:
            if pass_.is_applicable(**kwargs):
                pass_(graph)
        graph.recompile()
        return graph

    def add(self, pass_: VllmInductorPass):
        assert isinstance(pass_, VllmInductorPass)
        self.passes.append(pass_)
        
    def configure(self, config: VllmConfig):
        # By default, we enable the graph rewriter and quantization fusion pass.
        self.ascend_compilation_config: dict = config.additional_config.get(
            "ascend_compilation_config", {})
        if self.ascend_compilation_config.get("enable_quantization_fusion",
                                              True):
            from .quant_fusion_pass import AscendQuantFusionPass
            self.passes.append(AscendQuantFusionPass(config))
        # Add more passes here as needed