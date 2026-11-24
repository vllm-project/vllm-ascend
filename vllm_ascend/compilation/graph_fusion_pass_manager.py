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
from vllm.compilation.passes.inductor_pass import get_pass_context
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


class GraphFusionPassManager:
    """
    A pass manager for graph fusion passes.
    It handles the configuration and execution of passes.
    The counterpart in vllm is PostGradPassManager. Since torch_npu
    does not support triton for now, we define our own pass manager.
    """

    def __init__(self):
        self.passes: list[VllmInductorPass] = []

    def __call__(self, graph: fx.Graph) -> fx.Graph:
        compile_range = get_pass_context().compile_range

        for pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)
        graph.recompile()
        return graph

    def add(self, pass_: VllmInductorPass):
        assert isinstance(pass_, VllmInductorPass)
        self.passes.append(pass_)

    def configure(self, config: VllmConfig):
        from vllm_ascend.ascend_config import get_ascend_config

        # Consume the recursively validated config rather than re-reading the
        # raw additional_config dict (where e.g. "false" is truthy).
        self.ascend_compilation_config = get_ascend_config().ascend_compilation_config
        profile = get_current_hardware_profile()
        if self.ascend_compilation_config.fuse_norm_quant and profile.supports(
            HardwareCapability.GRAPH_NORM_QUANT_FUSION
        ):
            from .passes.norm_quant_fusion_pass import AddRMSNormQuantFusionPass

            self.passes.append(AddRMSNormQuantFusionPass(config))

        if self.ascend_compilation_config.fuse_qknorm_rope:
            from .passes.qknorm_rope_fusion_pass import QKNormRopeFusionPass

            self.passes.append(QKNormRopeFusionPass(config))

        if self.ascend_compilation_config.fuse_muls_add and profile.supports(HardwareCapability.GRAPH_MULS_ADD_FUSION):
            from .passes.muls_add_pass import MulsAddFusionPass

            self.passes.append(MulsAddFusionPass(config))
