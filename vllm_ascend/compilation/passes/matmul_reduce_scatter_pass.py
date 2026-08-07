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

from __future__ import annotations

import torch
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger


class MatmulReduceScatterFusionPass(VllmInductorPass):
    """
    Route sequence-parallel row-parallel matmul + reduce-scatter through the
    fused NPU primitive.

    The generic ``matmul_and_reduce`` op takes a layer name string and resolves
    weights from the static forward context at runtime, so a tensor-only
    pattern matcher cannot recover the matmul operands. This pass keeps the
    enablement in graph compilation by replacing that generic op with the
    fused custom op; the fused op still falls back to the generic path for
    unsupported shapes, world sizes, or quantization methods.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.enabled = vllm_config.model_config.dtype in (torch.float16, torch.bfloat16)
        if not self.enabled:
            logger.debug(
                "Matmul reduce-scatter fusion not enabled: unsupported dtype %s",
                vllm_config.model_config.dtype,
            )

    def __call__(self, graph: torch.fx.Graph) -> None:  # type: ignore[override]
        if not self.enabled:
            return

        self.begin()
        self.matched_count = 0
        matmul_and_reduce_targets = {
            torch.ops.vllm.matmul_and_reduce,
            torch.ops.vllm.matmul_and_reduce.default,
        }
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in matmul_and_reduce_targets:
                continue
            with graph.inserting_after(node):
                replacement = graph.call_function(
                    torch.ops.vllm.matmul_reduce_scatter.default,
                    args=node.args,
                    kwargs=node.kwargs,
                )
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)
            self.matched_count += 1

        logger.debug("Fused %s matmul_reduce_scatter patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
