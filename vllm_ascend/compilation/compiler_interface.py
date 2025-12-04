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
import functools
from typing import Any, Callable, Optional

import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import (graph_returns_tuple,
                                        make_graph_return_tuple)
from torch._inductor.decomposition import select_decomp_table
from torch.fx import GraphModule
from vllm.compilation.compiler_interface import CompilerInterface


def compile_fx(graph: GraphModule, example_inputs: list,
               inner_compile: Callable, decompositions: dict) -> Callable:
    recursive_compile_fx = functools.partial(compile_fx,
                                             inner_compile=inner_compile,
                                             decompositions=decompositions)

    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(graph, example_inputs,
                                       recursive_compile_fx)
    return aot_autograd(fw_compiler=inner_compile)(graph, example_inputs)


class AscendCompiler(CompilerInterface):
    """
    AscendCompiler is a custom compiler interface for the Ascend platform.
    This class provides a method to compile a PyTorch FX graph module with
    specific configurations for graph fusion and decomposition.
    """
    name = "AscendCompiler"

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
    ) -> tuple[Optional[Callable], Optional[Any]]:

        def compile_inner(graph, example_inputs):
            current_pass_manager = compiler_config["graph_fusion_manager"]
            graph = current_pass_manager(graph, runtime_shape)
            return graph

        decompositions = select_decomp_table()

        compiled_fn = compile_fx(
            graph=graph,
            example_inputs=example_inputs,
            inner_compile=compile_inner,
            decompositions=decompositions,
        )

        return compiled_fn, None
