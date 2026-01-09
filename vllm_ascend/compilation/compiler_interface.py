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

import torch
import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import (graph_returns_tuple,
                                        make_graph_return_tuple)
from torch._inductor.decomposition import select_decomp_table
from torch.fx import GraphModule
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config.utils import Range

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import COMPILATION_PASS_KEY


def compile_fx(graph: GraphModule, example_inputs: list,
               inner_compile: Callable, decompositions: dict) -> Callable:
    recursive_compile_fx = functools.partial(compile_fx,
                                             inner_compile=inner_compile,
                                             decompositions=decompositions)

    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(graph, example_inputs,
                                       recursive_compile_fx)
    return aot_autograd(fw_compiler=inner_compile)(graph, example_inputs)


def fusion_pass_compile(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    compile_range: Range,
    key: Optional[str] = None,
) -> tuple[Optional[Callable], Optional[Any]]:

    def compile_inner(graph, example_inputs):
        current_pass_manager = compiler_config[COMPILATION_PASS_KEY]
        graph = current_pass_manager(graph)
        return graph

    decompositions = select_decomp_table()

    compiled_fn = compile_fx(
        graph=graph,
        example_inputs=example_inputs,
        inner_compile=compile_inner,
        decompositions=decompositions,
    )

    return compiled_fn, None


def npugraph_ex_compile(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    compile_range: Range,
    key: Optional[str] = None,
) -> tuple[Optional[Callable], Optional[Any]]:
    # When currently using the FULL_DECODE_ONLY mode,
    # the piecewise compilation level slicing process
    # in vllm is also encountered.
    # This process causes the output to no longer be
    # wrapped as a tuple when the fx graph has a single
    # output, but torch.compile has a mandatory check.
    fx_graph = graph.graph
    if not graph_returns_tuple(graph):
        output_node = fx_graph.output_node()
        with fx_graph.inserting_before(output_node):
            return_value = output_node.args[0]
            tuple_node = fx_graph.create_node("call_function",
                                              tuple,
                                              args=([return_value], ))
        output_node.args = (tuple_node, )
        graph.recompile()

    import torchair

    # TODO: use a better way to lazy register replacement, instead of import one by one
    # As an example, we directly import here to register replacement.
    # import vllm_ascend.compilation.npugraph_ex_passes.add_rms_norm_quant  # noqa

    torch.npu.set_compile_mode(jit_compile=False)
    config = torchair.CompilerConfig()
    # use aclgraph mode, avoid the transformation from fx graph to Ascend IR.
    config.mode = "reduce-overhead"
    # execute FX graph in eager mode before graph mode to optimize FX graph.
    config.debug.run_eagerly = True
    # static kernel switch, suitable for static shapes or scenes with less shape changes.
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True

    npugraph_ex = torchair.get_npu_backend(compiler_config=config)
    compile_graph = npugraph_ex(graph, example_inputs)
    return compile_graph, None


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
        compile_range: Range,
        key: Optional[str] = None,
    ) -> tuple[Optional[Callable], Optional[Any]]:

        ascend_config = get_ascend_config()
        if ascend_config.enable_npugraph_ex:
            return npugraph_ex_compile(graph, example_inputs, compiler_config,
                                       compile_range, key)
        else:
            return fusion_pass_compile(graph, example_inputs, compiler_config,
                                       compile_range, key)
