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
import copy
import functools
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple
from torch._inductor.decomposition import select_decomp_table
from torch.fx import GraphModule
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig
from vllm.config.utils import Range

from vllm_ascend.ascend_config import NpugraphExConfig, get_ascend_config
from vllm_ascend.utils import COMPILATION_PASS_KEY


def compile_fx(graph: GraphModule, example_inputs: list, inner_compile: Callable, decompositions: dict) -> Callable:
    recursive_compile_fx = functools.partial(compile_fx, inner_compile=inner_compile, decompositions=decompositions)

    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(graph, example_inputs, recursive_compile_fx)
    return aot_autograd(fw_compiler=inner_compile)(graph, example_inputs)


def fusion_pass_compile(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    compile_range: Range,
    key: str | None = None,
) -> tuple[Callable | None, Any | None]:
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
    vllm_config: VllmConfig,
    npugraph_ex_config: NpugraphExConfig,
    compile_range: Range,
    key: str | None = None,
) -> tuple[Callable | None, Any | None]:
    import torchair

    torch.npu.set_compile_mode(jit_compile=False)
    config = torchair.CompilerConfig()
    # use aclgraph mode, avoid the transformation from fx graph to Ascend IR.
    config.mode = "reduce-overhead"
    # execute FX graph in eager mode before graph mode to optimize FX graph.
    config.debug.run_eagerly = True
    if npugraph_ex_config.enable_static_kernel:
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        # According to the cudagraph_capture_size configuration, set the shapes
        # that can trigger the compilation of static kernel. If this configuration is
        # not applied, new shapes will trigger the compilation of static kernels,
        # affecting program execution.
        num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        uniform_decode_query_len = num_spec_tokens + 1
        max_num_tokens = vllm_config.scheduler_config.max_num_seqs * uniform_decode_query_len
        decode_cudagraph_batch_sizes = [
            x
            for x in vllm_config.compilation_config.cudagraph_capture_sizes
            if max_num_tokens >= x >= uniform_decode_query_len
        ]
        config.experimental_config.aclgraph._aclnn_static_shape_kernel_sym_value_range = decode_cudagraph_batch_sizes

    npugraph_ex = torchair.get_npu_backend(compiler_config=config)

    # torch.compile requires the output of the fx graph to be a tuple
    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(graph, example_inputs, npugraph_ex), None
    return npugraph_ex(graph, example_inputs), None


class AscendCompiler(CompilerInterface):
    """
    AscendCompiler is a custom compiler interface for the Ascend platform.
    This class provides a method to compile a PyTorch FX graph module with
    specific configurations for graph fusion and decomposition.
    """

    name = "AscendCompiler"

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        npugraph_ex_config = get_ascend_config().npugraph_ex_config
        if npugraph_ex_config.enable:
            self.vllm_config = vllm_config
        return vllm_config.compute_hash()

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable | None, Any | None]:
        # inductor can inplace modify the graph, so we need to copy it
        # see https://github.com/pytorch/pytorch/issues/138980
        graph = copy.deepcopy(graph)

        npugraph_ex_config = get_ascend_config().npugraph_ex_config
        if npugraph_ex_config.enable:
            assert hasattr(self, "vllm_config")
            return npugraph_ex_compile(
                graph, example_inputs, compiler_config, self.vllm_config, npugraph_ex_config, compile_range, key
            )
        else:
            return fusion_pass_compile(graph, example_inputs, compiler_config, compile_range, key)
