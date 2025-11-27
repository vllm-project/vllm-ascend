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

from typing import Any, Callable, Optional

import torch
import torch.fx as fx
import vllm
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.compilation.counter import compilation_counter

from vllm_ascend.ascend_config import get_ascend_config


class EagerAdaptorPatch(CompilerInterface):
    name = "eager"

    def compile(
        self,
        fx_graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
    ) -> tuple[Optional[Callable], Optional[Any]]:

        ascend_config = get_ascend_config()
        if not ascend_config.enable_npugraph_ex_optimize:
            compilation_counter.num_eager_compiles += 1
            return fx_graph, None

        # When currently using the FULL_DECODE_ONLY mode,
        # the piecewise compilation level slicing process
        # in vllm is also encountered.
        # This process causes the output to no longer be
        # wrapped as a tuple when the fx graph has a single
        # output, but torch.compile has a mandatory check.
        graph = fx_graph.graph
        output_node = graph.output_node()
        return_value = output_node.args[0]
        if not (isinstance(return_value, tuple) or
                (isinstance(return_value, fx.Node) and return_value.op
                 == "call_function" and return_value.target is tuple)):
            with graph.inserting_before(output_node):
                tuple_node = graph.create_node("call_function",
                                               tuple,
                                               args=([return_value], ))
            output_node.args = (tuple_node, )
            fx_graph.recompile()

        import torchair

        torch.npu.set_compile_mode(jit_compile=False)
        config = torchair.CompilerConfig()
        config.debug.run_eagerly = True
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        config.debug.aclgraph.disable_mempool_reuse_in_same_fx = True

        npugraph_ex = torchair.get_npu_backend(compiler_config=config)
        compile_graph = npugraph_ex(fx_graph, example_inputs)
        compilation_counter.num_eager_compiles += 1
        return compile_graph, None


vllm.compilation.compiler_interface.EagerAdaptor = EagerAdaptorPatch
