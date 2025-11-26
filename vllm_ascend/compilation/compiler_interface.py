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
from collections.abc import Sequence
from typing import Any, Callable, Optional

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.utils import InputType, output_node
from torch.fx import GraphModule
from vllm.compilation.compiler_interface import CompilerInterface


def graph_returns_tuple(gm: fx.GraphModule) -> bool:
    """True if a FX graph returns a tuple"""
    if not isinstance(gm, fx.GraphModule):
        return True  # can't check this, assume true
    (rv, ) = output_node(gm).args
    if isinstance(rv, (list, tuple)):
        return True
    if (isinstance(rv, torch.fx.node.Node) and hasattr(rv.target, "_schema")
            and len(rv.target._schema.returns) > 1 and all(
                str(ret.type) == "Tensor"
                for ret in rv.target._schema.returns)):
        # for graphs whose result is one node with multiple outputs
        return True
    return False


def make_graph_return_tuple(
    gm: GraphModule,
    inputs: Sequence[InputType],
    compile_gm: Callable[..., Any],
) -> Callable[..., Any]:
    """
    Mutate gm so it returns a tuple.  This is only needed for graphs
    not created by torchdynamo that return non-tuples.
    """
    node = output_node(gm)
    (rv, ) = node.args
    rv, spec = pytree.tree_flatten(rv)
    with gm.graph.inserting_before(node):
        gm.graph.output(rv)
    gm.graph.erase_node(node)
    assert graph_returns_tuple(gm)

    compiled_fn = compile_gm(gm, inputs)

    @functools.wraps(compiled_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return pytree.tree_unflatten(compiled_fn(*args, **kwargs), spec)

    return wrapper


def compile_fx(model_: GraphModule, example_inputs_: list,
               inner_compile: Callable, decompositions: dict) -> Callable:
    recursive_compile_fx = functools.partial(compile_fx,
                                             inner_compile=inner_compile,
                                             decompositions=decompositions)

    if not graph_returns_tuple(model_):
        return make_graph_return_tuple(model_, example_inputs_,
                                       recursive_compile_fx)
    return aot_autograd(fw_compiler=inner_compile)(model_, example_inputs_)


class AscendAdaptor(CompilerInterface):
    name = "AscendAdaptor"

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
            model_=graph,
            example_inputs_=example_inputs,
            inner_compile=compile_inner,
            decompositions=decompositions,
        )

        return compiled_fn, None
