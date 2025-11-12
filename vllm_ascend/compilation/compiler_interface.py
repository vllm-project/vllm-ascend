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

from typing import Any, Callable, Optional
import functools
import torch
import torch.fx as fx
from vllm.compilation.compiler_interface import CompilerInterface
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.utils import output_node
import torch.utils._pytree as pytree

def get_dtype_from_args(args: list[Any]) -> list[torch.dtype]:
    """
    Extract the dtype from the kwargs dictionary.
    """
    dtype_list = []
    for value in args:
        if isinstance(value, torch.Tensor):
            dtype_list.append(value.dtype)
    return dtype_list


def get_shapes_from_args(args: list[Any]) -> list[torch.Size]:
    """
    Extract the shapes from the kwargs dictionary.
    """
    shape_list = []
    for value in args:
        if isinstance(value, torch.Tensor):
            shape_list.append(value.shape)
    return shape_list


def graph_returns_tuple(gm: fx.GraphModule) -> bool:
    """True if a FX graph returns a tuple"""
    if not isinstance(gm, fx.GraphModule):
        return True  # can't check this, assume true
    (rv,) = output_node(gm).args
    if isinstance(rv, (list, tuple)):
        return True
    if (
        isinstance(rv, torch.fx.node.Node)
        and hasattr(rv.target, "_schema")
        and len(rv.target._schema.returns) > 1
        and all(str(ret.type) == "Tensor" for ret in rv.target._schema.returns)
    ):
        # for graphs whose result is one node with multiple outputs
        return True
    return False
    
    
def make_graph_return_tuple(
    gm: fx.GraphModule,
) -> tuple[Any, fx.GraphModule]:
    """
    Mutate gm so it returns a tuple.  This is only needed for graphs
    not created by torchdynamo that return non-tuples.
    Returns:
        spec: The original output structure specification
        gm: The modified GraphModule that returns a tuple
    """
    node = output_node(gm)
    (rv,) = node.args
    rv, spec = pytree.tree_flatten(rv)
    with gm.graph.inserting_before(node):
        gm.graph.output(rv)
    gm.graph.erase_node(node)
    assert graph_returns_tuple(gm)
    
    return spec, gm


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
            arg_dtypes = get_dtype_from_args(example_inputs)
            arg_shapes = get_shapes_from_args(example_inputs)
            kwargs = {
                "runtime_shape": runtime_shape,
                "arg_shapes": arg_shapes,
                "arg_dtypes": arg_dtypes
            }
            graph = current_pass_manager(graph, **kwargs)
            return graph

        if not graph_returns_tuple(graph):
            spec, graph = make_graph_return_tuple(graph)
        else:
            spec = None

        compiled_fn = aot_autograd(fw_compiler=compile_inner)(graph, example_inputs)

        if spec is not None:
            @functools.wraps(compiled_fn)
            def wrapper(*args, **kwargs):
                return pytree.tree_unflatten(compiled_fn(*args, **kwargs), spec)
            return wrapper, None
        else:
            return compiled_fn, None