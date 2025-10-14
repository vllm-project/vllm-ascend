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

import torch
import torch.fx as fx
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.compilation.counter import compilation_counter


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

        current_pass_manager = compiler_config["graph_rewriter_manager"]
        arg_dtypes = get_dtype_from_args(example_inputs)
        arg_shapes = get_shapes_from_args(example_inputs)
        kwargs = {
            "runtime_shape": runtime_shape,
            "arg_shapes": arg_shapes,
            "arg_dtypes": arg_dtypes
        }
        graph = current_pass_manager(graph, **kwargs)
        compilation_counter.num_eager_compiles += 1
        
        