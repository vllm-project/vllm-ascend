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
import hashlib
import os
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch
import torch
import torch._inductor.compile_fx
import torch.fx as fx
from vllm.compilation.compiler_interface import CompilerInterface, AlwaysHitShapeEnv, get_inductor_factors, set_inductor_config
from vllm.compilation.pass_manager import PostGradPassManager

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.utils import is_torch_equal_or_newer

from vllm.compilation.inductor_pass import pass_context



class AscendAdaptor(CompilerInterface):
    name = "ascend_adaptor"

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
    ) -> tuple[Optional[Callable], Optional[Any]]:

        graph_rewriter_manager = compiler_config["graph_rewriter_manager"]
        graph = graph_rewriter_manager(graph)

        compilation_counter.num_eager_compiles += 1
        # we don't need to compile the graph, just return the graph itself.
        # It does not support caching, return None for the handle.
        return graph, None
