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
