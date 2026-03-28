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
import os
from collections.abc import Callable
from contextlib import contextmanager
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
from vllm.logger import logger

from vllm_ascend.ascend_config import AscendCompilationConfig, get_ascend_config
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


@contextmanager
def _capture_acl_codegen(cache_path: str):
    """Patch AclConcreteGraph.optimize_graph_without_runtime to capture the optimized fx_graph.

    After the expensive graph optimization completes, the resulting fx_graph (optimized
    eager-runnable GraphModule) is captured and saved to cache_path via torch.save.
    """
    try:
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
    except ImportError:
        logger.debug("torchair AclConcreteGraph not available, skipping cache capture")
        yield []
        return

    captured: list = []
    original = AclConcreteGraph.optimize_graph_without_runtime

    def _patched(self, *sample_args, observer=None):
        original(self, *sample_args, observer=observer)
        # After optimization self is a fully-optimized AclConcreteGraph.
        # Only capture once per compile() call.
        if captured:
            return
        try:
            fx_graph = getattr(self, "fx_graph", None)
            if fx_graph is not None:
                captured.append(fx_graph)
                logger.debug("Captured optimized fx_graph for cache")
            else:
                logger.debug("No fx_graph found on AclConcreteGraph, skipping cache capture")
        except Exception as e:
            logger.debug("Failed to capture optimized fx_graph: %s", e)

    AclConcreteGraph.optimize_graph_without_runtime = _patched
    try:
        yield captured
    finally:
        AclConcreteGraph.optimize_graph_without_runtime = original

    # Save to disk after the with-block (and after the patch is restored).
    if captured and cache_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save(captured[0], cache_path)
            logger.info("Saved npugraph_ex compilation cache: %s", cache_path)
        except Exception as e:
            logger.warning("Failed to write npugraph_ex cache to %s: %s", cache_path, e)


def npugraph_ex_compile(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    vllm_config: VllmConfig,
    ascend_compilation_config: AscendCompilationConfig,
    compile_range: Range,
    key: str | None = None,
    cache_dir: str | None = None,
) -> tuple[Callable | None, Any | None]:
    import torchair

    torch.npu.set_compile_mode(jit_compile=False)
    config = torchair.CompilerConfig()
    # use aclgraph mode, avoid the transformation from fx graph to Ascend IR.
    config.mode = "reduce-overhead"
    # execute FX graph in eager mode before graph mode to optimize FX graph.
    config.debug.run_eagerly = True
    # This is a temporary fix to resolve issues with inplace operations in some testcases like test_whisper.
    # Avoid to change torch.ops.aten.gelu.default to torch.ops.aten.gelu_.default which will fallback to CPU
    # and cause copy_between_host_and_device error.
    config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    if ascend_compilation_config.enable_static_kernel:
        logger.info(
            "enable_static_kernel is enabled, static shape kernel will be used to accelerate aclgraph execution."
        )
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

    # Determine the cache artifact path for this graph.
    cache_path: str | None = None
    if cache_dir and key:
        cache_path = os.path.join(cache_dir, key + ".acl_fx.pt")

    if cache_path:
        # Patch optimize_graph_without_runtime to capture the optimized fx_graph while
        # keeping run_eagerly=True so ACLGraphWrapper behaviour is unchanged.
        with _capture_acl_codegen(cache_path) as captured:
            if not graph_returns_tuple(graph):
                compiled_fn = make_graph_return_tuple(graph, example_inputs, npugraph_ex)
            else:
                compiled_fn = npugraph_ex(graph, example_inputs)
        handle = (key, cache_path) if captured else None
    else:
        # torch.compile requires the output of the fx graph to be a tuple
        if not graph_returns_tuple(graph):
            compiled_fn = make_graph_return_tuple(graph, example_inputs, npugraph_ex)
        else:
            compiled_fn = npugraph_ex(graph, example_inputs)
        handle = None

    return compiled_fn, handle


class AscendCompiler(CompilerInterface):
    """
    AscendCompiler is a custom compiler interface for the Ascend platform.
    This class provides a method to compile a PyTorch FX graph module with
    specific configurations for graph fusion and decomposition.
    """

    name = "AscendCompiler"

    def initialize_cache(self, cache_dir: str, disable_cache: bool = False, prefix: str = "") -> None:
        self.cache_dir: str | None = None if disable_cache else cache_dir

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        self.vllm_config = vllm_config
        ascend_compilation_config = get_ascend_config().ascend_compilation_config
        import torchair
        import torch_npu
        from hashlib import sha256
        factors = {
            "torchair_version": getattr(torchair, "__version__", "unknown"),
            "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
            "enable_npugraph_ex": ascend_compilation_config.enable_npugraph_ex,
            "enable_static_kernel": ascend_compilation_config.enable_static_kernel,
        }
        logger.debug("AscendCompiler hash factors: %s", factors)
        return sha256(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]

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

        from torch._guards import detect_fake_mode

        current_fake_mode = detect_fake_mode()
        if current_fake_mode is not None:
            example_inputs = [
                current_fake_mode.from_tensor(inp)
                if (
                    isinstance(inp, torch.Tensor)
                    and hasattr(inp, "fake_mode")
                    and inp.fake_mode is not current_fake_mode
                )
                else inp
                for inp in example_inputs
            ]

        ascend_compilation_config = get_ascend_config().ascend_compilation_config
        if ascend_compilation_config.enable_npugraph_ex:
            logger.info("enable_npugraph_ex is enabled, which will bring graph compilation optimization.")
            cache_dir = getattr(self, "cache_dir", None)
            return npugraph_ex_compile(
                graph, example_inputs, compiler_config, self.vllm_config,
                 ascend_compilation_config, compile_range, key, cache_dir=cache_dir,
            )
        else:
            return fusion_pass_compile(graph, example_inputs, compiler_config, compile_range, key)

    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable:
        assert isinstance(handle, tuple) and len(handle) == 2, f"Unexpected handle format: {handle}"
        _key, cache_path = handle

        if not os.path.exists(cache_path):
            raise RuntimeError(
                f"npugraph_ex cache file not found: {cache_path}. "
                "Delete the cache directory and restart to recompile."
            )

        try:
            # Load the optimized FX graph (eager-runnable GraphModule).
            # weights_only=False is required because we saved a full GraphModule, not just tensors.
            fx_graph = torch.load(cache_path, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load npugraph_ex cache {cache_path}: {e}") from e

        # The cached fx_graph was compiled from the (possibly tuple-wrapped) AOT graph,
        # so it always returns a tuple. Restore the calling convention that the original
        # compile() call would have produced.
        returns_tuple = graph_returns_tuple(graph)

        def compiled(*args: Any) -> Any:
            result = fx_graph(*args)
            if returns_tuple:
                return result
            # Non-tuple graph: unwrap the single output element.
            if isinstance(result, (list, tuple)):
                return result[0]
            return result

        logger.info("Loaded npugraph_ex compilation cache from %s", cache_path)
        return compiled
