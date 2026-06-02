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


def _compute_decode_cudagraph_batch_sizes(vllm_config: VllmConfig) -> list[int]:
    num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
    uniform_decode_query_len = num_spec_tokens + 1
    max_num_tokens = vllm_config.scheduler_config.max_num_seqs * uniform_decode_query_len
    return [
        x
        for x in vllm_config.compilation_config.cudagraph_capture_sizes
        if max_num_tokens >= x >= uniform_decode_query_len
    ]


def _configure_backend(
    config: Any,
    ascend_compilation_config: AscendCompilationConfig,
    vllm_config: VllmConfig,
    process_kwargs_options: Callable | None = None,
) -> None:
    if process_kwargs_options is not None:
        # npugraph_ex (both old and new): build options dict and use _process_kwargs_options.
        # It maps flat option names to nested config paths for old versions,
        # and directly setattr for new versions with flat CompilerConfig.
        # force_eager=True: execute FX graph in eager mode before graph capture.
        # inplace_pass=False: disable reinplace pass to avoid gelu fallback to CPU.
        options: dict[str, Any] = {
            "force_eager": True,
            "inplace_pass": False,
        }
        if ascend_compilation_config.enable_static_kernel:
            logger.info_once(
                "enable_static_kernel is enabled, static shape kernel will be used to accelerate aclgraph execution.",
                scope="global",
            )
            options["static_kernel_compile"] = True
            # Set sym_range to limit static kernel compilation to specified batch sizes.
            options["_vllm_aclnn_static_kernel_sym_range"] = _compute_decode_cudagraph_batch_sizes(vllm_config)
        process_kwargs_options(config, {"options": options})
    else:
        # torchair (reduce-overhead): use nested config structure directly.
        # mode="reduce-overhead": use aclgraph mode, avoid fx graph to Ascend IR transformation.
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        # Disable reinplace pass to avoid gelu fallback to CPU causing host-device copy error.
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        if ascend_compilation_config.enable_static_kernel:
            logger.info_once(
                "enable_static_kernel is enabled, static shape kernel will be used to accelerate aclgraph execution.",
                scope="global",
            )
            config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
            config.experimental_config.aclgraph._aclnn_static_shape_kernel_sym_value_range = (
                _compute_decode_cudagraph_batch_sizes(vllm_config)
            )


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
    # Try npugraph_ex first, fall back to torchair for backward compatibility.
    try:
        import npugraph_ex as nge

        cache_path = os.path.join(cache_dir, key) if (cache_dir and key) else None

        torch.npu.set_compile_mode(jit_compile=False)
        config = nge.CompilerConfig()
        # _process_kwargs_options exists in both old and new npugraph_ex,
        # but in different modules: new -> compiler_config, old -> npugraphex_config.
        try:
            from npugraph_ex.configs.compiler_config import _process_kwargs_options
        except ImportError:
            from npugraph_ex.configs.npugraphex_config import _process_kwargs_options
        _configure_backend(
            config, ascend_compilation_config, vllm_config, process_kwargs_options=_process_kwargs_options
        )
        import npugraph_ex.npu_fx_compiler as nfx

        _original_get_compiled_gm = nfx._NpuFxCompiler._get_compiled_gm

        handle = None

        def patched_get_compiled_gm(self, graph, example_inputs):
            compiled_gm = _original_get_compiled_gm(self, graph, example_inputs)
            if cache_path:
                py_code = compiled_gm.get_code()
                if py_code:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "w") as f:
                        f.write(py_code)
                    logger.debug("Saved compiled graph to cache: %s", cache_path)
            return compiled_gm

        nfx._NpuFxCompiler._get_compiled_gm = patched_get_compiled_gm
        npugraph_ex = nge.get_npu_backend(compiler_config=config)
    except ImportError:
        import torchair

        torch.npu.set_compile_mode(jit_compile=False)
        config = torchair.CompilerConfig()
        _configure_backend(config, ascend_compilation_config, vllm_config)
        npugraph_ex = torchair.get_npu_backend(compiler_config=config)

    # torch.compile requires the output of the fx graph to be a tuple
    if not graph_returns_tuple(graph):
        compiled_fn = make_graph_return_tuple(graph, example_inputs, npugraph_ex)
    else:
        compiled_fn = npugraph_ex(graph, example_inputs)
    handle = (key, cache_path)

    nfx._NpuFxCompiler._get_compiled_gm = _original_get_compiled_gm
    return compiled_fn, handle


class AscendCompiler(CompilerInterface):
    """
    AscendCompiler is a custom compiler interface for the Ascend platform.
    This class provides a method to compile a PyTorch FX graph module with
    specific configurations for graph fusion and decomposition.
    """

    name = "AscendCompiler"

    # TODO(wxs): add passes related to compilation in compute_hash
    def compute_hash(self, vllm_config: VllmConfig) -> str:
        self.vllm_config = vllm_config
        ascend_compilation_config = get_ascend_config().ascend_compilation_config
        from hashlib import sha256

        import torch_npu

        factors = {
            "torch_npu_version": torch_npu.__version__,
            "enable_npugraph_ex": ascend_compilation_config.enable_npugraph_ex,
            "enable_static_kernel": ascend_compilation_config.enable_static_kernel,
        }
        logger.debug("AscendCompiler hash factors: %s", factors)
        return sha256(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]

    def initialize_cache(self, cache_dir, disable_cache=False, prefix=""):
        self.cache_dir = cache_dir
        self.disable_cache = disable_cache

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
            cache_dir = getattr(self, "cache_dir", None)
            logger.info("enable_npugraph_ex is enabled, which will bring graph compilation optimization.")
            assert hasattr(self, "vllm_config")
            return npugraph_ex_compile(
                graph,
                example_inputs,
                compiler_config,
                self.vllm_config,
                ascend_compilation_config,
                compile_range,
                key,
                cache_dir,
            )
        else:
            return fusion_pass_compile(graph, example_inputs, compiler_config, compile_range, key)

    def load(self, handle, graph, example_inputs, graph_index, compile_range):
        key, path = handle
        from npugraph_ex.npu_fx_compiler import _CompiledFxArtifacts, _CompiledFxGraph

        with open(path) as f:
            py_code = f.read()
        artifacts = _CompiledFxArtifacts()
        artifacts.py_code = py_code
        logger.debug("Loaded npugraph_ex compilation cache from %s", path)
        compiled_fn = _CompiledFxGraph.load_artifacts(artifacts)

        # The saved code was compiled from the graph after make_graph_return_tuple mutated it
        # to return a flat tuple. If the original graph didn't return a tuple, we need to
        # recreate the unflatten wrapper so callers receive the original output structure.
        if not graph_returns_tuple(graph):
            _inner_fn = compiled_fn

            def compiled_fn(*args, **kwargs):
                result = _inner_fn(*args, **kwargs)
                if isinstance(result, (tuple, list)) and len(result) == 1:
                    return result[0]
                return result

        return compiled_fn
