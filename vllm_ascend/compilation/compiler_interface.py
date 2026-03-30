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
import os
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


def _patch_acl_graph_run_eagerly() -> None:
    """Patch AclGraph.__call__ to run the FX graph eagerly when fx_forward is available.

    When loading a compiled graph from cache, load_artifacts() creates AclGraph(fx_forward=forward),
    and ACLGraphWrapper already handles NPU graph capture for the entire model forward. If AclGraph
    also tries to capture an NPU graph internally, the two capture scopes nest, and NPU raises:
        RuntimeError: Cannot prepare for replay during capturing stage.

    The patch only redirects to fx_run_eagerly when fx_forward is set (cache load path). Cold
    compilation creates AclGraph(fx_graph=..., fx_forward=None), which must go through the normal
    NPU graph compile path.
    """
    import importlib
    classes_to_patch = []
    for mod_path in (
        "torch_npu.dynamo.torchair._acl_concrete_graph.acl_graph",
        "torchair._acl_concrete_graph.acl_graph",
    ):
        try:
            mod = importlib.import_module(mod_path)
            cls = mod.AclGraph
            if cls not in classes_to_patch:
                classes_to_patch.append(cls)
        except Exception:
            pass

    for AclGraph in classes_to_patch:
        original_call = AclGraph.__call__

        def patched_call(self, *args, _orig=original_call, **kwargs):
            if self._fx_forward is not None:
                return self.fx_run_eagerly(*args, **kwargs)
            return _orig(self, *args, **kwargs)

        AclGraph.__call__ = patched_call

    logger.debug("Patched AclGraph.__call__ to use fx_run_eagerly for cache-loaded graphs (%d class(es))", len(classes_to_patch))


# Apply the patch globally at import time so it takes effect regardless of which code path
# is taken (cold compile or cache load). The guard inside patched_call (fx_forward is not None)
# ensures cold-compile AclGraph instances still go through the normal NPU graph path.
try:
    _patch_acl_graph_run_eagerly()
except Exception:
    # torch_npu may not be available at import time in some environments; the patch
    # will be retried inside npugraph_ex_compile() and load() when torchair is imported.
    pass


def _wrap_compiled_fn_for_cache(compiled_fn: Callable, cache_path: str) -> Callable:
    """Wrap compiled_fn to capture and save py_code on first real execution.

    _CompiledFxGraph.get_code() is called lazily on the first forward pass (not at compile
    time), so we cannot hook it during npugraph_ex(). Instead, we wrap the compiled callable
    and install the hook on the first actual call, when get_code() will fire naturally.
    """
    from torchair.npu_fx_compiler import _CompiledFxGraph

    first_call = [True]

    def wrapper(*args, **kwargs):
        if not first_call[0]:
            return compiled_fn(*args, **kwargs)

        first_call[0] = False
        py_code_holder = [None]
        original_get_code = _CompiledFxGraph.get_code

        def hijacked_get_code(self, extend_config=None):
            code = original_get_code(self, extend_config)
            if isinstance(code, str):
                py_code_holder[0] = code
            return code

        _CompiledFxGraph.get_code = hijacked_get_code
        try:
            result = compiled_fn(*args, **kwargs)
        finally:
            _CompiledFxGraph.get_code = original_get_code

        if py_code_holder[0]:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w") as f:
                    f.write(py_code_holder[0])
                logger.info("Saved compiled graph to cache: %s", cache_path)
            except Exception as e:
                logger.warning("Failed to save compiled graph to cache: %s, error: %s", cache_path, e)
        else:
            logger.warning("py_code not captured at first execution, cache skipped: %s", cache_path)

        return result

    return wrapper


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
    from torchair.npu_fx_compiler import _CompiledFxGraph, _CompiledFxArtifacts

    cache_path = os.path.join(cache_dir, key) if (cache_dir and key) else None
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                py_code = f.read()
            artifacts = _CompiledFxArtifacts()
            artifacts.py_code = py_code
            compiled = _CompiledFxGraph.load_artifacts(artifacts)
            logger.info("Loaded compiled graph from cache: %s", cache_path)
            return compiled, (key, cache_path)
        except Exception as e:
            logger.warning("Failed to load compiled graph from cache: %s, error: %s", cache_path, e)

    torch.npu.set_compile_mode(jit_compile=False)
    config = torchair.CompilerConfig()
    # use aclgraph mode, avoid the transformation from fx graph to Ascend IR.
    config.mode = "reduce-overhead"
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
    # torch.compile requires the output of the fx graph to be a tuple
    if not graph_returns_tuple(graph):
        compiled_fn = make_graph_return_tuple(graph, example_inputs, npugraph_ex)
    else:
        compiled_fn = npugraph_ex(graph, example_inputs)

    if cache_path:
        compiled_fn = _wrap_compiled_fn_for_cache(compiled_fn, cache_path)

    return compiled_fn, None


class AscendCompiler(CompilerInterface):
    """
    AscendCompiler is a custom compiler interface for the Ascend platform.
    This class provides a method to compile a PyTorch FX graph module with
    specific configurations for graph fusion and decomposition.
    """

    name = "AscendCompiler"

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
    
    def initialize_cache(self, cache_dir, disable_cache = False, prefix = ""):
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
            if getattr(self, "disable_cache", False):
                cache_dir = None
            logger.info("enable_npugraph_ex is enabled, which will bring graph compilation optimization.")
            assert hasattr(self, "vllm_config")
            return npugraph_ex_compile(
                graph, example_inputs, compiler_config, self.vllm_config, ascend_compilation_config, compile_range, key, cache_dir
            )
        else:
            return fusion_pass_compile(graph, example_inputs, compiler_config, compile_range, key)
        
    def load(self, handle, graph, example_inputs, graph_index, compile_range):
        key, path = handle
        from torchair.npu_fx_compiler import _CompiledFxGraph, _CompiledFxArtifacts
        with open(path, "r") as f:
            py_code = f.read()
        artifacts = _CompiledFxArtifacts()
        artifacts.py_code = py_code
        logger.info("Loaded npugraph_ex compilation cache from %s", path)
        return _CompiledFxGraph.load_artifacts(artifacts)