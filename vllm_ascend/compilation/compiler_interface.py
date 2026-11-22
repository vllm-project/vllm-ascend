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
import hashlib
import json
import os
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.fx as fx
import torch_npu
from npugraph_ex._acl_concrete_graph.static_kernel import _set_static_kernel_blacklist
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple
from torch._inductor.decomposition import select_decomp_table
from torch.fx import GraphModule
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import logger
from vllm.utils.math_utils import round_up

from vllm_ascend.ascend_config import AscendCompilationConfig, get_ascend_config
from vllm_ascend.utils import COMPILATION_PASS_KEY, vllm_version_is

# Marker attribute used to detect that the static kernel run-package cache
# patch is already installed. The state is tracked on the patched target
# itself, so it disappears together with the patch.
_STATIC_KERNEL_CACHE_PATCH_FLAG = "_vllm_ascend_static_kernel_run_package_cache"

# Name of the metadata file that records the runtime properties a cached run
# package was built with.
_STATIC_KERNEL_CACHE_META_FILE = "static_kernel_cache_meta.json"


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
    compilation_config = vllm_config.compilation_config
    capture_sizes = set(compilation_config.cudagraph_capture_sizes)
    if vllm_config.use_v2_model_runner and compilation_config.cudagraph_mode.separate_routine():
        # V2 uniform decode capture rounds token counts up to the query length.
        # With MTP5, for example, configured size 56 captures 60 tokens. Keep
        # the raw sizes as well for routines that do not perform this rounding.
        max_capture_size = compilation_config.max_cudagraph_capture_size
        capture_sizes.update(
            rounded
            for size in compilation_config.cudagraph_capture_sizes
            if (rounded := round_up(size, uniform_decode_query_len)) <= max_capture_size
        )
    return sorted(x for x in capture_sizes if max_num_tokens >= x >= uniform_decode_query_len)


def _initialize_static_kernel_runtime(vllm_config: VllmConfig) -> None:
    if "LOCAL_WORLD_SIZE" not in os.environ:
        # External DP config counts one replica; runtime topology includes all local ranks.
        # shortcut: uniform workers and contiguous node ranks; revisit for other layouts.
        parallel_config = vllm_config.parallel_config
        actual_local_world_size = parallel_config.local_world_size * parallel_config.data_parallel_size_local
        topology_source = "parallel_config"
        if parallel_config.data_parallel_external_lb:
            from vllm.distributed.parallel_state import get_node_count

            runtime_world_size = torch.distributed.get_world_size()
            node_count = get_node_count()
            if runtime_world_size != parallel_config.world_size_across_dp:
                raise RuntimeError("External DP static compilation requires the full DP worker world")
            actual_local_world_size, remainder = divmod(runtime_world_size, node_count)
            if remainder:
                raise RuntimeError("Static compilation requires worker world size divisible by node count")
            topology_source = f"runtime_world_size={runtime_world_size}, node_count={node_count}"
        os.environ["LOCAL_WORLD_SIZE"] = str(actual_local_world_size)
        logger.info_once(
            "Setting LOCAL_WORLD_SIZE=%d for static kernel "
            "(source=%s, local_world_size=%d, data_parallel_size_local=%d).",
            actual_local_world_size,
            topology_source,
            parallel_config.local_world_size,
            parallel_config.data_parallel_size_local,
            scope="global",
        )

    # TODO: Temporary workaround. Remove this static kernel blacklist once these ops are supported.
    incompatible_ops = [
        "TopKV2",  # minimax-m3 910C
        "GenericBlockSparseAttention",  # minimax-m3 910C
        "QuantBatchMatmulV3",  # glm5.2 910C
        "TransposeQuantBatchMatMul",  # dsv4 950
        "QuantLightningIndexer",  # glm5.2 950
        "PoolKeyIndexer",  # glm5.3-flash
    ]
    logger.info_once(
        "Skipping static kernel compilation for incompatible ops: %s.",
        ",".join(incompatible_ops),
        scope="global",
    )
    _set_static_kernel_blacklist(incompatible_ops)


def _static_kernel_cache_root() -> str:
    """Return the root directory that holds the cached static kernel run packages.

    torch_npu looks the run packages up under this directory, so it has to
    survive a restart to be worth anything. It therefore lives under
    ``VLLM_CACHE_ROOT``, which the operator already points at persistent
    storage for the other compile caches (the AOT and npugraph caches live
    there too).
    """
    cache_root = os.getenv("VLLM_CACHE_ROOT")
    if not cache_root:
        cache_root = os.path.join(os.path.expanduser("~"), ".cache", "vllm")
    return os.path.join(cache_root, "ascend_static_kernel")


def _static_kernel_deterministic_level() -> int:
    """Return the CANN deterministic level of the current runtime."""
    try:
        return int(torch_npu.npu._get_deterministic_level())
    except Exception:  # noqa: BLE001
        # The level is only available on the NPU images; treat an unknown
        # level as the default one.
        return 0


def _static_kernel_runtime_properties() -> dict[str, Any]:
    """Return the runtime properties a cached run package is bound to.

    A run package is only valid for the CANN version, the device and the
    deterministic settings it was compiled with, so all of them are recorded
    next to the packages and checked before a package is installed.
    """
    try:
        device_name = torch_npu.npu.get_device_name()
    except Exception:  # noqa: BLE001
        # The device name is only available on the NPU images.
        device_name = ""
    return {
        "cann_version": torch_npu.npu.utils.get_cann_version(),
        "device_name": device_name,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_level": _static_kernel_deterministic_level(),
    }


def _static_kernel_graph_id(converter: Any = None) -> str:
    """Return a stable identifier of the graph that is being compiled.

    The identifier has to survive a restart, so it is derived from the
    generated code of the fx graph instead of from the object identity.
    """
    graph = getattr(converter, "_fx_graph", None)
    code = getattr(graph, "code", None)
    if not isinstance(code, str):
        return ""
    return hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()[:32]


def _static_kernel_function_id(fx_func: Any = None) -> str:
    """Return a stable identifier of the compiled function.

    It is the fallback for the cases where the identifier of the graph is
    not available, e.g. when the cache keys are not built by
    ``AclConcreteGraph.normalize_config``.
    """
    code = getattr(fx_func, "__code__", None)
    if code is None:
        return ""
    return hashlib.sha256(code.co_code).hexdigest()[:32]


def _static_kernel_input_signature(args: tuple) -> str:
    """Return a signature of the inputs the static kernels are compiled for.

    The static kernels are compiled for concrete shapes, so the shapes are
    part of the identity of a compiled piece.
    """
    parts = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            parts.append(f"{arg.dtype}:{tuple(arg.shape)}")
        else:
            parts.append(type(arg).__name__)
    return "|".join(parts)


def _static_kernel_world_size() -> int:
    """Return the number of ranks that take part in the compilation."""
    try:
        return torch.distributed.get_world_size()
    except Exception:  # noqa: BLE001
        return 1


def _static_kernel_piece_cache_dir(config: dict[str, Any], fx_func: Any, args: tuple) -> str:
    """Return the cache directory of the piece that is being compiled.

    torch_npu installs the first run package it finds under
    ``<compile_cache_dir>/static_kernel_compile_outputs`` without checking
    which graph it belongs to, so a directory shared by several graphs would
    let the first compiled piece short-circuit all the others. Every piece
    therefore gets its own directory.

    The directory name is keyed by the graph, the input shapes, the topology
    and the runtime versions, so a piece is only ever reused for the very
    same graph and shapes, and a CANN upgrade or a change of the
    deterministic settings selects a fresh directory instead of reusing stale
    run packages.
    """
    root = config.get("static_kernel_compile.compile_cache_dir") or _static_kernel_cache_root()
    graph_id = config.get("static_kernel_compile.graph_id") or _static_kernel_function_id(fx_func)
    digest = hashlib.sha256()
    digest.update(str(graph_id).encode())
    digest.update(_static_kernel_input_signature(args).encode())
    digest.update(f"world_size={_static_kernel_world_size()}".encode())
    digest.update(f"local_world_size={os.environ.get('LOCAL_WORLD_SIZE', '')}".encode())
    for name, value in sorted(_static_kernel_runtime_properties().items()):
        digest.update(f"{name}={value}".encode())
    return os.path.join(root, digest.hexdigest()[:32])


def _static_kernel_cache_meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _STATIC_KERNEL_CACHE_META_FILE)


def _read_static_kernel_cache_meta(cache_dir: str) -> dict[str, Any] | None:
    """Return the runtime properties the cached run packages were built with."""
    try:
        with open(_static_kernel_cache_meta_path(cache_dir)) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    required = ("cann_version", "deterministic", "deterministic_level")
    if any(key not in meta for key in required):
        return None
    return meta


def _write_static_kernel_cache_meta(cache_dir: str, meta: dict[str, Any]) -> None:
    """Record the runtime properties the run packages are built with."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(_static_kernel_cache_meta_path(cache_dir), "w") as f:
            json.dump(meta, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to record the static kernel cache metadata in %s: %s", cache_dir, exc)


def _static_kernel_cache_options(converter: Any = None) -> dict[str, Any]:
    """Return the torch_npu static kernel cache keys for the current runtime.

    These are the same keys that torch_npu's ``ModelCacheSaver`` injects for
    ``npugraph_ex.inference.cache_compile``:

    * ``use_cache_compile`` selects the "look the run package up, install it"
      path instead of recompiling from scratch,
    * ``cann_version`` invalidates the cache when the CANN version changes,
    * ``compile_cache_dir`` is where the run packages are looked up,
    * ``deterministic`` invalidates the cache when the deterministic-algorithms
      setting changes,
    * ``deterministic_level`` invalidates the cache when the CANN
      deterministic level changes.

    ``graph_id`` is added on top of them: it identifies the graph the cache
    keys belong to, so the run packages of one graph are never reused for
    another one.
    """
    options = {
        "static_kernel_compile.use_cache_compile": True,
        "static_kernel_compile.cann_version": torch_npu.npu.utils.get_cann_version(),
        "static_kernel_compile.compile_cache_dir": _static_kernel_cache_root(),
        "static_kernel_compile.deterministic": torch.are_deterministic_algorithms_enabled(),
        "static_kernel_compile.deterministic_level": _static_kernel_deterministic_level(),
    }
    graph_id = _static_kernel_graph_id(converter)
    if graph_id:
        options["static_kernel_compile.graph_id"] = graph_id
    return options


def _static_kernel_multicard_env_valid(static_kernel: Any) -> bool:
    """Return whether the static kernel feature may run in this environment.

    torch_npu disables the static kernel feature when the multi-card
    environment is not set up, so the same check is done before the cache
    path is taken.
    """
    is_valid = getattr(static_kernel, "_is_multicard_env_valid", None)
    if is_valid is None:
        return True
    try:
        return bool(is_valid())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to check the static kernel multi-card environment: %s", exc)
        return False


def _enable_static_kernel_run_package_cache() -> None:
    """Let torch_npu reuse cached static kernel run packages.

    torch_npu only reads the cache keys when the config is *not* the
    ``force_eager`` + ``static_kernel_compile`` combination::

        if not (config.get("force_eager", False) and config.get("static_kernel_compile", False)):
            use_cache_compile = config.get("static_kernel_compile.use_cache_compile", False)

    vllm-ascend compiles every npugraph_ex piece with ``force_eager=True``, so
    that branch is never taken and every restart recompiles all static
    kernels from scratch, leaving a new ``ts<timestamp>_pid<pid>_outputs``
    directory behind each time.

    The run-package cache itself does not depend on ``force_eager``: the
    install path only reselects the already-compiled kernels. This patch
    therefore restores the upstream decision logic, reading the cache keys
    whenever they are present.

    The keys are injected into the runtime config through
    ``AclConcreteGraph.normalize_config``, the single place that builds the
    config dict the static kernel compiler reads.
    """
    from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel
    from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph

    # The patch is applied once per process; initialize_cache() can be
    # called more than once, and stacking the wrappers would nest the
    # decisions. The state is tracked on the patched targets themselves.
    if getattr(AclConcreteGraph.normalize_config, _STATIC_KERNEL_CACHE_PATCH_FLAG, False) and getattr(
        static_kernel.compile_static_kernel, _STATIC_KERNEL_CACHE_PATCH_FLAG, False
    ):
        return

    original_normalize_config = AclConcreteGraph.normalize_config

    def normalize_config(self):
        # The config object belongs to npugraph_ex, so read it defensively:
        # a missing attribute must not break the compilation.
        config_obj = getattr(self, "_config", None)
        if config_obj is None or not getattr(config_obj, "static_kernel_compile", False):
            return original_normalize_config(self)
        configs = original_normalize_config(self)
        configs.update(_static_kernel_cache_options(self))
        return configs

    original_compile_static_kernel = static_kernel.compile_static_kernel

    def compile_static_kernel(fx_func, *args, config=None, build_dir=None, **kwargs):
        if config is None or not config.get("static_kernel_compile.use_cache_compile", False):
            return original_compile_static_kernel(fx_func, *args, config=config, build_dir=build_dir, **kwargs)

        if not _static_kernel_multicard_env_valid(static_kernel):
            # torch_npu disables the static kernel feature in that case, so
            # let it take the decision.
            return original_compile_static_kernel(fx_func, *args, config=config, build_dir=build_dir, **kwargs)

        try:
            cache_dir = _static_kernel_piece_cache_dir(config, fx_func, args)
            # torch_npu refuses to use a cache directory that does not exist,
            # and it skips the compilation entirely in that case, so the
            # directory has to be created before it is handed over.
            os.makedirs(cache_dir, exist_ok=True)
            # The runtime properties the cached run packages were built with
            # are read back from the metadata that was written when they
            # were compiled, so that torch_npu compares them against the
            # current runtime instead of comparing the runtime with itself.
            meta = _read_static_kernel_cache_meta(cache_dir)
            if meta is None:
                meta = _static_kernel_runtime_properties()
                _write_static_kernel_cache_meta(cache_dir, meta)
            static_kernel._install_or_compile_static_kernel(
                fx_func,
                *args,
                cached_cann_version=meta["cann_version"],
                compile_cache_dir=cache_dir,
                cached_deterministic=meta["deterministic"],
                cached_deterministic_level=meta["deterministic_level"],
                config=config,
                **kwargs,
            )
            return
        except Exception as exc:  # noqa: BLE001
            # The install path is a private npugraph_ex API. If it is
            # gone or changed, fall back to compiling instead of failing
            # the whole compilation.
            logger.warning(
                "Failed to install the cached static kernel run package, falling back to compiling it: %s",
                exc,
            )
        return original_compile_static_kernel(fx_func, *args, config=config, build_dir=build_dir, **kwargs)

    setattr(normalize_config, _STATIC_KERNEL_CACHE_PATCH_FLAG, True)
    setattr(compile_static_kernel, _STATIC_KERNEL_CACHE_PATCH_FLAG, True)
    AclConcreteGraph.normalize_config = normalize_config
    static_kernel.compile_static_kernel = compile_static_kernel
    # acl_graph.py imported the symbol directly, so rebind it there too.
    acl_graph.compile_static_kernel = compile_static_kernel

    logger.info_once(
        "Static kernel run packages will be cached in %s and reused across restarts.",
        _static_kernel_cache_root(),
        scope="global",
    )


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
            "clone_input": False,
            "clone_output": False,
        }
        if ascend_compilation_config.enable_static_kernel:
            logger.info_once(
                "enable_static_kernel is enabled, static shape kernel will be used to accelerate aclgraph execution.",
                scope="global",
            )

            options["static_kernel_compile"] = True
            # Set sym_range to limit static kernel compilation to specified batch sizes.
            # PIECEWISE graphs capture and replay exactly at the cudagraph capture
            # sizes, so compile those shapes; other routes use decode batch sizes.
            compilation_config = vllm_config.compilation_config
            if compilation_config.cudagraph_mode.requires_piecewise_compilation():
                # Pieces are captured and replayed exactly at the cudagraph
                # capture sizes. Newer vllm resolves the prefill token grid
                # and the decode request grid separately, so prefer them and
                # cover both; fall back to the legacy unified sizes.
                prefill_sizes = getattr(compilation_config, "prefill_cudagraph_capture_token_sizes", None)
                decode_sizes = getattr(compilation_config, "decode_cudagraph_capture_request_sizes", None)
                if prefill_sizes is None and decode_sizes is None:
                    sym_range = list(compilation_config.cudagraph_capture_sizes or [])
                else:
                    sym_range = list(prefill_sizes or [])
                    sym_range.extend(size for size in decode_sizes or [] if size not in sym_range)
                options["_vllm_aclnn_static_kernel_sym_range"] = sym_range
            else:
                options["_vllm_aclnn_static_kernel_sym_range"] = _compute_decode_cudagraph_batch_sizes(vllm_config)
            if ascend_compilation_config.enable_super_kernel:
                options["super_kernel_optimize"] = True
                options["super_kernel_optimize_options"] = {
                    "dcci_after_kernel_end": [".*"],
                }
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

        def patched_get_compiled_gm(self, graph, example_inputs):
            compiled_gm = _original_get_compiled_gm(self, graph, example_inputs)
            if cache_path:
                py_code = compiled_gm.get_code()
                if py_code:
                    # Triton kernel indices (kernel_side_table) are registered in-process
                    # at compile time and are not serializable across process boundaries.
                    # Graphs containing triton_kernel_wrapper calls must not be cached,
                    # because loading the py_code in a new process will hit an
                    # AssertionError in kernel_side_table.get_kernel().
                    if "triton_kernel_wrapper" in py_code:
                        logger.info(
                            "Skipping npugraph_ex cache for graph containing Triton kernels "
                            "(kernel_side_table indices are process-local): %s",
                            cache_path,
                        )
                    else:
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        with open(cache_path, "w") as f:
                            f.write(py_code)
                        logger.info("Saved compiled graph to cache: %s", cache_path)
            return compiled_gm

        nfx._NpuFxCompiler._get_compiled_gm = patched_get_compiled_gm
        backend = nge.get_npu_backend(compiler_config=config)
        # torch.compile requires the output of the fx graph to be a tuple
        if not graph_returns_tuple(graph):
            compiled_fn = make_graph_return_tuple(graph, example_inputs, backend)
        else:
            compiled_fn = backend(graph, example_inputs)
        nfx._NpuFxCompiler._get_compiled_gm = _original_get_compiled_gm
        return compiled_fn, (key, cache_path)
    except ImportError:
        import torchair

        torch.npu.set_compile_mode(jit_compile=False)
        config = torchair.CompilerConfig()
        _configure_backend(config, ascend_compilation_config, vllm_config)
        backend = torchair.get_npu_backend(compiler_config=config)
        # torch.compile requires the output of the fx graph to be a tuple
        if not graph_returns_tuple(graph):
            compiled_fn = make_graph_return_tuple(graph, example_inputs, backend)
        else:
            compiled_fn = backend(graph, example_inputs)
        return compiled_fn, None


class AscendCompiler(CompilerInterface):
    """
    AscendCompiler is a custom compiler interface for the Ascend platform.
    This class provides a method to compile a PyTorch FX graph module with
    specific configurations for graph fusion and decomposition.
    """

    name = "AscendCompiler"

    def compile_factors(self, vllm_config: VllmConfig) -> dict[str, Any]:
        self.vllm_config = vllm_config
        ascend_compilation_config = get_ascend_config().ascend_compilation_config

        factors = {
            "torch_npu_version": torch_npu.__version__,
            "enable_npugraph_ex": ascend_compilation_config.enable_npugraph_ex,
            "enable_static_kernel": ascend_compilation_config.enable_static_kernel,
            "enable_super_kernel": ascend_compilation_config.enable_super_kernel,
        }
        logger.info("AscendCompiler hash factors: %s", factors)
        return factors

    if vllm_version_is("0.33.0"):

        def compute_hash(self, vllm_config: VllmConfig) -> str:
            # Main rejects subclasses defining compute_hash (#65652).
            from hashlib import sha256

            return sha256(str(self.compile_factors(vllm_config)).encode(), usedforsecurity=False).hexdigest()[:10]

    def initialize_cache(self, cache_dir: str, disable_cache: bool = False, prefix: str = "") -> None:
        """Initialize cache settings and runtime state before compiling or loading a graph."""
        self.cache_dir = cache_dir
        self.disable_cache = disable_cache
        ascend_compilation_config = get_ascend_config().ascend_compilation_config
        if ascend_compilation_config.enable_npugraph_ex and ascend_compilation_config.enable_static_kernel:
            _initialize_static_kernel_runtime(self.vllm_config)
            if disable_cache:
                # The compile caches are disabled, so the static kernel run
                # packages must not be cached either.
                logger.info_once(
                    "The compilation cache is disabled, the static kernel run packages will not be reused.",
                    scope="global",
                )
            else:
                _enable_static_kernel_run_package_cache()

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
            cache_dir = None if getattr(self, "disable_cache", False) else getattr(self, "cache_dir", None)
            logger.info_once(
                "enable_npugraph_ex is enabled, which will bring graph compilation optimization.",
                scope="global",
            )
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
        # Cache file may be absent when the graph was skipped at save time (e.g. it
        # contained Triton kernels whose kernel_side_table indices are process-local
        # and cannot be serialized).  Fall back to a fresh compilation so the Triton
        # kernels are properly registered in the current process.
        if not path or not os.path.exists(path):
            logger.info(
                "npugraph_ex cache miss for key %s (file absent or not saved), recompiling",
                key,
            )
            # Mirror the same pre-processing done in compile(): deepcopy the graph
            # to prevent make_graph_return_tuple from mutating the caller's copy,
            # and re-wrap FakeTensor inputs under the current fake mode to avoid
            # "fake mode mismatch" in aot_module_simplified.
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
            assert hasattr(self, "vllm_config")
            compiled_fn, _ = npugraph_ex_compile(
                graph,
                example_inputs,
                {},
                self.vllm_config,
                ascend_compilation_config,
                compile_range,
                key,
                getattr(self, "cache_dir", None),
            )
            return compiled_fn

        from npugraph_ex.npu_fx_compiler import _CompiledFxArtifacts, _CompiledFxGraph

        with open(path) as f:
            py_code = f.read()
        artifacts = _CompiledFxArtifacts()
        artifacts.py_code = py_code
        logger.info("Loaded npugraph_ex compilation cache from %s", path)
        compiled_fn = cast(Callable[..., Any], _CompiledFxGraph.load_artifacts(artifacts))

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
