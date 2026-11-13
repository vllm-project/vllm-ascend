# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import atexit
import logging
import os
import signal
from collections.abc import Callable

import torch
from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_node_count

from vllm_ascend.ascend_config import AscendCompilationConfig

logger = logging.getLogger(__name__)

# TODO: Temporary workaround. Remove this static kernel blacklist once these ops are supported.
STATIC_KERNEL_BLACKLIST = (
    "TopKV2",  # minimax-m3 910C
    "GenericBlockSparseAttention",  # minimax-m3 910C
    "QuantBatchMatmulV3",  # glm5.2 910C
    "TransposeQuantBatchMatMul",  # dsv4 950
    "QuantLightningIndexer",  # glm5.2 950
    "PoolKeyIndexer",  # glm5.3-flash
)


def initialize_static_kernel_runtime(vllm_config: VllmConfig, *, capture: bool = False) -> None:
    if capture:
        parallel = vllm_config.parallel_config
        if parallel.pipeline_parallel_size > 1:
            raise RuntimeError("Capture-time static preparation currently requires PP=1")
        spec = vllm_config.speculative_config
        if spec is not None and not spec.enforce_eager:
            draft_parallel = spec.draft_parallel_config
            if draft_parallel is not None and draft_parallel.tensor_parallel_size != parallel.tensor_parallel_size:
                raise RuntimeError("Static preparation requires target and draft to use the same TP ranks")

        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            import npugraph_ex
            from npugraph_ex._acl_concrete_graph import static_kernel
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        atexit.unregister(npugraph_ex._finalize)
        if not static_kernel._use_process_isolation:
            raise RuntimeError("Capture-time static kernels require isolated installation support")

    if "LOCAL_WORLD_SIZE" not in os.environ:
        parallel_config = vllm_config.parallel_config
        local_world_size = parallel_config.local_world_size * parallel_config.data_parallel_size_local
        if parallel_config.data_parallel_external_lb:
            world_size = torch.distributed.get_world_size()
            if world_size != parallel_config.world_size_across_dp:
                raise RuntimeError("Static kernels require the complete DP worker world")
            local_world_size, remainder = divmod(world_size, get_node_count())
            if remainder:
                raise RuntimeError("Static kernels require equal worker counts on each node")
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

    from npugraph_ex._acl_concrete_graph.static_kernel import _set_static_kernel_blacklist

    _set_static_kernel_blacklist(list(STATIC_KERNEL_BLACKLIST))


def prepare_static_kernels(forward_fn: Callable[[CUDAGraphMode], None], config: AscendCompilationConfig) -> None:
    from npugraph_ex._acl_concrete_graph.static_kernel import compile_static_kernel

    options = {
        "force_eager": True,
        "static_kernel_compile": True,
        "super_kernel_optimize": config.enable_super_kernel,
        "super_kernel_optimize_options": {"dcci_after_kernel_end": [".*"]},
        "disable_static_kernel_compile_cache": envs.VLLM_DISABLE_COMPILE_CACHE,
    }
    forward_started = False

    def collect() -> None:
        nonlocal forward_started
        forward_started = True
        forward_fn(CUDAGraphMode.NONE)
        torch.npu.synchronize()

    compile_static_kernel(collect, config=options)
    if not forward_started:
        logger.warning("Static preparation skipped collection; running ordinary capture warmup")
        forward_fn(CUDAGraphMode.NONE)


def release_static_kernels() -> None:
    """Release installed kernels after all participating workers have released their graphs."""
    from npugraph_ex._acl_concrete_graph.static_kernel import uninstall_static_kernel

    uninstall_static_kernel()
