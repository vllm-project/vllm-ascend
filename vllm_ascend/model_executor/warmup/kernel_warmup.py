# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up Triton kernels used during model execution on Ascend NPU."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.model_executor.warmup.indexer_triton_warmup import (
    indexer_triton_warmup,
)
from vllm_ascend.model_executor.warmup.penalties_triton_warmup import (
    penalties_triton_warmup,
)
from vllm_ascend.model_executor.warmup.rejection_sampler_triton_warmup import (
    rejection_sampler_triton_warmup,
)
from vllm_ascend.model_executor.warmup.rms_triton_warmup import triton_rms_warmup

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


def _run_warmup(
    name: str,
    warmup_fn: Callable[[NPUWorker], None],
    worker: NPUWorker,
) -> None:
    start = time.perf_counter()
    warmup_fn(worker)
    elapsed = time.perf_counter() - start
    logger.info("%s Triton warmup complete in %.3fs.", name, elapsed)


def _register_dcp_warmup(worker: NPUWorker) -> None:
    """Bind target and draft attention owners after their group/layout setup."""
    from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl
    from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl
    from vllm_ascend.ops.triton.dcp.dcp_warmup import register_dcp_warmup

    # Draft layers are also published in the runner's static forward context.
    # Read each implementation's own config, including its draft dtype/capacity.
    seen = set()
    with worker.model_runner.jit_warmup_registry.activate():
        for layer in worker.vllm_config.compilation_config.static_forward_context.values():
            impl = getattr(layer, "impl", None)
            if not isinstance(impl, (AscendSFADCPImpl, AscendMlaDCPImpl)) or impl.dcp_size <= 1:
                continue
            mla = isinstance(impl, AscendMlaDCPImpl)
            # Physical PCP groups remain present for replicated MTP drafts,
            # even when that draft's logical PCP config is disabled.
            pcp_size = impl.pcp_group.world_size
            scatter_size = impl.dcp_size
            if pcp_size > 1:
                scatter_size = impl.tp_group.world_size if impl.dcp_size > pcp_size else 1
            config = impl.vllm_config
            scheduler = config.scheduler_config
            speculative = config.speculative_config
            query_len = 1 + (speculative.num_speculative_tokens or 0) if speculative is not None else 1
            layout = (
                config.model_config.dtype,
                impl.num_heads if mla else impl.local_num_heads,
                impl.kv_lora_rank,
                scatter_size,
                pcp_size,
                max(scheduler.max_num_batched_tokens, scheduler.max_num_seqs * query_len),
                False if mla else not impl.enable_sparse_sfa_turboquant,
                mla,
            )
            if layout in seen:
                continue
            seen.add(layout)
            dtype, local_heads, head_dim, scatter_size, pcp_size, max_tokens, from_stats, mla = layout
            register_dcp_warmup(
                dtype=dtype,
                local_heads=local_heads,
                head_dim=head_dim,
                scatter_size=scatter_size,
                pcp_size=pcp_size,
                max_tokens=max_tokens,
                from_stats=from_stats,
                mla=mla,
            )


def _prune_unlaunchable_registrations(registry) -> int:
    """Drop registrations whose kernels cannot compile on NPU.

    The upstream base RejectionSampler registers CUDA-only kernels during
    its own ``__init__``, which also runs on NPU inside the v1 runner's
    ``_set_up_drafter`` window; compiling those would fail
    ``registry.warmup()``.
    """
    from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

    pruned = [kernel for kernel in registry._registrations if not isinstance(kernel, AscendTritonJitKernel)]
    for kernel in pruned:
        del registry._registrations[kernel]
    return len(pruned)


def _warmup_registered_kernels(worker: NPUWorker) -> None:
    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend

    registry = getattr(worker.model_runner, "jit_warmup_registry", None)
    if registry is None or not getattr(worker.vllm_config.kernel_config, "enable_jit_warmup", True):
        return
    with registry.activate():
        for layer in worker.vllm_config.compilation_config.static_forward_context.values():
            indexer = getattr(getattr(layer, "impl", None), "indexer", None)
            if isinstance(indexer, AscendSFAIndexerBackend):
                indexer.register_k_cache_warmup()
    _register_dcp_warmup(worker)
    _prune_unlaunchable_registrations(registry)
    if len(registry):
        registry.warmup()


def kernel_warmup(worker: NPUWorker) -> None:
    """Run Triton kernel warmups before ACL graph capture."""
    if not HAS_TRITON:
        return

    logger.info("Starting Triton kernel warmup.")
    start = time.perf_counter()

    _run_warmup("rejection_sampler", rejection_sampler_triton_warmup, worker)
    _run_warmup("penalties", penalties_triton_warmup, worker)
    _run_warmup("rms", triton_rms_warmup, worker)
    _run_warmup("indexer", indexer_triton_warmup, worker)

    _warmup_registered_kernels(worker)
    elapsed = time.perf_counter() - start
    logger.info("Triton kernel warmup finished in %.3fs.", elapsed)
