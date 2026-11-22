# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""JIT kernel warmup orchestration on Ascend NPU.

Kernels register themselves declaratively (``AscendTritonJitKernel``
wrappers) into the upstream ``JitWarmupRegistry`` during model runner
construction; this module expands those registrations and compiles every
kernel/key pair once, before ACL graph capture.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


def _iter_dspark_draft_groups(runner: Any) -> Any:
    """Yield ``(sliding_window_adapter, attn_group)`` for draft groups.

    The v1 runner keeps its draft attention groups (and the optional
    sliding-window adapter cropping the block table) on the drafter; the v2
    runner stores them on the speculator without an adapter.
    """
    drafter = getattr(runner, "drafter", None)
    if drafter is not None and hasattr(drafter, "draft_attn_groups"):
        adapter = getattr(drafter, "sliding_window", None)
        groups: Any = drafter.draft_attn_groups
    else:
        speculator = getattr(runner, "speculator", None)
        if speculator is None or not hasattr(speculator, "attn_groups"):
            return
        adapter = None
        groups = speculator.attn_groups

    def _flatten(items: Any) -> Any:
        for item in items:
            if isinstance(item, (list, tuple)):
                yield from _flatten(item)
            else:
                yield item

    for group in _flatten(groups):
        yield adapter, group


def _register_dspark_swa_indices_warmup(worker: NPUWorker) -> int:
    """Late-register the DSpark SWA indices kernel for each draft attention
    group; returns the number of groups registered.

    The block-table width (and therefore the JIT key) is only final once the
    draft proposer/speculator and the input batch exist, so this runs here
    rather than from the metadata builder's ``__init__``.
    """
    runner = worker.model_runner
    if not hasattr(runner, "jit_warmup_registry"):
        return 0

    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
    from vllm_ascend.ops.triton.dspark_swa_indices import (
        register_dspark_swa_indices_warmup,
    )

    registered = 0
    with runner.jit_warmup_registry.activate():
        for adapter, attn_group in _iter_dspark_draft_groups(runner):
            builder = attn_group.get_metadata_builder()
            if not isinstance(builder, AscendDSAMetadataBuilder):
                continue
            # Only the device-metadata (graph-capture) path needs pre-capture
            # JIT; the eager path compiles on first call outside any graph.
            if not getattr(builder, "_device_metadata_enabled", False):
                continue
            index_width = getattr(builder, "_dspark_index_width", None)
            if index_width is None:
                continue
            # builder.block_table is assigned per step from the runtime
            # metadata and is still None here. Compute the width the builder
            # will see: the sliding-window adapter crops the table to its
            # window clone, the plain path reads the full per-group table
            # from the input batch.
            if adapter is not None:
                table_width = adapter.max_window_blocks
            else:
                gid = attn_group.kv_cache_group_id
                table_width = runner.input_batch.block_table[gid].get_device_tensor().shape[1]
            registered += int(
                register_dspark_swa_indices_warmup(
                    window_size=builder.model_config.hf_config.sliding_window,
                    block_size=builder.storage_block_size,
                    index_width=index_width,
                    block_table_width=table_width,
                    num_speculative_tokens=builder.speculative_config.num_speculative_tokens,
                )
            )
    return registered


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


def kernel_warmup(worker: NPUWorker) -> None:
    """Expand JIT warmup registrations and compile every kernel/key pair
    before ACL graph capture."""
    if not HAS_TRITON:
        return

    kernel_config = getattr(worker.vllm_config, "kernel_config", None)
    enable_jit_warmup = getattr(kernel_config, "enable_jit_warmup", True)
    if not enable_jit_warmup:
        return

    logger.info("JIT kernel warmup starting.")
    start = time.perf_counter()
    registry = worker.model_runner.jit_warmup_registry

    # Cache-dependent keys are unavailable during model construction. Bindings
    # are final here, before registry compilation and graph capture.
    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend

    with registry.activate():
        for layer in worker.vllm_config.compilation_config.static_forward_context.values():
            indexer = getattr(getattr(layer, "impl", None), "indexer", None)
            if isinstance(indexer, AscendSFAIndexerBackend):
                indexer.register_k_cache_warmup()

    _register_dcp_warmup(worker)
    dspark_registered = _register_dspark_swa_indices_warmup(worker)
    pruned = _prune_unlaunchable_registrations(registry)
    if dspark_registered or pruned:
        logger.info(
            "JIT warmup registrations adjusted: %d dspark group(s) added, %d non-Ascend registration(s) pruned.",
            dspark_registered,
            pruned,
        )

    if len(registry) == 0:
        logger.warning(
            "JIT kernel warmup found no registered kernels; Triton kernels "
            "will compile lazily at runtime. If this run should use Ascend "
            "Triton kernels, check that the model runner held a registry "
            "window open during construction."
        )
        return

    try:
        registry.warmup()
    except Exception:
        logger.exception(
            "JIT kernel warmup failed after %.2fs.",
            time.perf_counter() - start,
        )
        raise
    logger.info("JIT kernel warmup finished in %.2fs.", time.perf_counter() - start)
