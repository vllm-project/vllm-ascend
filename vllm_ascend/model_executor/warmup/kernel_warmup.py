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
    from vllm.config import VllmConfig

    from vllm_ascend.worker.worker import NPUWorker


def register_v1_sampling_warmup(vllm_config: VllmConfig) -> None:
    """Register the v1 sampling stack's Triton kernels with the active JIT
    warmup registry.

    Called from the v1 model runner inside a registry window; outside a
    window every ``register_warmup`` below is a no-op, which also keeps the
    v2 runner (which never constructs the v1 sampling stack) free of these
    kernels.
    """
    from vllm_ascend.ops.triton.bincount import _TOKEN_BIN_COUNTS_AND_MASK_KERNEL
    from vllm_ascend.ops.triton.penalty import _APPLY_ALL_PENALTIES_KERNEL
    from vllm_ascend.ops.triton.reject_sample import register_rejection_warmup
    from vllm_ascend.ops.triton.spec_decode.utils import (
        _PREPARE_INPUTS_PADDED_KERNEL,
    )

    _APPLY_ALL_PENALTIES_KERNEL.register_warmup(vllm_config)
    _TOKEN_BIN_COUNTS_AND_MASK_KERNEL.register_warmup(vllm_config)
    _PREPARE_INPUTS_PADDED_KERNEL.register_warmup()
    register_rejection_warmup(vllm_config)


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
