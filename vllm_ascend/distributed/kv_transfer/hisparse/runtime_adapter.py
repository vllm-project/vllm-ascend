# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ascend-owned device adapters for the reviewed merged-main HiSparse API.

Keep packed prefill mapping and private Host allocation local. Core owns
resolution, grouped prefetch, mirror completion, invalidation and binding cleanup.
Remove the wrappers when equivalent upstream backend hooks are available.
No scheduler, allocation policy, model-runner timing or global Torch changes.
"""

from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path

import torch

COMMON_RUNTIME_SHA256 = "89b940424066b62bf042bcdade3b7efacf03968d9f449888c12b351c636cb926"


def _ensure_gpu_sources(
    self,
    resident_state_rows: torch.Tensor,
    state_indices: torch.Tensor,
    resident_block_size: int,
) -> None:
    """Resolve which staged rows can be served from the resident cache.

    ``resident_state_rows`` is a resident group's persistent table by
    request state row and ``state_indices`` holds each staged request's
    state row. Computed once per plan and resident group (every layer in
    the group shares the table); non-null resident pages become
    miss_mask=0 rows gathered device-to-device by ``gather_prefill_cache``.
    """
    source_key = (resident_state_rows.data_ptr(), resident_block_size)
    if self.gpu_source_key == source_key:
        return
    block_size = self.block_size
    if resident_block_size <= 0 or block_size % resident_block_size != 0:
        return
    device = self.row_ids.device
    num_unique = self.row_ids.shape[1] // block_size
    host_ids = self.row_ids[0].view(num_unique, block_size)[:, 0] // block_size
    new_bt = self.block_table.to(torch.int64)
    num_rows, num_cols = new_bt.shape
    if num_rows == 0 or state_indices.shape[0] < num_rows:
        return
    resident_block_table = resident_state_rows.index_select(0, state_indices[:num_rows].clamp(min=0))
    # One representative (row, col) per unique host block: any request
    # referencing the block holds an equivalent (refcounted) resident view.
    flat_pos = torch.arange(num_rows * num_cols, device=device)
    rep = torch.full((num_unique,), num_rows * num_cols, device=device, dtype=torch.int64)
    rep.scatter_reduce_(0, new_bt.reshape(-1), flat_pos, reduce="amin", include_self=True)
    rep_row = (rep // num_cols).clamp(max=resident_block_table.shape[0] - 1)
    rep_col = rep % num_cols
    pages_per_block = block_size // resident_block_size
    res_pos = rep_col[:, None] * pages_per_block + torch.arange(pages_per_block, device=device)
    res_cols = resident_block_table.shape[1]
    res_blocks = torch.where(
        res_pos < res_cols,
        torch.gather(
            resident_block_table.to(torch.int64)[rep_row],
            1,
            res_pos.clamp(max=max(res_cols - 1, 0)),
        ),
        torch.zeros_like(res_pos),
    )
    offsets = torch.arange(block_size, device=device)
    per_off_block = res_blocks.index_select(1, offsets // resident_block_size)
    gpu_rows = per_off_block * resident_block_size + (offsets % resident_block_size)[None, :]
    # Block id 0 is the null block in both id spaces.
    valid_rows = self.row_ids.view(num_unique, block_size) >= 0
    hit = (per_off_block > 0) & (host_ids[:, None] > 0) & valid_rows
    self.gpu_row_ids = torch.where(hit, gpu_rows, -1).reshape(1, -1).to(torch.int32)
    self.miss_mask = ((self.gpu_row_ids < 0) & valid_rows.view(1, -1)).int()
    self.gpu_source_key = source_key


def _verify_source(obj, digest):
    from vllm_ascend.distributed.kv_transfer.hisparse.optimization_profile import verify_core_source

    path = Path(inspect.getsourcefile(obj))
    verify_core_source(path, digest)


def install_private_host_allocator():
    """Use pinned private storage at Core's pre-view allocation boundary.

    Remove this exact-source adapter when common HiSparse exposes a backend
    host-allocation hook. Existing shared regions and non-owning backends keep
    their original allocation and registration owners.
    """
    from vllm.v1.hisparse import runtime as common
    from vllm.v1.hisparse.backend import get_hisparse_backend

    original = common.allocate_pinned_host_pool
    if getattr(original, "_ascend_private_host_allocator", False):
        return
    _verify_source(original, COMMON_RUNTIME_SHA256)

    @wraps(original)
    def allocate(size):
        backend = get_hisparse_backend()
        allocator = getattr(backend, "allocate_private_host_pool", None)
        return original(size) if allocator is None else allocator(size)

    allocate._ascend_private_host_allocator = True
    allocate._ascend_original = original
    common.allocate_pinned_host_pool = allocate


def install_call_sites():
    from vllm.v1.hisparse.runtime import HiSparsePrefillStagingPlan

    original = HiSparsePrefillStagingPlan.ensure_gpu_sources
    if not getattr(original, "_ascend_backend_adapter", False):
        _verify_source(original, COMMON_RUNTIME_SHA256)

        # Freeze the captured function; subsequent wrappers use another local.
        ensure_original = original

        @wraps(ensure_original)
        def ensure(plan, resident_state_rows, state_indices, resident_block_size):
            if resident_state_rows.device.type != "npu":
                return ensure_original(plan, resident_state_rows, state_indices, resident_block_size)
            return _ensure_gpu_sources(plan, resident_state_rows, state_indices, resident_block_size)

        ensure._ascend_backend_adapter = True
        ensure._ascend_original = ensure_original
        HiSparsePrefillStagingPlan.ensure_gpu_sources = ensure
