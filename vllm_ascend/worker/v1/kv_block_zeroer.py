#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""V1-only KV block zeroer."""

from collections.abc import Iterable
from typing import Any

import torch
from vllm.utils.math_utils import largest_power_of_2_divisor
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
from vllm_ascend.worker.utils import (
    _component_views_share_slot,
    _is_page_strided_mla_cache,
    _zero_kv_blocks_kernel,
)


class AscendKVBlockZeroer(KVBlockZeroer):
    """Manages efficient zeroing of KV cache blocks via a Triton kernel.

    Call :meth:`init_meta` once after KV caches are allocated to precompute
    segment addresses, then call :meth:`zero_block_ids` each step to zero
    newly-allocated blocks.
    """

    def __init__(self, device: torch.device, pin_memory: bool) -> None:
        self.device = device
        self.pin_memory = pin_memory
        self._meta: tuple[torch.Tensor, torch.Tensor, int, int, int] | None = None
        self._seg_page_strides: torch.Tensor | None = None
        self._id_cap: int = 0
        self._ids_pinned: torch.Tensor | None = None
        self._ids_gpu: torch.Tensor | None = None

    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
    ) -> None:
        """One-time precomputation for zero_block_ids.

        Builds absolute-address table for the Triton zeroing kernel.
        Each entry is the absolute byte address of a segment start on the
        GPU, so segments in different CUDA allocations work correctly.

        Block IDs from the scheduler reference logical blocks whose size
        may differ from the kernel block size (virtual block splitting).
        Each segment's page size accounts for this ratio so that
        ``block_id * page_size_el`` lands at the correct offset.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        """
        seen_ptrs: set[int] = set()
        seg_addrs: list[int] = []
        seg_page_sizes: list[int] = []
        seg_page_strides: list[int] = []

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            assert kernel_bs > 0 and spec.block_size % kernel_bs == 0
            ratio = spec.block_size // kernel_bs

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv_cache = static_forward_context[layer_name].kv_cache
                # Fused MLA is one tensor, and component-major MLA views share
                # one physical page; zero once from the NoPE start. The legacy
                # K/V protocol still zeroes each component separately.
                if _component_views_share_slot(kv_cache, spec):
                    kv_tensors = (kv_cache[0],)
                elif isinstance(kv_cache, torch.Tensor):
                    kv_tensors = (kv_cache,)
                else:
                    kv_tensors = kv_cache
                    if cache_dtype == "mxfp8" and len(kv_tensors) == 4:
                        # V scales are checkpoint constants, initialized before
                        # capture. Clearing a recycled block must preserve them.
                        kv_tensors = kv_tensors[:3]
                for kv in kv_tensors:
                    dp = kv.data_ptr()
                    if dp in seen_ptrs:
                        continue
                    seen_ptrs.add(dp)

                    el = kv.element_size()
                    if _is_page_strided_mla_cache(kv, spec, ratio):
                        physical_page_bytes = kv.stride(0) * el * ratio
                        assert physical_page_bytes % 4 == 0
                        physical_page_el = physical_page_bytes // 4
                        seg_addrs.append(dp)
                        seg_page_sizes.append(physical_page_el)
                        seg_page_strides.append(physical_page_el)
                        continue

                    payload_bytes = kv[0].numel() * el
                    stride_bytes = kv.stride(0) * el
                    assert kv[0].is_contiguous(), "KV block payload must be contiguous"
                    assert payload_bytes % 4 == 0 and stride_bytes % 4 == 0
                    # A physical stride may include other caches and padding.
                    # Only clear the payload. Contiguous subblocks can be
                    # coalesced; strided subblocks need one segment each.
                    contiguous = payload_bytes == stride_bytes
                    for subblock in range(1 if contiguous else ratio):
                        seg_addrs.append(dp + subblock * stride_bytes)
                        seg_page_sizes.append(payload_bytes * (ratio if contiguous else 1) // 4)
                        seg_page_strides.append(stride_bytes * ratio // 4)

        if not seg_addrs:
            self._meta = None
            self._seg_page_strides = None
            return

        # _zero_kv_blocks_kernel will use int64 zeros, to meet the UB size, we use blk_size=64B/8B=8192
        max_page_size_el = max(seg_page_sizes)
        blk_size = min(
            min(largest_power_of_2_divisor(page_size_el) for page_size_el in seg_page_sizes),
            8192,
        )
        self._id_cap = 8192
        self._ids_pinned = torch.empty(
            self._id_cap,
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._ids_gpu = torch.empty(self._id_cap, dtype=torch.int64, device=self.device)
        self._seg_page_strides = torch.tensor(seg_page_strides, dtype=torch.int64, device=self.device)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            max_page_size_el // blk_size,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if not block_ids or self._meta is None:
            return
        seg_addrs, seg_page_sizes, max_chunks, blk_size, n_segs = self._meta
        n_blocks = len(block_ids)
        if n_blocks > self._id_cap:
            self._id_cap = n_blocks * 2
            self._ids_pinned = torch.empty(
                self._id_cap,
                dtype=torch.int64,
                pin_memory=self.pin_memory,
            )
            self._ids_gpu = torch.empty(self._id_cap, dtype=torch.int64, device=self.device)
        assert self._ids_pinned is not None and self._ids_gpu is not None
        self._ids_pinned[:n_blocks].numpy()[:] = block_ids
        idx = self._ids_gpu[:n_blocks]
        idx.copy_(self._ids_pinned[:n_blocks], non_blocking=True)
        total_work = n_blocks * n_segs * max_chunks
        grid = min(total_work, get_vectorcore_num()) if total_work > 0 else 0
        if grid == 0:
            return
        _zero_kv_blocks_kernel[(grid,)](
            seg_addrs,
            seg_page_sizes,
            idx,
            n_blocks,
            self._seg_page_strides if self._seg_page_strides is not None else seg_page_sizes,
            N_SEGS=n_segs,
            MAX_CHUNKS=max_chunks,
            BLOCK_SIZE=blk_size,
            GRID_SIZE=grid,
        )
