# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.model_executor.layers.mamba.mamba_utils import get_conv_copy_spec, is_conv_state_dim_first
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    get_mamba_groups,
    preprocess_mamba_align_fused_kernel,
    validate_mamba_state_copy_funcs,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.utils import model_uses_kpool_indexer
from vllm_ascend.worker.v2.attn_utils import _allocate_int8_cache_tensor, build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext

# Copy block size used by the upstream fused mamba kernels.
_COPY_BLOCK_SIZE = 1024
_PREPROCESS_BLOCK_SIZE = 256
_STATE_COPY_BLOCK_BYTES = 8192


def _copy_state_rows(source, destination, source_rows, destination_rows, valid):
    """Use the existing Mamba byte-copy kernel; page padding is never copied."""
    if source.device.type == "cpu":
        for src, dst, active in zip(source_rows.tolist(), destination_rows.tolist(), valid.tolist()):
            if active:
                destination[dst].copy_(source[src])
        return
    from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel

    src = source.data_ptr() + source_rows.long() * source.stride(0) * source.element_size()
    dst = destination.data_ptr() + destination_rows.long() * destination.stride(0) * destination.element_size()
    sizes = valid.long() * prod(source.shape[1:]) * source.element_size()
    batch_memcpy_kernel[(sizes.numel(),)](src, dst, sizes, BLOCK_SIZE=_STATE_COPY_BLOCK_BYTES)


@dataclass
class _LayerwiseMambaCopyState:
    """Per-step bookkeeping for the deferred mamba align pre-copy.

    With a layerwise KV pool connector, each mamba layer's running-state copy
    must wait until that layer's remote load has landed. The copies stay
    GPU-resident (same fused kernel as the bulk path) and are launched from
    ``AscendMambaHybridModelState.do_mamba_copy_for_layer`` by the connector's
    ``wait_for_layer_load`` hook.
    """

    ctx: MambaSpecDecodeGPUContext
    num_reqs: int
    idx_mapping: torch.Tensor
    pending_layers: set[str]


class AscendMambaHybridModelState(MambaHybridModelState, AscendModelState):
    """Mamba state with Ascend-specific attention metadata construction.

    Mamba request lifecycle and cache-state handling are inherited from
    :class:`MambaHybridModelState`. ``AscendModelState`` remains the second
    base so cooperative ``super()`` calls retain the Ascend model-state MRO.

    On top of the upstream behaviour this class supports deferring the align
    pre-copy per layer when a layerwise KV transfer connector is active: the
    decision kernel still runs during ``preprocess_state`` (it only advances
    GPU state indices), while the actual state copies execute one layer at a
    time right after that layer's KV load finishes.
    """

    # Annotated locally because mypy (``--follow-imports skip``) does not
    # analyze the upstream base class body where these are defined.
    _mamba_ctx: MambaSpecDecodeGPUContext | None
    _mamba_state_copy_funcs: Any

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._layerwise_mamba_copy: _LayerwiseMambaCopyState | None = None
        self._layer_state_ranges: dict[str, tuple[int, int]] | None = None
        self._pending_bulk_copy: tuple[MambaSpecDecodeGPUContext, int, torch.Tensor] | None = None
        self._mamba_staging: dict[str, list[torch.Tensor]] = {}
        self._mamba_batch: tuple[tuple[str, ...], torch.Tensor, tuple[int, ...], tuple[torch.Tensor, ...]] | None = None
        self._mamba_capture_pending: set[str] = set()
        self._mamba_accepted = False
        self._mamba_staging_failed = False

    def initialize_kv_cache(self, kv_cache_config, block_tables) -> None:
        """Keep bounded candidates alive until acceptance without pinning main KV.

        The existing Mamba copy context performs acceptance on these views;
        layerwise IO publishes the checkpoint views through its normal queue.
        Capacity scales with request slots and speculation width, not context.
        """
        super().initialize_kv_cache(kv_cache_config, block_tables)
        self._mamba_staging = {}
        self._mamba_batch = None
        self._mamba_staging_failed = False
        self._layer_state_ranges = None
        layout = getattr(kv_cache_config, "layerwise_reuse_layout", None)
        if layout is None or not layout.has_layer_reuse or not self.vllm_config.num_speculative_tokens:
            return
        reused = {
            layout.layer_cache_specs[index].main.layer_name
            for index in set(layout.prefetch_layer_map) | set(layout.prefetch_layer_map.values())
        }
        groups = {
            gid
            for gid in self._mamba_group_ids
            if reused.intersection(kv_cache_config.kv_cache_groups[gid].layer_names)
        }
        if not groups:
            return
        assert self._mamba_ctx is not None
        if not self._align_mode or self.recoverssm is not None:
            raise NotImplementedError("Shared speculative state requires the existing Mamba align copy contract")
        if self.vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("Shared speculative state requires acceptance before the next target forward")
        self.layerwise_mamba_groups = groups
        self._mamba_reused = reused
        self._mamba_width = self.vllm_config.num_speculative_tokens + 1
        self._mamba_slots = self._mamba_width + 1  # Checkpoint followed by the ordinary candidate pages.
        context = self.vllm_config.compilation_config.static_forward_context
        drafts = set(getattr(kv_cache_config, "layerwise_draft_layer_names", ()))
        self._mamba_layer_groups = {
            name: gid
            for gid in groups
            for name in kv_cache_config.kv_cache_groups[gid].layer_names
            if name not in drafts
        }
        self._mamba_original_caches = {name: tuple(context[name].kv_cache) for name in self._mamba_layer_groups}
        self._mamba_copy_funcs = {}
        regions, size = [], 0
        for name, caches in self._mamba_original_caches.items():
            spec = mamba_utils._get_mamba_spec_for_layer(
                kv_cache_config.kv_cache_groups[self._mamba_layer_groups[name]], name
            )
            funcs = self._mamba_state_copy_funcs[spec.mamba_type]
            if len(caches) != len(funcs) or any(not cache[0].is_contiguous() for cache in caches):
                raise NotImplementedError("Shared state requires the existing pool's dense per-page copy contract")
            self._mamba_copy_funcs[name] = funcs
            for cache in caches:
                rows = self.max_num_reqs * (self._mamba_slots if name in reused else 1)
                shape = (rows, *cache.shape[1:])
                size = (size + cache.element_size() - 1) // cache.element_size() * cache.element_size()
                end = size + prod(shape) * cache.element_size()
                regions.append((name, size, end, shape, cache.dtype))
                size = end
        alignment = getattr(kv_cache_config, "kv_cache_allocation_alignment", 2 * 1024 * 1024)
        self.layerwise_mamba_backing = _allocate_int8_cache_tensor(size, alignment, self.device)
        for name, start, end, shape, dtype in regions:
            self._mamba_staging.setdefault(name, []).append(
                self.layerwise_mamba_backing[start:end].view(dtype).view(shape)
            )
        self.layerwise_mamba_checkpoints = {
            name: tuple(
                cache.view(self.max_num_reqs, self._mamba_slots, *cache.shape[1:])[:, 0] if name in reused else cache
                for cache in caches
            )
            for name, caches in self._mamba_staging.items()
        }
        self._mamba_valid = torch.zeros(self.max_num_reqs, dtype=torch.bool, device=self.device)
        self._mamba_checkpoint_tokens = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=self.device)
        self._mamba_stage_base = torch.arange(self.max_num_reqs, device=self.device) * self._mamba_slots
        self._mamba_stage_table = torch.empty(
            (self.max_num_reqs, self._mamba_ctx.block_table_stride_req), dtype=torch.int32, device=self.device
        )
        # Rebind only the copy context. Attention still uses the planned shared
        # main buffers, and resident/draft state retains its original context.
        ctx = self._mamba_postprocess_ctx = copy(self._mamba_ctx)
        for field in ("state_base_addrs", "state_block_strides", "state_dim_row_stride", "state_group_indices"):
            setattr(ctx, field, getattr(ctx, field).clone())
        ctx.block_table_ptrs = torch.cat(
            (
                ctx.block_table_ptrs,
                torch.tensor([self._mamba_stage_table.data_ptr()], dtype=torch.int64, device=self.device),
            )
        )
        for name, (start, end) in self._get_layer_state_ranges(kv_cache_config).items():
            if name not in reused:
                continue
            staged_caches = self._mamba_staging[name]
            for field, values in (
                ("state_base_addrs", [cache.data_ptr() for cache in staged_caches]),
                ("state_block_strides", [cache.stride(0) * cache.element_size() for cache in staged_caches]),
                ("state_dim_row_stride", [cache.stride(1) * cache.element_size() for cache in staged_caches]),
            ):
                getattr(ctx, field)[start:end].copy_(torch.tensor(values, device=self.device, dtype=torch.int64))
            ctx.state_group_indices[start:end].fill_(self._mamba_ctx.num_groups)
        if not getattr(self, "_layerwise_is_profiling", False):
            get_kv_transfer_group().configure_mamba_state(self)

    def add_request(self, req_index, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        if self._mamba_staging:
            self._mamba_valid[req_index] = False
            self._mamba_checkpoint_tokens[req_index] = 0

    def prepare_mamba_state_load(self, requests) -> None:
        if not self._mamba_staging or self._mamba_batch is None:
            return
        req_ids, _mapping, slots, _tables = self._mamba_batch
        rows = dict(zip(req_ids, slots))
        for request in requests:
            spec = request.load_spec
            if spec is not None and spec.can_load and not spec.layerwise_previous_step:
                self._mamba_valid[rows[request.req_id]] = False
                self._mamba_checkpoint_tokens[rows[request.req_id]] = 0

    def _copy_layer_candidates(self, name, *, restore):
        assert self._mamba_batch is not None
        _req_ids, mapping, _slots, tables = self._mamba_batch
        slots = mapping.long()
        columns = (self._mamba_src_col_gpu if restore else self._mamba_state_idx_gpu)[slots].long()
        table = tables[self._mamba_layer_groups[name]][: len(slots)]
        for cache, staged, func in zip(
            self._mamba_original_caches[name], self._mamba_staging[name], self._mamba_copy_funcs[name]
        ):
            width = 1 if func is get_conv_copy_spec else self._mamba_width
            offsets = torch.arange(width, device=self.device)
            selected = columns[:, None] + offsets
            blocks = table.gather(1, selected.clamp(0, table.shape[1] - 1)).long()
            rows = slots[:, None] * self._mamba_slots + 1 + offsets
            valid = (columns[:, None] >= 0) & (selected < table.shape[1]) & (blocks > 0)
            if restore:
                valid &= self._mamba_valid[slots, None]
                _copy_state_rows(staged, cache, rows.flatten(), blocks.flatten(), valid.flatten())
            else:
                _copy_state_rows(cache, staged, blocks.flatten(), rows.flatten(), valid.flatten())

    def capture_mamba_layer(self, name):
        if name in self._mamba_reused and name in self._mamba_staging:
            self._copy_layer_candidates(name, restore=False)
            self._mamba_capture_pending.remove(name)

    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None) -> None:
        if not self._mamba_staging or self._mamba_batch is None:
            return super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)
        if self._mamba_capture_pending or num_computed_tokens is None:
            raise RuntimeError("Mamba acceptance requires every shared layer and the computed endpoint")
        _req_ids, mapping, _slots, tables = self._mamba_batch
        if idx_mapping.shape != mapping.shape:
            raise RuntimeError("Mamba acceptance batch changed")
        torch._assert_async(torch.all(idx_mapping == mapping), "Mamba acceptance batch changed")
        ctx = self._mamba_ctx
        try:
            self._mamba_ctx = self._mamba_postprocess_ctx
            super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)
        finally:
            self._mamba_ctx = ctx
        # This tracks only which checkpoint may be published. State selection,
        # conv shifts and accepted-count resets are all performed upstream.
        slots = mapping.long()
        computed = num_computed_tokens[slots]
        accepted = max(num_sampled, 1) if isinstance(num_sampled, int) else num_sampled.clamp_min(1)
        aligned = computed // self._mamba_spec.block_size * self._mamba_spec.block_size
        ready = (aligned > 0) & (aligned >= computed - accepted + 1)
        column = (aligned // self._mamba_spec.block_size - 1).clamp_min(0).long()
        sources: Sequence[torch.Tensor]
        for name, caches in self._mamba_original_caches.items():
            if name in self._mamba_reused:
                # A same-column align copy updates candidate zero in place.
                source_rows = slots * self._mamba_slots + 1
                mask = ready & (column == self._mamba_state_idx_gpu[slots])
                sources = self._mamba_staging[name]
            else:
                source_rows = (
                    tables[self._mamba_layer_groups[name]][: len(slots)].gather(1, column[:, None]).flatten().long()
                )
                mask = ready & (source_rows > 0)
                sources = caches
            for source, checkpoint in zip(sources, self.layerwise_mamba_checkpoints[name]):
                _copy_state_rows(source, checkpoint, source_rows, slots, mask)
        self._mamba_checkpoint_tokens[slots] = torch.where(ready, aligned, self._mamba_checkpoint_tokens[slots])
        self._mamba_valid[slots] = True
        self._mamba_accepted = True

    def get_mamba_checkpoints(self):
        if self._mamba_batch is None:
            return {}, {}
        if not self._mamba_accepted:
            raise RuntimeError("Mamba checkpoint publication preceded acceptance")
        req_ids, mapping, slots, _tables = self._mamba_batch
        # One bounded readback for ordinary host-side pool metadata planning.
        endpoints = self._mamba_checkpoint_tokens[mapping.long()].cpu().tolist()
        return dict(zip(req_ids, endpoints)), dict(zip(req_ids, slots))

    def finish_mamba_save(self, *, failed=False):
        self._mamba_staging_failed |= failed
        self._mamba_batch = None

    def _get_layer_state_ranges(self, kv_cache_config: KVCacheConfig) -> dict[str, tuple[int, int]]:
        """Map every mamba layer name to its (start, end) range in the
        flattened (layer, state-type) metadata arrays of
        ``MambaSpecDecodeGPUContext`` (same iteration order as
        ``initialize_from_forward_context``)."""
        if self._layer_state_ranges is None:
            assert self._mamba_spec is not None
            ranges: dict[str, tuple[int, int]] = {}
            idx = 0
            for mamba_group_id in self._mamba_group_ids:
                group = kv_cache_config.kv_cache_groups[mamba_group_id]
                for layer_name in group.layer_names:
                    # Match the per-type rows populated by the existing copy context.
                    spec = mamba_utils._get_mamba_spec_for_layer(group, layer_name)
                    num_state_types = len(self._mamba_state_copy_funcs[spec.mamba_type])
                    ranges[layer_name] = (idx, idx + num_state_types)
                    idx += num_state_types
            self._layer_state_ranges = ranges
        return self._layer_state_ranges

    def _prepare_layerwise_mamba_copy(
        self,
        ctx: MambaSpecDecodeGPUContext,
        num_reqs: int,
        idx_mapping: torch.Tensor,
        kv_cache_config: KVCacheConfig,
    ) -> bool:
        """Hand the deferred pre-copy over to a layerwise-capable connector.

        Returns True when the connector takes over executing the per-layer
        copies (it will call ``do_mamba_copy_for_layer`` from its
        ``wait_for_layer_load``); in that case the bulk pre-copy launch is
        skipped for this step.
        """
        if not has_kv_transfer_group():
            return False
        connector = get_kv_transfer_group()
        prepare_mamba_state_copy = getattr(connector, "prepare_mamba_state_copy", None)
        if not callable(prepare_mamba_state_copy):
            return False
        if not prepare_mamba_state_copy(self):
            return False
        self._layerwise_mamba_copy = _LayerwiseMambaCopyState(
            ctx=ctx,
            num_reqs=num_reqs,
            idx_mapping=idx_mapping,
            pending_layers=set(self._get_layer_state_ranges(kv_cache_config))
            - set(getattr(kv_cache_config, "layerwise_draft_layer_names", ())),
        )
        return True

    def _finish_previous_layerwise_mamba_copy(self) -> None:
        """Validate and drop the previous step's deferral, if any.

        Runs at the start of the next ``preprocess_state`` so the check is
        serialized behind every per-layer hook of the previous step even under
        async scheduling (the sample-path postprocess may lag one step).
        """
        pending = self._layerwise_mamba_copy
        self._layerwise_mamba_copy = None
        if pending is None:
            return
        if pending.pending_layers:
            raise RuntimeError(f"Mamba state copy was not executed for loaded layers: {sorted(pending.pending_layers)}")
        connector = get_kv_transfer_group()
        finish_mamba_state_copy = getattr(connector, "finish_mamba_state_copy", None)
        if callable(finish_mamba_state_copy):
            finish_mamba_state_copy()

    def do_mamba_copy_for_layer(self, layer_name: str) -> None:
        """Execute one layer's deferred align pre-copy.

        Called by the layerwise connector after ``layer_name``'s KV load (the
        layer's conv/ssm state included) has completed. The launch reuses the
        per-request src/dst columns produced by this step's decision kernel
        and slices the GPU-resident metadata arrays down to the layer's own
        (layer, state-type) entries, so no CPU-GPU sync is needed.
        """
        pending = self._layerwise_mamba_copy
        if pending is None or layer_name not in pending.pending_layers:
            return
        state_range = (self._layer_state_ranges or {}).get(layer_name)
        if state_range is None:
            return
        start, end = state_range
        ctx = pending.ctx
        if layer_name in getattr(self, "_mamba_staging", {}) and layer_name in self._mamba_reused:
            self._copy_layer_candidates(layer_name, restore=True)
        # Same kernel and arguments as MambaSpecDecodeGPUContext.run_fused_
        # precopy, restricted to one layer's state rows: the grid's second
        # axis is the state index and the kernel fast-exits per request when
        # src_col < 0 or src_col == dst_col (no copy for that request).
        # Resolved through the mamba_utils module attribute so the NPU-safe
        # kernel installed by patch_mamba_utils is used (upstream's uint64
        # vectorized temporal copy faults the Ascend vector core).
        mamba_utils.precopy_mamba_align_fused_kernel[(pending.num_reqs, end - start)](
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            ctx.block_table_ptrs,
            ctx.block_table_stride_req,
            ctx.state_base_addrs[start:end],
            ctx.state_block_strides[start:end],
            ctx.state_elem_sizes[start:end],
            ctx.state_inner_sizes[start:end],
            ctx.state_conv_widths[start:end],
            ctx.state_group_indices[start:end],
            ctx.state_dim_row_count[start:end],
            ctx.state_dim_row_stride[start:end],
            pending.idx_mapping,
            pending.num_reqs,
            COPY_BLOCK_SIZE=_COPY_BLOCK_SIZE,
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
            HAS_IDX_MAPPING=True,
        )
        pending.pending_layers.discard(layer_name)

    def _ensure_align_ctx(
        self,
        kv_cache_config: KVCacheConfig,
        mamba_group_ids: list[int],
        block_tables: tuple[torch.Tensor, ...],
    ) -> MambaSpecDecodeGPUContext:
        if self._mamba_ctx is None:
            if self._mamba_state_copy_funcs is None:
                mamba_groups = get_mamba_groups(kv_cache_config)
                mamba_types = {spec.mamba_type for spec in mamba_groups}
                copy_funcs = self.model.get_mamba_state_copy_funcs(mamba_types)
                validate_mamba_state_copy_funcs(mamba_groups, copy_funcs)
                self._mamba_state_copy_funcs = copy_funcs
            self._mamba_ctx = MambaSpecDecodeGPUContext.create(
                max_num_reqs=self.max_num_reqs,
                kv_cache_config=kv_cache_config,
                copy_funcs=self._mamba_state_copy_funcs,
                device=self.device,
                make_buffer=lambda n, dtype: CpuGpuBuffer(n, dtype=dtype, device=self.device),
            )
        ctx = self._mamba_ctx
        if not ctx.is_initialized:
            ctx.initialize_from_forward_context(
                kv_cache_config,
                self.vllm_config.compilation_config.static_forward_context,
                self._mamba_state_copy_funcs,
                [block_tables[gid] for gid in mamba_group_ids],
            )
        return ctx

    def preprocess_state(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Migrate each request's mamba state across block boundaries before
        the forward (V1 align semantics, done on GPU).

        Mirrors the upstream flow, but when a layerwise KV pool connector is
        active the actual state copies are deferred: they are launched per
        layer from the connector's ``wait_for_layer_load`` hook so each copy
        only reads state that has already been loaded remotely. Running the
        bulk pre-copy here instead would race with the in-flight layerwise
        loads and read half-loaded state. Non-layerwise connectors can opt in
        to a bulk copy after the runner connector completes pre_forward.
        """
        if not self._align_mode:
            return
        pcp_manager = getattr(self, "pcp_manager", None)
        if pcp_manager is not None and model_uses_kpool_indexer(self.vllm_config.model_config):
            pcp_context = pcp_manager.build_attention_context(input_batch, block_tables)
            input_batch = pcp_context.global_batch
            block_tables = pcp_context.global_block_tables
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return
        self._finish_previous_layerwise_mamba_copy()
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._ensure_align_ctx(kv_cache_config, mamba_group_ids, block_tables)
        if getattr(self, "_mamba_staging", {}):
            if self._mamba_batch is not None or self._mamba_staging_failed:
                raise RuntimeError("Previous Mamba staging buffers still have readers or failed")
            self._mamba_batch = (
                tuple(input_batch.req_ids),
                input_batch.idx_mapping.clone(),
                tuple(int(slot) for slot in input_batch.idx_mapping_np),
                block_tables,
            )
            self._mamba_accepted = False
            self._mamba_capture_pending = set(self._mamba_staging).intersection(self._mamba_reused)

        # The state-advance + pre-copy kernels run every step; they fast-exit
        # per request when src_col < 0 or src_col == dst_col, so no copy
        # happens on steps that don't cross a block boundary.
        grid = ((num_reqs + _PREPROCESS_BLOCK_SIZE - 1) // _PREPROCESS_BLOCK_SIZE,)
        preprocess_mamba_align_fused_kernel[grid](
            input_batch.idx_mapping,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            num_reqs,
            BLOCK_SIZE=_PREPROCESS_BLOCK_SIZE,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )
        if getattr(self, "_mamba_staging", {}):
            slots = input_batch.idx_mapping.long()
            columns = self._mamba_state_idx_gpu[slots, None] + torch.arange(self._mamba_width, device=self.device)
            torch._assert_async(
                torch.all((columns >= 0) & (columns < self._mamba_stage_table.shape[1])),
                "Mamba candidate columns exceed the block table",
            )
            self._mamba_stage_table.copy_(self._mamba_stage_base[:, None])
            self._mamba_stage_table[slots[:, None], columns.long()] = (
                self._mamba_stage_base[slots, None] + 1 + torch.arange(self._mamba_width, device=self.device)
            ).to(torch.int32)
        if self._prepare_layerwise_mamba_copy(ctx, num_reqs, input_batch.idx_mapping, kv_cache_config):
            # Per-layer copies are executed by the connector right after each
            # layer's KV load finishes (do_mamba_copy_for_layer).
            return
        if has_kv_transfer_group() and getattr(get_kv_transfer_group(), "requires_mamba_state_copy_after_load", False):
            # Sync KV loads run later in KVConnector.pre_forward. Keep the
            # decision kernel above before attention metadata construction,
            # but copy the loaded state only after pre_forward has completed.
            if self._pending_bulk_copy is not None:
                raise RuntimeError("Previous Mamba bulk copy was not consumed")
            self._pending_bulk_copy = (ctx, num_reqs, input_batch.idx_mapping)
            return
        ctx.run_fused_precopy(
            num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
        )

    def copy_state_after_kv_load(self) -> None:
        """Run bulk pre-copy before eager execution or FULL graph replay."""
        pending = self._pending_bulk_copy
        if pending is None:
            return
        self._pending_bulk_copy = None
        ctx, num_reqs, idx_mapping = pending
        ctx.run_fused_precopy(
            num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            idx_mapping,
        )

    def finish_execution(self, *, failed: bool) -> None:
        if failed:
            self._pending_bulk_copy = None
            if getattr(self, "_mamba_staging", {}):
                self.finish_mamba_save(failed=True)
        super().finish_execution(failed=failed)

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        # Match the upstream Mamba contract without enabling DBO.
        assert ubatch_idx == 0, "DBO is not supported on Ascend"
        pcp_manager = getattr(self, "pcp_manager", None)
        pcp_context = None
        if (
            pcp_manager is not None
            and model_uses_kpool_indexer(self.vllm_config.model_config)
            and pcp_manager.is_target_attn_groups(attn_groups)
        ):
            pcp_context = pcp_manager.build_attention_context(input_batch, block_tables, slot_mappings)
        if pcp_context is not None and not input_batch.is_dummy:
            # PCP turns one prefill request into two rank-local segments. The
            # recurrent state must advance once in original request order.
            local_groups = [
                [group for group in groups if not isinstance(group.kv_cache_spec, MambaSpec)] for groups in attn_groups
            ]
            mamba_groups = [
                [group for group in groups if isinstance(group.kv_cache_spec, MambaSpec)] for groups in attn_groups
            ]
            local_metadata = AscendMambaHybridModelState._build_attn_metadata_for_batch(
                self,
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                local_groups,
                kv_cache_config,
                for_capture,
                use_pcp_padding=True,
                pcp_context=pcp_context,
            )
            global_metadata = AscendMambaHybridModelState._build_attn_metadata_for_batch(
                self,
                pcp_context.global_batch,
                cudagraph_mode,
                pcp_context.global_block_tables,
                pcp_context.global_slot_mappings,
                mamba_groups,
                kv_cache_config,
                for_capture,
                use_pcp_padding=False,
                pcp_context=pcp_context,
            )
            local_metadata.update(global_metadata)
            self.attn_metadata = local_metadata
        else:
            self.attn_metadata = AscendMambaHybridModelState._build_attn_metadata_for_batch(
                self,
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture,
                use_pcp_padding=pcp_context is not None,
                pcp_context=pcp_context,
            )
        if self.recoverssm is not None:
            self.recoverssm.record_step(self.attn_metadata, attn_groups, for_capture=for_capture)
        return self.attn_metadata

    def _build_attn_metadata_for_batch(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool,
        *,
        use_pcp_padding: bool,
        pcp_context: "AscendPCPAttentionContext | None" = None,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            # PCP pads all ranks to the largest local token count. Attention
            # metadata must describe the actual model-input tensor width.
            num_tokens = input_batch.num_tokens_after_padding if use_pcp_padding else input_batch.num_tokens

        # Keep recurrent queries logical while attention inputs include graph padding.
        # TODO: Remove this dummy-padding special case once the common metadata
        # path handles padded attention inputs separately from logical queries.
        num_input_tokens = num_tokens
        if cudagraph_mode == CUDAGraphMode.PIECEWISE and input_batch.is_dummy:
            num_input_tokens = input_batch.num_tokens_after_padding

        is_prefilling_np = input_batch.is_prefilling_np[: input_batch.num_reqs]
        if input_batch.prefill_runs_as_decode_np is not None:
            # vLLM #58400 runs one-token prompt tails as decodes. Their
            # placeholder drafts need rollback-capable decode metadata.
            is_prefilling_np = is_prefilling_np & ~input_batch.prefill_runs_as_decode_np[: input_batch.num_reqs]
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(is_prefilling_np)

        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping[: input_batch.num_reqs]
            ]

            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                is_decode = (
                    input_batch.num_scheduled_tokens[: input_batch.num_reqs]
                    == num_draft_tokens_per_req[: input_batch.num_reqs] + 1
                )
                spec_decode_mask = (num_draft_tokens_per_req[: input_batch.num_reqs] > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    num_draft_tokens_per_req[: input_batch.num_reqs],
                    -1,
                )
                if cudagraph_mode == CUDAGraphMode.FULL and num_reqs > input_batch.num_reqs and spec_decode_mask.all():
                    padded_query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])[input_batch.num_reqs :]
                    expected_query_len = self.vllm_config.num_speculative_tokens + 1
                    if np.all(padded_query_lens == expected_query_len):
                        # Full graph capture represents every padded request as
                        # a speculative decode request. Keep replay on the same
                        # pure-spec GDN path so its persistent state metadata is
                        # refreshed before graph replay.
                        num_decode_draft_tokens_np[input_batch.num_reqs :] = padded_query_lens - 1
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        model_specific_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_actual_reqs=input_batch.num_reqs,
            num_tokens=num_tokens,
            num_actual_tokens=input_batch.num_tokens,
            num_input_tokens=num_input_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=input_batch.num_scheduled_tokens.max().item(),
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            parallel_config=self.vllm_config.parallel_config,
            seq_lens_np=input_batch.seq_lens_np,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            model_specific_attn_metadata=model_specific_metadata,
            pcp_context=pcp_context,
            for_cudagraph_capture=for_capture,
        )
