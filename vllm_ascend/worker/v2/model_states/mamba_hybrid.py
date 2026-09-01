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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    preprocess_mamba_align_fused_kernel,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState

# Copy block size used by the upstream fused mamba kernels.
_COPY_BLOCK_SIZE = 1024
_PREPROCESS_BLOCK_SIZE = 256


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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._layerwise_mamba_copy: _LayerwiseMambaCopyState | None = None
        self._layer_state_ranges: dict[str, tuple[int, int]] | None = None

    def _get_layer_state_ranges(self, kv_cache_config: KVCacheConfig) -> dict[str, tuple[int, int]]:
        """Map every mamba layer name to its (start, end) range in the
        flattened (layer, state-type) metadata arrays of
        ``MambaSpecDecodeGPUContext`` (same iteration order as
        ``initialize_from_forward_context``)."""
        if self._layer_state_ranges is None:
            assert self._mamba_spec is not None
            num_state_types = len(self.model.get_mamba_state_copy_func())
            ranges: dict[str, tuple[int, int]] = {}
            idx = 0
            for mamba_group_id in self._mamba_group_ids:
                layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
                for layer_name in layer_names:
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
            pending_layers=set(self._get_layer_state_ranges(kv_cache_config)),
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
        loads and read half-loaded state.
        """
        if not self._align_mode:
            return
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return
        self._finish_previous_layerwise_mamba_copy()
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._ensure_align_ctx(kv_cache_config, mamba_group_ids, block_tables)

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
        if self._prepare_layerwise_mamba_copy(ctx, num_reqs, input_batch.idx_mapping, kv_cache_config):
            # Per-layer copies are executed by the connector right after each
            # layer's KV load finishes (do_mamba_copy_for_layer).
            return
        ctx.run_fused_precopy(
            num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
        )

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(input_batch.is_prefilling_np)

        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[input_batch.idx_mapping]

            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                is_decode = input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    num_draft_tokens_per_req,
                    -1,
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        model_specific_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        self.attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=input_batch.num_scheduled_tokens.max().item(),
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            seq_lens_np=input_batch.seq_lens_np,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            model_specific_attn_metadata=model_specific_metadata,
            for_cudagraph_capture=for_capture,
        )
        return self.attn_metadata
