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
#

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_config import AscendConfig, QuestDecodeConfig
from vllm_ascend.attention.utils import enable_cp
from vllm_ascend.ops.select_attention import quest_prefill_metadata
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

QUEST_PAGE_SIZE = 128
QUEST_HEAD_SIZE = 128
QUEST_MAX_METADATA_BLOCKS_PER_REQ = 6
QUEST_INDEX_ALIGNMENT = 8
QUEST_SPARSE_SELECTED_BLOCK_RATIO_THRESHOLD = 0.5


def get_quest_decode_config(vllm_config: VllmConfig) -> QuestDecodeConfig:
    """Resolve QUEST config without requiring the global AscendConfig singleton."""
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, Mapping):
        return QuestDecodeConfig()
    return QuestDecodeConfig(additional_config.get("quest_decode_config"))


@dataclass(frozen=True)
class QuestSparseDecodeInputs:
    batch_size: int
    selected_k: int
    rounded_selected_k: int
    metadata_block_tables: torch.Tensor
    seq_lens: torch.Tensor
    maxblocks: torch.Tensor
    minblocks: torch.Tensor


@dataclass
class QuestBatchMetadata:
    """Temporary active-batch QUEST handle carried by attention metadata."""

    _manager: "QuestDecodeMetadataManager | None" = None
    quest_enabled_for_batch: bool = False
    batch_size: int = 0
    _seq_lens_cpu: tuple[int, ...] = ()
    _seq_lens: torch.Tensor | None = None
    _metadata_block_tables: torch.Tensor | None = None
    _selected_k: int = 0
    _refresh_layer_ids: set[int] = field(default_factory=set)

    def refresh_layer_after_cache_update(
        self,
        *,
        layer_name: str,
        k_cache: torch.Tensor | None,
        block_tables: torch.Tensor | None,
    ) -> None:
        if not self.quest_enabled_for_batch or self._manager is None:
            return
        self._manager._refresh_layer_after_cache_update(
            self,
            layer_name=layer_name,
            k_cache=k_cache,
            block_tables=block_tables,
        )

    def get_sparse_decode_inputs(
        self,
        layer_name: str,
    ) -> QuestSparseDecodeInputs | None:
        if not self.quest_enabled_for_batch or self._manager is None:
            return None
        return self._manager._get_sparse_decode_inputs(
            self,
            layer_name=layer_name,
        )


class _QuestLayerMetadata:
    """Per-layer QUEST metadata tensors and freshness bookkeeping."""

    def __init__(
        self,
        *,
        layer_name: str,
        max_num_reqs: int,
        maxblocks: torch.Tensor,
        minblocks: torch.Tensor,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        self.layer_name = layer_name
        self.maxblocks = maxblocks
        self.minblocks = minblocks
        self.valid_tokens = np.full(max_num_reqs, -1, dtype=np.int32)
        self.refresh_start_seq_lens = torch.zeros((max_num_reqs,), dtype=torch.int32, device=device)
        self.refresh_start_seq_lens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.refresh_start_seq_lens_cpu = self.refresh_start_seq_lens_cpu_tensor.numpy()
        self.refresh_seq_lens = torch.zeros((max_num_reqs,), dtype=torch.int32, device=device)
        self.refresh_seq_lens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.refresh_seq_lens_cpu = self.refresh_seq_lens_cpu_tensor.numpy()

    def invalidate_rows(self, row_indices: Sequence[int]) -> None:
        for row_idx in row_indices:
            self.valid_tokens[row_idx] = -1

    def prepare(
        self,
        batch_metadata: QuestBatchMetadata,
    ) -> bool:
        num_reqs = batch_metadata.batch_size
        if num_reqs <= 0:
            return False

        self.refresh_start_seq_lens_cpu[:num_reqs].fill(0)
        self.refresh_seq_lens_cpu[:num_reqs].fill(0)
        refresh_required = False
        for row_idx in range(num_reqs):
            seq_len = batch_metadata._seq_lens_cpu[row_idx]
            valid_tokens = int(self.valid_tokens[row_idx])
            stale_or_shrunk = valid_tokens < 0 or valid_tokens > seq_len
            if stale_or_shrunk:
                self.refresh_seq_lens_cpu[row_idx] = seq_len
            else:
                crossed_page_boundary = seq_len // QUEST_PAGE_SIZE > valid_tokens // QUEST_PAGE_SIZE
                if crossed_page_boundary:
                    self.refresh_start_seq_lens_cpu[row_idx] = (valid_tokens // QUEST_PAGE_SIZE) * QUEST_PAGE_SIZE
                    self.refresh_seq_lens_cpu[row_idx] = seq_len
            if self.refresh_seq_lens_cpu[row_idx] > self.refresh_start_seq_lens_cpu[row_idx]:
                refresh_required = True

        if refresh_required:
            self.refresh_start_seq_lens[:num_reqs].copy_(
                self.refresh_start_seq_lens_cpu_tensor[:num_reqs],
                non_blocking=True,
            )
            self.refresh_seq_lens[:num_reqs].copy_(
                self.refresh_seq_lens_cpu_tensor[:num_reqs],
                non_blocking=True,
            )

        return refresh_required

    def commit(self, batch_metadata: QuestBatchMetadata) -> None:
        num_reqs = batch_metadata.batch_size
        if num_reqs <= 0:
            return

        for row_idx in range(num_reqs):
            refreshed_seq_len = int(self.refresh_seq_lens_cpu[row_idx])
            if refreshed_seq_len <= 0:
                continue
            self.valid_tokens[row_idx] = refreshed_seq_len

    def is_ready_for_sparse_decode(self, batch_metadata: QuestBatchMetadata) -> bool:
        num_reqs = batch_metadata.batch_size
        if num_reqs <= 0:
            return False

        for row_idx in range(num_reqs):
            seq_len = batch_metadata._seq_lens_cpu[row_idx]
            valid_tokens = int(self.valid_tokens[row_idx])
            if valid_tokens < 0 or valid_tokens > seq_len:
                return False
            if seq_len // QUEST_PAGE_SIZE > valid_tokens // QUEST_PAGE_SIZE:
                return False
        return True


class QuestDecodeMetadataManager:
    """Central QUEST metadata owner for batch rows and per-layer tensors."""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.pin_memory = pin_memory
        self.metadata_block_tables: torch.Tensor | None = None
        self.max_num_metadata_blocks_per_req = 0
        self.owner_req_ids: list[str | None] = [None] * max_num_reqs
        self.layers: dict[str, _QuestLayerMetadata] = {}
        self.ready = False
        self.topk_pages = 0

    def _reset(self) -> None:
        self.metadata_block_tables = None
        self.max_num_metadata_blocks_per_req = 0
        self.owner_req_ids = [None] * self.max_num_reqs
        self.layers.clear()
        self.ready = False
        self.topk_pages = 0

    def _disable(self, reason: str) -> None:
        logger.warning_once(f"QUEST decode is disabled: {reason}")
        self._reset()

    def _invalidate_rows(self, row_indices: Sequence[int]) -> None:
        if not row_indices:
            return
        seen_layers: set[int] = set()
        for layer_metadata in self.layers.values():
            layer_id = id(layer_metadata)
            if layer_id in seen_layers:
                continue
            seen_layers.add(layer_id)
            layer_metadata.invalidate_rows(row_indices)

    def initialize(
        self,
        *,
        vllm_config: VllmConfig,
        ascend_config: AscendConfig,
        model_config: Any,
        max_encoder_len: int | None,
        max_num_reqs: int,
        device: torch.device,
        use_sparse: bool,
        kv_caches: dict[str, Any],
        shared_kv_cache_layers: dict[str, str],
    ) -> None:
        """Validate and allocate all QUEST metadata for a loaded model."""
        self._reset()
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.owner_req_ids = [None] * max_num_reqs

        attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)

        if not ascend_config.quest_decode_config.enable:
            return
        self.topk_pages = ascend_config.quest_decode_config.topk_pages

        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        if cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs():
            self._disable(
                "full graph execution is enabled, but QUEST decode currently requires "
                "runtime switching between dense and sparse attention paths."
            )
            return

        if get_ascend_device_type() not in {AscendDeviceType.A2, AscendDeviceType.A3}:
            self._disable(
                "current hardware is unsupported, QUEST decode currently supports only "
                "Ascend A2/A3 (ascend910b/ascend910_93)."
            )
            return

        if vllm_config.kv_transfer_config is not None:
            self._disable("kv_transfer_config is set, but QUEST decode requires a local KV cache.")
            return
        if enable_cp():
            self._disable("context parallel is enabled, but QUEST decode requires unsharded request metadata.")
            return
        if ascend_config.xlite_graph_config.enabled:
            self._disable(
                "xLite graph execution is enabled, but QUEST decode only supports the standard v1 decode path."
            )
            return
        if model_config.use_mla:
            self._disable("MLA attention is enabled, but QUEST decode only supports standard v1 attention.")
            return
        if use_sparse:
            self._disable("sparse attention is enabled, but QUEST decode only supports standard v1 attention.")
            return

        max_num_metadata_blocks_per_req = _get_max_num_metadata_blocks_per_req(model_config, max_encoder_len)
        if max_num_metadata_blocks_per_req > QUEST_MAX_METADATA_BLOCKS_PER_REQ:
            self._disable(
                "the configured max_model_len requires more metadata blocks per request "
                f"({max_num_metadata_blocks_per_req}) than the kernel limit "
                f"({QUEST_MAX_METADATA_BLOCKS_PER_REQ})."
            )
            return

        base_layer_k_caches: dict[str, torch.Tensor] = {}
        for layer_name, attn_layer in attn_layers.items():
            if layer_name in shared_kv_cache_layers:
                continue
            if layer_name not in kv_caches:
                self._disable(f"attention layer {layer_name} does not have a local KV cache.")
                return

            impl = getattr(attn_layer, "impl", None)
            if not getattr(impl, "quest_layer_supported", False):
                self._disable(f"attention layer {layer_name} is not QUEST-compatible.")
                return

            kv_cache = kv_caches[layer_name]
            if not isinstance(kv_cache, tuple) or len(kv_cache) < 2:
                self._disable(f"attention layer {layer_name} does not expose a standard KV cache tuple.")
                return

            k_cache = kv_cache[0]
            if not isinstance(k_cache, torch.Tensor) or k_cache.ndim != 4:
                self._disable(f"attention layer {layer_name} has an unsupported key-cache layout.")
                return
            if k_cache.shape[1] != QUEST_PAGE_SIZE or k_cache.shape[-1] != QUEST_HEAD_SIZE:
                self._disable(
                    f"attention layer {layer_name} has block_size={k_cache.shape[1]} and "
                    f"head_size={k_cache.shape[-1]}, but QUEST requires block_size={QUEST_PAGE_SIZE} "
                    f"and head_size={QUEST_HEAD_SIZE}."
                )
                return

            base_layer_k_caches[layer_name] = k_cache

        for layer_name, target_layer_name in shared_kv_cache_layers.items():
            if layer_name not in attn_layers:
                continue
            if target_layer_name not in base_layer_k_caches:
                self._disable(
                    f"attention layer {layer_name} shares KV cache with {target_layer_name}, "
                    "but the target layer is not QUEST-compatible."
                )
                return

        if not base_layer_k_caches:
            self._disable("no compatible v1 attention layers were found with block_size 128 and head_size 128.")
            return

        self.max_num_metadata_blocks_per_req = max_num_metadata_blocks_per_req
        num_metadata_blocks = max_num_reqs * max_num_metadata_blocks_per_req
        self.metadata_block_tables = torch.arange(
            num_metadata_blocks,
            dtype=torch.int32,
            device=device,
        ).view(max_num_reqs, max_num_metadata_blocks_per_req)

        base_layer_metadata: dict[str, _QuestLayerMetadata] = {}
        for layer_name, k_cache in base_layer_k_caches.items():
            metadata_shape = (num_metadata_blocks, QUEST_PAGE_SIZE, k_cache.shape[2], QUEST_HEAD_SIZE)
            maxblocks = torch.zeros(metadata_shape, dtype=k_cache.dtype, device=device)
            layer_metadata = _QuestLayerMetadata(
                layer_name=layer_name,
                max_num_reqs=max_num_reqs,
                maxblocks=maxblocks,
                minblocks=torch.zeros_like(maxblocks),
                device=device,
                pin_memory=self.pin_memory,
            )
            base_layer_metadata[layer_name] = layer_metadata
            self.layers[layer_name] = layer_metadata

        for layer_name, target_layer_name in shared_kv_cache_layers.items():
            if layer_name not in attn_layers:
                continue
            self.layers[layer_name] = base_layer_metadata[target_layer_name]

        self.ready = True

    def prepare_batch(
        self,
        *,
        num_reqs: int,
        req_ids: Sequence[str | None] | None,
        seq_lens_cpu: torch.Tensor | np.ndarray,
        seq_lens: torch.Tensor,
        attn_state: Any,
        max_query_len: int | None,
        block_table_width: int,
    ) -> QuestBatchMetadata:
        if (
            not self.ready
            or self.metadata_block_tables is None
            or req_ids is None
            or getattr(attn_state, "name", None) != "DecodeOnly"
            or max_query_len != 1
            or block_table_width <= 0
        ):
            return QuestBatchMetadata()

        num_reqs = min(num_reqs, len(req_ids), self.max_num_reqs)
        if num_reqs <= 0:
            return QuestBatchMetadata()

        selected_blocks = min(self.topk_pages, block_table_width)
        if not self._should_use_sparse_decode(seq_lens_cpu, num_reqs, selected_blocks):
            return QuestBatchMetadata()

        req_ids_tuple = tuple(req_ids[:num_reqs])
        seq_lens_tuple = tuple(int(seq_lens_cpu[row_idx]) for row_idx in range(num_reqs))

        changed_rows: list[int] = []
        for row_idx in range(self.max_num_reqs):
            req_id = req_ids_tuple[row_idx] if row_idx < num_reqs else None
            if self.owner_req_ids[row_idx] != req_id:
                self.owner_req_ids[row_idx] = req_id
                changed_rows.append(row_idx)
        self._invalidate_rows(changed_rows)

        return QuestBatchMetadata(
            _manager=self,
            quest_enabled_for_batch=True,
            batch_size=num_reqs,
            _seq_lens_cpu=seq_lens_tuple,
            _seq_lens=seq_lens[:num_reqs],
            _metadata_block_tables=self.metadata_block_tables[:num_reqs],
            _selected_k=selected_blocks,
        )

    def _refresh_layer_after_cache_update(
        self,
        batch_metadata: QuestBatchMetadata,
        *,
        layer_name: str,
        k_cache: torch.Tensor | None,
        block_tables: torch.Tensor | None,
    ) -> None:
        if not batch_metadata.quest_enabled_for_batch:
            return
        layer_metadata = self.layers.get(layer_name)
        if (
            layer_metadata is None
            or k_cache is None
            or block_tables is None
            or batch_metadata._metadata_block_tables is None
        ):
            return

        layer_id = id(layer_metadata)
        refresh_required = layer_id in batch_metadata._refresh_layer_ids
        if not refresh_required:
            refresh_required = layer_metadata.prepare(batch_metadata)
            if refresh_required:
                batch_metadata._refresh_layer_ids.add(layer_id)
        if not refresh_required:
            return

        batch_size = batch_metadata.batch_size
        quest_prefill_metadata(
            k_cache=k_cache,
            block_tables=block_tables[:batch_size],
            refresh_start_seq_lens=layer_metadata.refresh_start_seq_lens[:batch_size],
            refresh_seq_lens=layer_metadata.refresh_seq_lens[:batch_size],
            metadata_block_tables=batch_metadata._metadata_block_tables,
            maxblocks=layer_metadata.maxblocks,
            minblocks=layer_metadata.minblocks,
        )
        layer_metadata.commit(batch_metadata)

    def _get_sparse_decode_inputs(
        self,
        batch_metadata: QuestBatchMetadata,
        *,
        layer_name: str,
    ) -> QuestSparseDecodeInputs | None:
        if (
            not batch_metadata.quest_enabled_for_batch
            or batch_metadata.batch_size <= 0
            or batch_metadata._selected_k <= 0
            or batch_metadata._metadata_block_tables is None
            or batch_metadata._seq_lens is None
        ):
            return None

        layer_metadata = self.layers.get(layer_name)
        if layer_metadata is None:
            return None
        if not layer_metadata.is_ready_for_sparse_decode(batch_metadata):
            return None

        return QuestSparseDecodeInputs(
            batch_size=batch_metadata.batch_size,
            selected_k=batch_metadata._selected_k,
            rounded_selected_k=cdiv(batch_metadata._selected_k, QUEST_INDEX_ALIGNMENT) * QUEST_INDEX_ALIGNMENT,
            metadata_block_tables=batch_metadata._metadata_block_tables,
            seq_lens=batch_metadata._seq_lens,
            maxblocks=layer_metadata.maxblocks,
            minblocks=layer_metadata.minblocks,
        )

    @staticmethod
    def _should_use_sparse_decode(
        seq_lens: torch.Tensor | np.ndarray,
        batch_size: int,
        selected_blocks: int,
    ) -> bool:
        if batch_size <= 0 or selected_blocks <= 0:
            return False

        seq_lens_tensor = seq_lens[:batch_size]
        if not isinstance(seq_lens_tensor, torch.Tensor):
            seq_lens_tensor = torch.as_tensor(seq_lens_tensor)

        total_blocks = torch.div(
            seq_lens_tensor + QUEST_PAGE_SIZE - 1,
            QUEST_PAGE_SIZE,
            rounding_mode="floor",
        )
        total_blocks = total_blocks[total_blocks > 0]
        if total_blocks.numel() == 0:
            return False

        selected_blocks_per_req = torch.clamp(total_blocks, max=selected_blocks)
        avg_selected_ratio = (selected_blocks_per_req.float() / total_blocks.float()).mean()
        return bool((avg_selected_ratio < QUEST_SPARSE_SELECTED_BLOCK_RATIO_THRESHOLD).item())


def _get_max_num_metadata_blocks_per_req(model_config: Any, max_encoder_len: int | None) -> int:
    quest_max_model_len = max(model_config.max_model_len, max_encoder_len or 0)
    return cdiv(cdiv(quest_max_model_len, QUEST_PAGE_SIZE), QUEST_PAGE_SIZE)
