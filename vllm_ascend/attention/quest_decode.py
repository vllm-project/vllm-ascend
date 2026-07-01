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
from contextvars import ContextVar
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.ascend_config import AscendConfig, QuestDecodeConfig
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, enable_cp
from vllm_ascend.ops.select_attention import (
    paged_select_attention_out,
    quest_block_select_paged,
    quest_prefill_metadata,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

QUEST_PAGE_SIZE = 128
QUEST_HEAD_SIZE = 128
QUEST_MAX_METADATA_BLOCKS_PER_REQ = 6
QUEST_SPARSE_SELECTED_BLOCK_RATIO_THRESHOLD = 0.5


def get_quest_decode_config(vllm_config: VllmConfig) -> QuestDecodeConfig:
    """Parse QUEST config from vLLM config before the AscendConfig singleton exists.
    Platform backend selection needs this early view to route dense attention through QUEST.
    """
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, Mapping):
        return QuestDecodeConfig()
    return QuestDecodeConfig(additional_config.get("quest_decode_config"))


def _get_max_num_metadata_blocks_per_req(model_config: Any, max_encoder_len: int | None) -> int:
    """Compute the per-request QUEST metadata table width for the largest context.
    This bounds the metadata block table allocation made once per model runner.
    """
    quest_max_model_len = max(model_config.max_model_len, max_encoder_len or 0)
    return cdiv(cdiv(quest_max_model_len, QUEST_PAGE_SIZE), QUEST_PAGE_SIZE)


@dataclass(frozen=True)
class QuestSparseDecodeInputs:
    batch_size: int
    selected_k: int
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

    @property
    def is_enabled(self) -> bool:
        """Return whether this batch can call back into the QUEST manager.
        Both the batch-level sparse decision and manager reference must be present.
        """
        return self.quest_enabled_for_batch and self._manager is not None

    def refresh_layer_after_kv_cache_update(
        self,
        *,
        layer_name: str,
        k_cache: torch.Tensor | None,
        block_tables: torch.Tensor | None,
    ) -> bool:
        """Refresh one layer's QUEST metadata after KV cache writes.
        The boolean result tells the attention impl whether sparse decode may proceed.
        """
        if not self.is_enabled:
            return False
        assert self._manager is not None
        return self._manager._refresh_layer_after_kv_cache_update(
            self,
            layer_name=layer_name,
            k_cache=k_cache,
            block_tables=block_tables,
        )

    def get_sparse_decode_inputs(
        self,
        layer_name: str,
    ) -> QuestSparseDecodeInputs | None:
        """Build sparse decode kernel inputs for one layer in this active batch.
        The manager owns the per-layer tensors, so this wrapper keeps callers batch-centric.
        """
        if not self.is_enabled:
            return None
        assert self._manager is not None
        return self._manager._get_sparse_decode_inputs(
            self,
            layer_name=layer_name,
        )


@dataclass
class QuestAttentionMetadata(AscendMetadata):
    """Ascend attention metadata plus the QUEST active-batch handle."""

    quest_metadata: QuestBatchMetadata = field(default_factory=QuestBatchMetadata)


class _QuestLayerMetadata:
    """Per-layer QUEST metadata tensors and freshness bookkeeping."""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        maxblocks: torch.Tensor,
        minblocks: torch.Tensor,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        """Allocate the min/max metadata tensors and freshness trackers for one layer.
        These buffers let QUEST refresh only the KV pages whose metadata is stale.
        """
        self.maxblocks = maxblocks
        self.minblocks = minblocks
        self.refresh_start_seq_lens = CpuGpuBuffer(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
            pin_memory=pin_memory,
        )
        self.refresh_start_seq_lens.np.fill(-1)
        self.refresh_end_seq_lens = CpuGpuBuffer(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
            pin_memory=pin_memory,
        )

    def invalidate_rows(self, row_indices: Sequence[int]) -> None:
        """Mark reused batch rows stale so their metadata is refreshed.
        Row ownership changes when scheduler slots are reused for different request ids.
        """
        for row_idx in row_indices:
            self.refresh_start_seq_lens.np[row_idx] = -1

    def maybe_prepare_refresh(
        self,
        batch_metadata: QuestBatchMetadata,
    ) -> bool:
        """Compute and upload the token ranges that need metadata refresh for this layer.
        Start lengths are persistent raw-token freshness markers, while end
        lengths are per-call refresh requests. The metadata kernel maps this
        raw range to selector-specific page boundaries.
        """
        num_reqs = batch_metadata.batch_size
        if num_reqs <= 0:
            return False

        self.refresh_end_seq_lens.np[:num_reqs].fill(0)
        refresh_required = False
        for row_idx in range(num_reqs):
            seq_len = batch_metadata._seq_lens_cpu[row_idx]
            refreshed_seq_len = int(self.refresh_start_seq_lens.np[row_idx])
            stale_or_shrunk = refreshed_seq_len < 0 or refreshed_seq_len > seq_len
            if stale_or_shrunk:
                self.refresh_start_seq_lens.np[row_idx] = 0
                refreshed_seq_len = 0
                self.refresh_end_seq_lens.np[row_idx] = seq_len
            else:
                crossed_page_boundary = seq_len // QUEST_PAGE_SIZE > refreshed_seq_len // QUEST_PAGE_SIZE
                if crossed_page_boundary:
                    self.refresh_end_seq_lens.np[row_idx] = seq_len
            if self.refresh_end_seq_lens.np[row_idx] > refreshed_seq_len:
                refresh_required = True

        if refresh_required:
            self.refresh_start_seq_lens.copy_to_gpu(num_reqs)
            self.refresh_end_seq_lens.copy_to_gpu(num_reqs)

        return refresh_required

    def commit(self, batch_metadata: QuestBatchMetadata) -> None:
        """Record refreshed lengths after the metadata refresh kernel completes.
        This advances the persistent start markers that later layer visits use for freshness checks.
        """
        num_reqs = batch_metadata.batch_size
        if num_reqs <= 0:
            return

        for row_idx in range(num_reqs):
            start_seq_len = int(self.refresh_start_seq_lens.np[row_idx])
            end_seq_len = int(self.refresh_end_seq_lens.np[row_idx])
            if end_seq_len > start_seq_len:
                self.refresh_start_seq_lens.np[row_idx] = end_seq_len


class QuestDecodeMetadataManager:
    """Central QUEST metadata owner for batch rows and per-layer tensors."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        ascend_config: AscendConfig,
        max_encoder_len: int | None,
        max_num_reqs: int,
        device: torch.device,
        pin_memory: bool,
        use_sparse: bool,
        kv_caches: dict[str, Any],
        shared_kv_cache_layers: dict[str, str],
    ) -> None:
        """Validate QUEST support and allocate metadata once KV caches exist.
        The runner creates this manager after KV cache initialization because layer shapes come from caches.
        """
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.pin_memory = pin_memory
        self.metadata_block_tables: torch.Tensor | None = None
        self.max_num_metadata_blocks_per_req = 0
        # Tracks which request id currently owns each active-batch row. If the
        # scheduler reuses a row for another request, the per-layer QUEST
        # metadata cached for that row no longer matches the new request.
        self.owner_req_ids: list[str | None] = [None] * max_num_reqs
        self.layer_metadata: list[_QuestLayerMetadata] = []
        self.layer_metadata_by_name: dict[str, _QuestLayerMetadata] = {}
        self._is_disabled = False
        self.ready = False
        self.topk_pages = 0

        if not ascend_config.quest_decode_config.enable:
            return

        max_num_metadata_blocks_per_req = _get_max_num_metadata_blocks_per_req(
            vllm_config.model_config,
            max_encoder_len,
        )
        is_disabled, disable_reason = self._validate_model_support_quest(
            vllm_config=vllm_config,
            ascend_config=ascend_config,
            use_sparse=use_sparse,
            max_num_metadata_blocks_per_req=max_num_metadata_blocks_per_req,
        )
        if is_disabled:
            self._disable(disable_reason)
            return

        layer_inputs = self._collect_quest_k_caches_layer_mappings(
            vllm_config=vllm_config,
            kv_caches=kv_caches,
            shared_kv_cache_layers=shared_kv_cache_layers,
        )
        if layer_inputs is None:
            return
        base_layers, shared_layers = layer_inputs

        self.topk_pages = ascend_config.quest_decode_config.topk_pages
        self.max_num_metadata_blocks_per_req = max_num_metadata_blocks_per_req
        num_metadata_blocks = max_num_reqs * max_num_metadata_blocks_per_req
        self.metadata_block_tables = torch.arange(
            num_metadata_blocks,
            dtype=torch.int32,
            device=device,
        ).view(max_num_reqs, max_num_metadata_blocks_per_req)

        base_layer_metadata: dict[str, _QuestLayerMetadata] = {}
        for layer_name, k_cache in base_layers.items():
            num_kv_heads = k_cache.shape[2]
            metadata_shape = (num_metadata_blocks, QUEST_PAGE_SIZE, num_kv_heads, QUEST_HEAD_SIZE)
            layer_metadata = _QuestLayerMetadata(
                max_num_reqs=max_num_reqs,
                maxblocks=torch.zeros(metadata_shape, dtype=k_cache.dtype, device=device),
                minblocks=torch.zeros(metadata_shape, dtype=k_cache.dtype, device=device),
                device=device,
                pin_memory=pin_memory,
            )
            base_layer_metadata[layer_name] = layer_metadata
            self.layer_metadata.append(layer_metadata)
            self.layer_metadata_by_name[layer_name] = layer_metadata

        for layer_name, target_layer_name in shared_layers.items():
            self.layer_metadata_by_name[layer_name] = base_layer_metadata[target_layer_name]

        self.ready = True

    def _disable(self, reason: str) -> None:
        """Permanently disable QUEST for this model runner and emit one warning.
        Dense attention remains available through the inherited backend path.
        """
        logger.warning_once(f"QUEST decode is disabled: {reason}")
        self._is_disabled = True
        self.ready = False

    def _invalidate_rows(self, row_indices: Sequence[int]) -> None:
        """Propagate row invalidation to every unique layer metadata object.
        Shared-KV layers point at the same object, so iterating this list avoids duplicate work.
        """
        if not row_indices:
            return
        for layer_metadata in self.layer_metadata:
            layer_metadata.invalidate_rows(row_indices)

    def _validate_model_support_quest(
        self,
        *,
        vllm_config: VllmConfig,
        ascend_config: AscendConfig,
        use_sparse: bool,
        max_num_metadata_blocks_per_req: int,
    ) -> tuple[bool, str]:
        """Validate model-wide support and return a disable reason when invalid.
        These are static constraints where selecting sparse decode would be invalid or unsafe.
        """
        model_config = vllm_config.model_config
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        if cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs():
            return True, (
                "full graph execution is enabled, but QUEST decode currently requires "
                "runtime switching between dense and sparse attention paths."
            )

        if get_ascend_device_type() not in {AscendDeviceType.A2, AscendDeviceType.A3}:
            return True, (
                "current hardware is unsupported, QUEST decode currently supports only "
                "Ascend A2/A3 (ascend910b/ascend910_93)."
            )

        if vllm_config.kv_transfer_config is not None:
            return True, "kv_transfer_config is set, but QUEST decode requires a local KV cache."
        if enable_cp():
            return True, "context parallel is enabled, but QUEST decode requires unsharded request metadata."
        if ascend_config.xlite_graph_config.enabled:
            return True, "xLite graph execution is enabled, but QUEST decode only supports the standard v1 decode path."
        if model_config.use_mla:
            return True, "MLA attention is enabled, but QUEST decode only supports standard v1 attention."
        if use_sparse:
            return True, "sparse attention is enabled, but QUEST decode only supports standard v1 attention."
        if max_num_metadata_blocks_per_req > QUEST_MAX_METADATA_BLOCKS_PER_REQ:
            return True, (
                "the configured max_model_len requires more metadata blocks per request "
                f"({max_num_metadata_blocks_per_req}) than the kernel limit "
                f"({QUEST_MAX_METADATA_BLOCKS_PER_REQ})."
            )
        return False, ""

    def _validate_quest_layer_impl(self, layer_name: str, attn_layer: Any) -> bool:
        """Ensure a layer is backed by a QUEST-capable attention impl.
        A mismatch means backend selection or layer replacement has made sparse decode unsafe.
        """
        impl = getattr(attn_layer, "impl", None)
        if not isinstance(impl, QuestAttentionBackendImpl):
            self._disable(
                f"attention layer {layer_name} does not use QuestAttentionBackendImpl; dense attention will be used."
            )
            return False
        if not impl.supports_quest():
            self._disable(f"attention layer {layer_name} is not QUEST-compatible.")
            return False
        return True

    def _get_layer_k_cache(self, layer_name: str, kv_caches: dict[str, Any]) -> torch.Tensor | None:
        """Return a layer key cache if its layout matches QUEST kernels.
        QUEST metadata kernels require the standard local KV tuple. The supported
        key-cache layout is [num_blocks, block_size, num_kv_heads, head_dim].
        """
        if layer_name not in kv_caches:
            self._disable(f"attention layer {layer_name} does not have a local KV cache.")
            return None

        kv_cache = kv_caches[layer_name]
        if not isinstance(kv_cache, tuple) or len(kv_cache) < 2:
            self._disable(f"attention layer {layer_name} does not expose a standard KV cache tuple.")
            return None

        k_cache = kv_cache[0]
        if not isinstance(k_cache, torch.Tensor) or k_cache.ndim != 4:
            self._disable(f"attention layer {layer_name} has an unsupported key-cache layout.")
            return None
        if k_cache.shape[1] != QUEST_PAGE_SIZE or k_cache.shape[-1] != QUEST_HEAD_SIZE:
            self._disable(
                f"attention layer {layer_name} has block_size={k_cache.shape[1]} and "
                f"head_size={k_cache.shape[-1]}, but QUEST requires block_size={QUEST_PAGE_SIZE} "
                f"and head_size={QUEST_HEAD_SIZE}."
            )
            return None
        return k_cache

    def _collect_quest_k_caches_layer_mappings(
        self,
        *,
        vllm_config: VllmConfig,
        kv_caches: dict[str, Any],
        shared_kv_cache_layers: dict[str, str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, str]] | None:
        """Collect base and shared layer KV-cache mappings for QUEST setup.
        The result drives per-layer metadata allocation while preserving shared-KV aliases.
        """
        attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
        base_layers: dict[str, torch.Tensor] = {}
        shared_layers: dict[str, str] = {}

        for layer_name, attn_layer in attn_layers.items():
            if not self._validate_quest_layer_impl(layer_name, attn_layer):
                return None

            target_layer_name = shared_kv_cache_layers.get(layer_name)
            if target_layer_name is not None:
                shared_layers[layer_name] = target_layer_name
                continue

            k_cache = self._get_layer_k_cache(layer_name, kv_caches)
            if k_cache is None:
                return None
            base_layers[layer_name] = k_cache

        for layer_name, target_layer_name in shared_layers.items():
            if target_layer_name not in base_layers:
                self._disable(
                    f"attention layer {layer_name} shares KV cache with {target_layer_name}, "
                    "but the target layer is not QUEST-compatible."
                )
                return None

        if not base_layers:
            self._disable("no compatible v1 attention layers were found with block_size 128 and head_size 128.")
            return None

        return base_layers, shared_layers

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
        """Create active-batch QUEST metadata when sparse decode is valid and worthwhile.
        Ineligible batches return empty metadata so the attention impl falls back to dense attention.
        """
        if (
            self._is_disabled
            or not self.ready
            or self.metadata_block_tables is None
            or req_ids is None
            or attn_state != AscendAttentionState.DecodeOnly
            or max_query_len != 1
            or block_table_width <= 0
        ):
            return QuestBatchMetadata()

        num_reqs = min(num_reqs, len(req_ids), self.max_num_reqs)
        if num_reqs <= 0:
            return QuestBatchMetadata()

        num_selected_blocks = min(self.topk_pages, block_table_width)
        # The selection kernel requires k to be a multiple of 8. topk_pages is already a
        # multiple of 8; this only rounds up when the block-table clamp lowered it. Rounding
        # up is always safe -- selecting more pages never changes attention correctness.
        num_selected_blocks = ((num_selected_blocks + 7) // 8) * 8
        if not self._meets_sparse_selection_ratio(seq_lens_cpu, num_reqs, num_selected_blocks):
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
            _selected_k=num_selected_blocks,
        )

    def _refresh_layer_after_kv_cache_update(
        self,
        batch_metadata: QuestBatchMetadata,
        *,
        layer_name: str,
        k_cache: torch.Tensor | None,
        block_tables: torch.Tensor | None,
    ) -> bool:
        """Refresh one layer's metadata and report if sparse decode can use it.
        Returning False keeps the layer on dense attention instead of using stale or missing metadata.
        """
        layer_metadata = self.layer_metadata_by_name.get(layer_name)
        if layer_metadata is None:
            return False

        if k_cache is None or block_tables is None:
            return False

        # No rows crossed a metadata page boundary, so existing QUEST metadata
        # is still fresh and sparse decode can proceed without a refresh launch.
        if not layer_metadata.maybe_prepare_refresh(batch_metadata):
            return True

        metadata_block_tables = batch_metadata._metadata_block_tables
        assert metadata_block_tables is not None

        batch_size = batch_metadata.batch_size
        quest_prefill_metadata(
            k_cache=k_cache,
            block_tables=block_tables[:batch_size],
            refresh_start_seq_lens=layer_metadata.refresh_start_seq_lens.gpu[:batch_size],
            refresh_end_seq_lens=layer_metadata.refresh_end_seq_lens.gpu[:batch_size],
            metadata_block_tables=metadata_block_tables,
            maxblocks=layer_metadata.maxblocks,
            minblocks=layer_metadata.minblocks,
        )
        layer_metadata.commit(batch_metadata)
        return True

    def _get_sparse_decode_inputs(
        self,
        batch_metadata: QuestBatchMetadata,
        *,
        layer_name: str,
    ) -> QuestSparseDecodeInputs | None:
        """Package already-refreshed layer metadata for the sparse kernel.
        Batch eligibility and freshness are checked before this helper, so it only resolves layer tensors.
        """
        layer_metadata = self.layer_metadata_by_name.get(layer_name)
        if layer_metadata is None:
            return None

        metadata_block_tables = batch_metadata._metadata_block_tables
        seq_lens = batch_metadata._seq_lens
        assert metadata_block_tables is not None
        assert seq_lens is not None

        return QuestSparseDecodeInputs(
            batch_size=batch_metadata.batch_size,
            selected_k=batch_metadata._selected_k,
            metadata_block_tables=metadata_block_tables,
            seq_lens=seq_lens,
            maxblocks=layer_metadata.maxblocks,
            minblocks=layer_metadata.minblocks,
        )

    @staticmethod
    def _meets_sparse_selection_ratio(
        seq_lens: torch.Tensor | np.ndarray,
        batch_size: int,
        selected_blocks: int,
    ) -> bool:
        """Return whether selected pages are sparse enough to justify QUEST decode.
        If the selected-page ratio is too high, dense attention is expected to be the better path.
        """
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
        valid_mask = total_blocks > 0
        safe_total_blocks = torch.where(
            valid_mask,
            total_blocks,
            torch.ones_like(total_blocks),
        )
        selected_blocks_per_req = torch.clamp(safe_total_blocks, max=selected_blocks)
        selected_ratio = selected_blocks_per_req.float() / safe_total_blocks.float()
        selected_ratio = torch.where(
            valid_mask,
            selected_ratio,
            torch.zeros_like(selected_ratio),
        )
        valid_count = valid_mask.sum()
        avg_selected_ratio = selected_ratio.sum() / torch.clamp(valid_count, min=1)
        return bool(((valid_count > 0) & (avg_selected_ratio < QUEST_SPARSE_SELECTED_BLOCK_RATIO_THRESHOLD)).item())


class QuestAttentionBackend(AscendAttentionBackend):
    """Dedicated QUEST backend that falls back to dense Ascend attention."""

    @staticmethod
    def get_impl_cls():
        """Use the QUEST impl so each forward can choose sparse or inherited dense decode.
        Unsupported cases are disabled by the manager and continue through the dense fallback.
        """
        return QuestAttentionBackendImpl

    @staticmethod
    def get_builder_cls():
        """Use the QUEST metadata builder to attach sparse-decode batch state.
        The base Ascend metadata is preserved and QUEST state is added only as an extension.
        """
        return QuestAttentionMetadataBuilder


class QuestAttentionMetadataBuilder(AscendAttentionMetadataBuilder):
    """Build dense Ascend metadata and attach QUEST batch metadata."""

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QuestAttentionMetadata:
        """Build dense Ascend metadata and attach optional QUEST batch state.
        This keeps dense behavior identical while giving the QUEST impl the request ids and sequence lengths it needs.
        """
        base_metadata = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        num_reqs = common_attn_metadata.num_reqs
        if common_attn_metadata._seq_lens_cpu is not None:
            quest_seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            quest_seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            # TODO: Keep a single authoritative seq_lens_cpu source in common
            # attention metadata so QUEST does not need this fallback sync.
            quest_seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        quest_metadata = QuestBatchMetadata()
        quest_manager = common_attn_metadata.quest_manager
        quest_req_ids = common_attn_metadata.quest_req_ids
        block_table = common_attn_metadata.block_table_tensor
        if quest_manager is not None:
            quest_metadata = quest_manager.prepare_batch(
                num_reqs=0 if quest_req_ids is None else len(quest_req_ids),
                req_ids=quest_req_ids,
                seq_lens_cpu=quest_seq_lens_cpu,
                seq_lens=common_attn_metadata.seq_lens,
                attn_state=base_metadata.attn_state,
                max_query_len=common_attn_metadata.max_query_len,
                block_table_width=block_table.shape[1] if block_table is not None else 0,
            )

        metadata_kwargs = {
            field_info.name: getattr(base_metadata, field_info.name) for field_info in dataclass_fields(AscendMetadata)
        }
        return QuestAttentionMetadata(**metadata_kwargs, quest_metadata=quest_metadata)


class QuestAttentionBackendImpl(AscendAttentionBackendImpl):
    """Sparse QUEST decode overlay for the dense Ascend v1 attention impl."""

    _current_layer_name: ContextVar[str | None] = ContextVar(
        "quest_current_layer_name",
        default=None,
    )

    def supports_quest(self) -> bool:
        """Return whether this attention layer shape can run QUEST kernels.
        The manager uses this to disable QUEST early for MLA, CP, sliding-window, cross-attention, or wrong head size.
        """
        return (
            not enable_cp()
            and not self.vllm_config.model_config.use_mla
            and self.head_size == QUEST_HEAD_SIZE
            and self.attn_type != AttentionType.ENCODER_DECODER
            and self.sliding_window is None
            and self.sinks is None
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Store the current layer name while preserving the inherited forward path.
        QUEST needs the layer name in forward_impl because the base AttentionImpl API does not pass it there.
        """
        token = self._current_layer_name.set(layer.layer_name)
        try:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        finally:
            self._current_layer_name.reset(token)

    def forward_quest_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        quest_inputs: QuestSparseDecodeInputs,
    ) -> torch.Tensor:
        """Run QUEST block selection followed by paged sparse attention.
        Inputs are already validated and refreshed, so this method only invokes the sparse kernels.
        """
        if attn_metadata.block_tables is None:
            raise RuntimeError("QUEST decode was selected without block tables.")

        batch_size = quest_inputs.batch_size
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        block_table = block_table[:batch_size]
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q[:batch_size]
        actual_seq_lengths_kv = actual_seq_lengths_kv[:batch_size]

        query = query[:batch_size]
        selected_q_indices = quest_block_select_paged(
            query=query,
            maxblocks=quest_inputs.maxblocks,
            minblocks=quest_inputs.minblocks,
            metadata_block_tables=quest_inputs.metadata_block_tables,
            seq_lens=quest_inputs.seq_lens,
            k=quest_inputs.selected_k,
            tokens_since_metadata_update=0,
        )
        quest_output = output[:batch_size]
        paged_select_attention_out(
            query,
            key,
            value,
            actual_seq_lengths_q,
            actual_seq_lengths_kv,
            block_table,
            selected_q_indices,
            self.num_heads,
            self.scale,
            self.num_kv_heads,
            block_size,
            quest_output,
        )
        return output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        """Try QUEST sparse decode for an eligible decode batch, otherwise fall back dense.
        Graph capture and missing or disabled QUEST metadata deliberately use the inherited dense implementation.
        """
        quest_metadata = getattr(attn_metadata, "quest_metadata", None)
        if _EXTRA_CTX.capturing or not isinstance(quest_metadata, QuestBatchMetadata) or not quest_metadata.is_enabled:
            return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)

        layer_name = self._current_layer_name.get()
        assert layer_name is not None, "QUEST sparse decode requires the current attention layer name."

        # We need to ensure the quest metadata from prefill and previous decode steps is ready
        metadata_ready = quest_metadata.refresh_layer_after_kv_cache_update(
            layer_name=layer_name,
            k_cache=self.key_cache,
            block_tables=attn_metadata.block_tables,
        )

        if metadata_ready:
            quest_inputs = quest_metadata.get_sparse_decode_inputs(layer_name)
            if quest_inputs is not None:
                return self.forward_quest_attention(query, key, value, attn_metadata, output, quest_inputs)

        return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)
