"""Standalone SFA backend for Sparse KV offload.

All Sparse KV offload related attention logic lives in this module and is selected by
``AscendSFABackend.get_impl_cls()`` / ``get_builder_cls()`` when
``sparse_kv_offload_config.enabled`` is set, keeping ``sfa_v1.py`` clean.

Data plane (see zsc-sfa-kv-offload-merge-plan.md):

- prefill (debug intermediate state, only reachable with
  ``keep_device_kv_cache=True``): ``exec_kv`` writes the NPU paged main
  cache as usual, then the layer's cache rows are committed D2H
  (``cache_cpu[slot] = cache_npu[slot]``) through the manager;
- decode: no NPU main K/V cache at all (indexer K cache only). The current
  token's K/V is produced compute-only and committed D2H directly; top-k
  misses are loaded H2D into the resident (topk) buffer and a single
  resident SFA attention runs.
- fused_overlap decode (optional via ``use_fused_overlap``): replace resident
  onload + SFA with ``npu_fused_sparse_attention_overlap``, reading full KV
  from the shared CPU pool while keeping a selection buffer on NPU.
"""

from __future__ import annotations

from typing import Any, NamedTuple, TypeVar

import torch
import torch_npu
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    PreprocessType,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    build_valid_topk_mask,
    split_decodes_and_prefills,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT,
    get_sparse_kv_offload_manager,
)
from vllm_ascend.utils import enable_dsa_cp

M = TypeVar("M", bound=AscendSFAMetadata)
_FSA_SELECTION_STATUS_ALIGNMENT = 8


def prepare_copy_sfa_queries(query, query_rope):
    """Prepare contiguous queries for native 8/16/32/64/128-head tiles.

    MLA shares KV across query heads, and softmax is independent per head.
    Zero-filled extra query heads cannot affect the original heads; callers
    must return only the original head range from the kernel output.
    """
    heads = query.shape[1]
    if 1 <= heads < 8:
        padded_query = query.new_zeros((query.shape[0], 8, query.shape[2]))
        padded_rope = query_rope.new_zeros((query_rope.shape[0], 8, query_rope.shape[2]))
        padded_query[:, :heads].copy_(query)
        padded_rope[:, :heads].copy_(query_rope)
        return padded_query, padded_rope
    if heads not in (8, 16, 32, 64, 128):
        raise ValueError(
            f"Generalized copy-SFA serving requires 1–8, 16, 32, 64 or 128 query heads per rank, got {heads}"
        )
    return query.contiguous(), query_rope.contiguous()


class _FusedOverlapDecodeCommonInputs(NamedTuple):
    seq_len_thresholds: torch.Tensor
    current_req_ids: torch.Tensor
    stable_prefix_lens: torch.Tensor
    full_kv_block_table: torch.Tensor
    full_kv_actual_seq: torch.Tensor
    full_q_actual_seq: torch.Tensor


def _fsa_selection_status_stride(topk: int) -> int:
    return (
        (topk + 1 + _FSA_SELECTION_STATUS_ALIGNMENT - 1)
        // _FSA_SELECTION_STATUS_ALIGNMENT
        * _FSA_SELECTION_STATUS_ALIGNMENT
    )


def _check_device_kv_cache_exist() -> None:
    # prefill/mixed handling only exists for single-node PD-colocate debug;
    # a PD-disaggregated decode node never receives prefill batches.
    if not get_ascend_config().sparse_kv_offload_config.keep_device_kv_cache:
        raise RuntimeError(
            "Sparse KV offload received a prefill/mixed batch without "
            "keep_device_kv_cache=True; a PD-disaggregated decode node "
            "only accepts decode requests"
        )


class AscendSFAKVOffloadMetadataBuilder(AscendSFAMetadataBuilder):
    """Fills the offload-specific SFA metadata (decode split + request ids)."""

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls,
            supports_dcp_with_varlen,
        )
        cfg = get_ascend_config().sparse_kv_offload_config
        self.use_nano = cfg.use_nano
        if self.use_nano:
            self._init_nano_metadata_buffers(vllm_config, device)
        kv_transfer_config = vllm_config.kv_transfer_config
        self.is_pd_decode_consumer = (
            kv_transfer_config is not None
            and kv_transfer_config.is_kv_consumer
            and not kv_transfer_config.is_kv_producer
        )

    def _init_nano_metadata_buffers(self, vllm_config, device) -> None:
        cfg = get_ascend_config().sparse_kv_offload_config
        requests = vllm_config.scheduler_config.max_num_seqs + 2
        self.nano_pool_capacity = requests
        # The proposer builds all draft metadata before executing any step.
        # Keep each step's derived tensors alive at distinct, stable addresses.
        # build() uses slot 0; build_for_drafting() selects the later slots.
        speculative_config = vllm_config.speculative_config
        self.nano_metadata_steps = max(1, speculative_config.num_speculative_tokens if speculative_config else 1)
        steps = self.nano_metadata_steps
        self.nano_hot_tokens = cfg.topk_buffer_size
        self.nano_stride_blocks = cfg.topk_buffer_size // 128 + 2
        self.nano_rows = torch.arange(requests, dtype=torch.int32, device=device)
        self.nano_blocks = torch.arange(self.nano_stride_blocks, dtype=torch.int32, device=device)
        self.nano_parts = torch.arange(2, dtype=torch.int64, device=device)
        self.nano_vectors = {
            name: torch.empty((steps, requests), dtype=dtype, device=device)
            for name, dtype in (
                ("query_ends", torch.int32),
                ("seq_lens", torch.int32),
                ("prefix_lens", torch.int32),
                ("cache_tokens", torch.int32),
                ("logical_lens", torch.int32),
                ("pool_entries", torch.int32),
                ("generations", torch.int64),
                ("active", torch.bool),
            )
        }
        self.nano_hbm_block_table = torch.empty(
            (steps, requests, self.nano_stride_blocks), dtype=torch.int32, device=device
        )
        source_blocks = cdiv(vllm_config.model_config.max_model_len, 128)
        self.nano_source_block_table = torch.empty((steps, requests, source_blocks), dtype=torch.int32, device=device)
        tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.nano_token_positions = torch.arange(tokens, dtype=torch.int64, device=device)
        self.nano_device_slots = torch.empty((steps, tokens), dtype=torch.int64, device=device)
        self.nano_token_active = torch.empty((steps, tokens), dtype=torch.bool, device=device)
        self.nano_tail_src = torch.empty((steps, requests, 2), dtype=torch.int64, device=device)
        self.nano_tail_dst = torch.empty_like(self.nano_tail_src)
        self.nano_tail_lengths = torch.empty((steps, requests, 2), dtype=torch.int32, device=device)
        hf_config = vllm_config.model_config.hf_text_config
        self.nano_token_bytes = torch.tensor(
            [hf_config.kv_lora_rank * 2, hf_config.qk_rope_head_dim * 2], dtype=torch.int64, device=device
        ).view(2, 1, 1)
        self.nano_copy_src_offsets = torch.empty((steps, requests * 4), dtype=torch.int64, device=device)
        self.nano_copy_dst_offsets = torch.empty_like(self.nano_copy_src_offsets)
        self.nano_copy_lengths = torch.empty((steps, requests * 4), dtype=torch.int32, device=device)
        self.nano_copy_count = torch.empty((steps, 1), dtype=torch.int32, device=device)

    def _populate_offload_metadata(
        self,
        metadata: AscendSFAMetadata,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int = 0,
    ) -> AscendSFAMetadata:
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            # The D node has already loaded the prompt KV from P. vLLM still
            # marks the one-token boundary step as prefilling because its
            # computed-token count is N - 1, but SFA must execute it through
            # the decode-offload path. Keep colocated producer/debug behavior
            # unchanged so genuine short prefills still populate the cache.
            treat_short_extends_as_decodes=self.is_pd_decode_consumer,
        )
        metadata.num_decodes = num_decodes
        metadata.num_prefills = num_prefills
        metadata.num_decode_tokens = num_decode_tokens
        metadata.req_ids_tensor = common_attn_metadata.req_ids_tensor
        metadata.token_to_req = common_attn_metadata.token_to_req
        metadata.nano_enabled = (
            self.use_nano
            and common_attn_metadata.nano_eligible
            and (num_prefills == 0 or common_attn_metadata.offload_dummy)
            and 1 <= common_attn_metadata.max_query_len <= 7
        )
        if metadata.nano_enabled:
            # Include graph padding in the operator batch. These addresses
            # remain fixed from capture through replay; only device contents
            # change. Real rows are identified by allocation generation, not
            # by a Python branch evaluated while capturing.
            count = common_attn_metadata.num_reqs
            ends = common_attn_metadata.query_start_loc[1 : count + 1]
            starts = common_attn_metadata.query_start_loc[:count]
            widths = ends - starts
            generations = common_attn_metadata.req_topk_buffer_generations
            pools = common_attn_metadata.req_topk_buffer_slots
            if generations is None or pools is None:
                raise RuntimeError("nano offload requires runner-owned request slots and generations")
            active = (generations[:count] >= 0) & (widths > 0)
            # State -3 runs ordinary causal LI on just Q harmless keys for
            # padding. S=Q makes the kernel write a complete safe TopK output;
            # S=0 would skip LI and could leave selections from an earlier
            # active replay in these shared output rows.
            seq_lens = torch.where(active, common_attn_metadata.seq_lens[:count], widths)
            prefix = torch.div((seq_lens - widths).clamp_min(0), 128, rounding_mode="floor") * 128
            cache = torch.where(active, prefix.clamp_max(self.nano_hot_tokens), 2048)
            safe_pools = torch.where(active, pools[:count], self.nano_rows[:count] + self.nano_pool_capacity)
            logical = torch.where(active, cache + seq_lens - prefix, cache)
            for name, value in (
                ("query_ends", ends),
                ("seq_lens", seq_lens),
                ("prefix_lens", prefix),
                ("cache_tokens", cache),
                ("logical_lens", logical),
                ("pool_entries", safe_pools),
                ("generations", generations[:count]),
                ("active", active),
            ):
                buffer = self.nano_vectors[name][draft_index, :count]
                buffer.copy_(value)
                setattr(metadata, "nano_" + name, buffer)
            cache_blocks = cache[:, None] // 128
            blocks = self.nano_blocks[None, :]
            physical = safe_pools[:, None] * self.nano_stride_blocks
            hbm_table = physical + torch.where(
                blocks < cache_blocks,
                blocks,
                self.nano_stride_blocks - 2 + (prefix[:, None] // 128 + blocks - cache_blocks) % 2,
            )
            self.nano_hbm_block_table[draft_index, :count].copy_(hbm_table)
            metadata.nano_hbm_block_table = self.nano_hbm_block_table[draft_index, :count]
            # Main and indexer caches may have different block tables. Only
            # main-KV addresses belong here; LIM receives the indexer's own.
            source = common_attn_metadata.block_table_tensor[:count]
            if self.nano_source_block_table.shape[2] != source.shape[1]:
                # Block-table width is fixed by cache initialization. Allocate
                # once before capture if scheduler padding added a column.
                self.nano_source_block_table = torch.empty(
                    (self.nano_metadata_steps, self.nano_pool_capacity, source.shape[1]),
                    dtype=source.dtype,
                    device=source.device,
                )
            self.nano_source_block_table[draft_index, :count].copy_(source)
            metadata.nano_source_block_table = self.nano_source_block_table[draft_index, :count]
            tail_blocks = prefix[:, None].to(torch.int64) // 128 + self.nano_parts
            source_ids = source.gather(1, tail_blocks.clamp(0, source.shape[1] - 1)).to(torch.int64)
            # Restore previously computed KV only. Every TP rank scatters
            # its current query KV directly into its local tails, so it never
            # races a read of another TP rank's current-token D2H write.
            lengths = (seq_lens[:, None] - widths[:, None] - prefix[:, None] - self.nano_parts * 128).clamp(0, 128)
            lengths = torch.where(active[:, None] & (tail_blocks < source.shape[1]) & (source_ids >= 0), lengths, 0)
            self.nano_tail_src[draft_index, :count].copy_(source_ids.clamp_min(0) * 128)
            self.nano_tail_dst[draft_index, :count].copy_(
                safe_pools[:, None].to(torch.int64) * self.nano_stride_blocks * 128
                + self.nano_hot_tokens
                + tail_blocks % 2 * 128
            )
            self.nano_tail_lengths[draft_index, :count].copy_(lengths)
            metadata.nano_tail_src = self.nano_tail_src[draft_index, :count]
            metadata.nano_tail_dst = self.nano_tail_dst[draft_index, :count]
            metadata.nano_tail_lengths = self.nano_tail_lengths[draft_index, :count]
            descriptor_count = count * 4
            self.nano_copy_src_offsets[draft_index, :descriptor_count].copy_(
                (metadata.nano_tail_src[None] * self.nano_token_bytes).reshape(-1)
            )
            self.nano_copy_dst_offsets[draft_index, :descriptor_count].copy_(
                (metadata.nano_tail_dst[None] * self.nano_token_bytes).reshape(-1)
            )
            self.nano_copy_lengths[draft_index, :descriptor_count].copy_(
                (metadata.nano_tail_lengths[None] * self.nano_token_bytes).reshape(-1)
            )
            self.nano_copy_count[draft_index].fill_(descriptor_count)
            metadata.nano_copy_src_offsets = self.nano_copy_src_offsets[draft_index, :descriptor_count]
            metadata.nano_copy_dst_offsets = self.nano_copy_dst_offsets[draft_index, :descriptor_count]
            metadata.nano_copy_lengths = self.nano_copy_lengths[draft_index, :descriptor_count]
            metadata.nano_copy_count = self.nano_copy_count[draft_index]
            tokens = common_attn_metadata.num_input_tokens
            positions = self.nano_token_positions[:tokens]
            token_rows = torch.searchsorted(ends.contiguous(), positions.to(torch.int32), right=True).clamp_max(
                count - 1
            )
            token_active = active[token_rows] & (positions < ends[-1])
            logical_positions = seq_lens[token_rows] - widths[token_rows] + positions - starts[token_rows]
            # SP token padding in an eager draft can extend beyond the
            # final query end without adding a request row. It also needs a
            # private destination, never the final real request's tail.
            token_pools = torch.where(token_active, safe_pools[token_rows], self.nano_pool_capacity + token_rows)
            device_slots = token_pools.to(torch.int64) * self.nano_stride_blocks * 128
            device_slots = device_slots + self.nano_hot_tokens + logical_positions % 256
            self.nano_device_slots[draft_index, :tokens].copy_(device_slots)
            self.nano_token_active[draft_index, :tokens].copy_(token_active)
            metadata.nano_device_slots = self.nano_device_slots[draft_index, :tokens]
            metadata.nano_token_active = self.nano_token_active[draft_index, :tokens]
            metadata.num_prefills = 0
            metadata.num_decodes = count
            # Query boundaries already have an exact CPU mirror. Only
            # query-row count is read here, never a device/CPU sequence length.
            # Eager SP padding may be wider than the actual packed query batch.
            metadata.num_decode_tokens = int(common_attn_metadata.query_start_loc_cpu[count])
            metadata.attn_state = (
                AscendAttentionState.SpecDecoding
                if common_attn_metadata.max_query_len > 1
                else AscendAttentionState.DecodeOnly
            )
        return metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> AscendSFAMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build, **kwargs)
        return self._populate_offload_metadata(metadata, common_attn_metadata)

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        **kwargs: Any,
    ) -> AscendSFAMetadata:
        metadata = super().build_for_drafting(
            common_attn_metadata,
            draft_index,
            **kwargs,
        )
        return self._populate_offload_metadata(metadata, common_attn_metadata, draft_index)


class AscendSFAKVOffloadImpl(AscendSFAImpl):
    """SFA implementation that routes main MLA K/V through the CPU pool."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **kwargs,
        )
        if enable_dsa_cp():
            raise NotImplementedError("Sparse KV offload currently requires TP without context parallelism")
        if self.enable_sparse_sfa_c8:
            raise NotImplementedError(
                "Sparse KV offload does not support the sparse SFA C8 main "
                "cache; sparse LI C8 is supported for the device-resident "
                "indexer cache."
            )
        self._current_layer_name: str | None = None
        self.block_size = self.vllm_config.cache_config.block_size
        offload_cfg = get_ascend_config().sparse_kv_offload_config
        self.use_fused_overlap = offload_cfg.use_fused_overlap
        self.use_nano = offload_cfg.use_nano
        self.nano_indexer_owner = self
        self._nano_metadata = None
        if self.use_nano:
            if self.enable_sparse_li_c8:
                raise NotImplementedError("MTP C8 LIM is built but nano C8 serving is not enabled yet")
            requests = self.vllm_config.scheduler_config.max_num_seqs + 2
            tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            device = torch.device("npu")
            self.nano_states = torch.full((requests,), -3, dtype=torch.int32, device=device)
            self.nano_last_generation = torch.full((requests * 2,), -1, dtype=torch.int64, device=device)
            self.nano_last_prefix = torch.zeros(requests * 2, dtype=torch.int32, device=device)
            self.nano_last_cache = torch.zeros_like(self.nano_last_prefix)
            if not self.skip_topk:
                source_capacity = cdiv(self.vllm_config.model_config.max_model_len, 128) * 128
                self.nano_slot_map = torch.full(
                    (requests * 2, source_capacity), -(1 << 31), dtype=torch.int32, device=device
                )
                width = 1 + (
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config
                    else 0
                )
                output_tokens = min(tokens, requests * width + self.vllm_config.parallel_config.tensor_parallel_size)
                self.nano_topk_src = torch.zeros((output_tokens, 1, 2048), dtype=torch.int32, device=device)
                self.nano_topk_dst = torch.zeros_like(self.nano_topk_src)
                self.nano_topk_misses = torch.zeros(output_tokens, dtype=torch.int32, device=device)
                self.nano_miss_src = torch.empty((requests, 32768), dtype=torch.int32, device=device)
                self.nano_miss_dst = torch.empty_like(self.nano_miss_src)
                self.nano_misses = torch.zeros(requests, dtype=torch.int32, device=device)
                self.nano_reuse_cache_tokens = torch.empty(requests, dtype=torch.int32, device=device)
                self.nano_reuse_slots = torch.arange(2048, dtype=torch.int32, device=device)
                self.nano_query_scale = None
                self.nano_key_scale = None
            # Descriptor storage belongs to the attention implementation;
            # per-step source/destination geometry is supplied by metadata.
            self.nano_copy_src = torch.empty(requests * 4, dtype=torch.int64, device=device)
            self.nano_copy_dst = torch.empty_like(self.nano_copy_src)
            self.nano_host_bases = None
            self.nano_device_bases = None
        self.lru_resident_capacity = offload_cfg.topk_buffer_size
        self.sfa_sparse_topk = offload_cfg.topk

        if self.lru_resident_capacity % self.block_size != 0:
            raise ValueError(
                "sparse_kv_offload_config.topk_buffer_size must be divisible by "
                f"block_size ({self.block_size}); got {self.lru_resident_capacity}"
            )
        decode_width = 1
        if self.vllm_config.speculative_config is not None:
            decode_width += self.vllm_config.speculative_config.num_speculative_tokens
        self.max_num_topk_rows = min(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            self.vllm_config.scheduler_config.max_num_seqs * decode_width,
        )
        self.selection_kv_block_table: torch.Tensor | None = None
        self.selection_kv_block_status: torch.Tensor | None = None
        self.selection_membership_map: torch.Tensor | None = None
        self.fused_overlap_last_req_ids: torch.Tensor | None = None
        self._fused_overlap_selection_capacity: tuple[int, int, int, int] | None = None

    def _resolve_preprocess_type(self, act_dtype: torch.dtype) -> PreprocessType:
        logger.warning_once(
            "Sparse KV offload requires the native SFA preprocessing path; sfa_prolog_v3/mlapo is disabled."
        )
        return PreprocessType.NATIVE

    @staticmethod
    def _cpu_cache_pair(manager, layer_name: str):
        layer_id = manager._get_offload_layer_id(layer_name)
        if manager.tp_rank != 0:
            return None, None
        return manager.k_caches_cpu[layer_id], manager.v_caches_cpu[layer_id]

    @staticmethod
    def _resident_views(manager, layer_name: str, rows: int):
        layer_id = manager._get_offload_layer_id(layer_name)
        buffer_k = manager.topk_buffers_k[layer_id]
        buffer_v = manager.topk_buffers_v[layer_id]
        pages_per_row = manager.topk_buffer_size // manager.block_size
        resident_pages = rows * pages_per_row
        resident_k = buffer_k.reshape(-1, buffer_k.shape[-2], buffer_k.shape[-1])[
            : rows * manager.topk_buffer_size
        ].view(
            resident_pages,
            manager.block_size,
            buffer_k.shape[-2],
            buffer_k.shape[-1],
        )
        resident_v = buffer_v.reshape(-1, buffer_v.shape[-2], buffer_v.shape[-1])[
            : rows * manager.topk_buffer_size
        ].view(
            resident_pages,
            manager.block_size,
            buffer_v.shape[-2],
            buffer_v.shape[-1],
        )
        return (
            resident_k,
            resident_v,
            manager.current_slots_npu[:rows],
            manager.resident_block_table_npu[:rows],
            manager.resident_query_lens_npu[:rows],
            manager.resident_seq_lens_npu[:rows],
        )

    def _offload_layer_name(self) -> str:
        layer_name = self.layer_name or self._current_layer_name
        if layer_name is None:
            raise RuntimeError("Sparse KV offload requires a bound attention layer name")
        return layer_name

    @staticmethod
    def _is_decode_only(attn_metadata: M) -> bool:
        return (
            attn_metadata.attn_state in (AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding)
            and int(getattr(attn_metadata, "num_prefills", 0) or 0) == 0
            and int(getattr(attn_metadata, "num_decodes", 0) or 0) > 0
        )

    @staticmethod
    def _pad_to_input_tokens(
        attn_output: torch.Tensor,
        num_input_tokens: int,
    ) -> torch.Tensor:
        if attn_output.shape[0] >= num_input_tokens:
            return attn_output
        padded = attn_output.new_zeros(num_input_tokens, *attn_output.shape[1:])
        padded[: attn_output.shape[0]] = attn_output
        return padded

    @staticmethod
    def _in_graph_runtime() -> bool:
        if not is_forward_context_available():
            return False
        forward_context = get_forward_context()
        runtime_mode = getattr(
            forward_context,
            "cudagraph_runtime_mode",
            CUDAGraphMode.NONE,
        )
        return forward_context.capturing or runtime_mode not in (
            None,
            CUDAGraphMode.NONE,
        )

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._current_layer_name = layer_name
        self._nano_metadata = attn_metadata
        if self.use_nano and attn_metadata is not None and not attn_metadata.nano_enabled:
            # The fallback arena overlaps persistent nano rows. Force a cold
            # fill after prefill/mixed/fallback execution, including owners
            # whose source-to-slot maps otherwise still look valid.
            self.nano_last_generation.fill_(-1)
        try:
            return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, output)
        finally:
            self._current_layer_name = None
            self._nano_metadata = None

    def _prepare_indexer_metadata(self, indexer_metadata, attn_metadata) -> None:
        indexer_metadata.topk_selector = (
            self._nano_select if attn_metadata.nano_enabled and not self.skip_topk else None
        )
        if attn_metadata.nano_enabled:
            # Mask cache writes for every inactive graph/dummy query. This
            # operation is captured and uses the replay's active-row tensor.
            slots = indexer_metadata.slot_mapping
            active = attn_metadata.nano_token_active[: slots.numel()]
            indexer_metadata.slot_mapping = torch.where(active, slots, -1)

    def _prepare_nano_lim_state(self, metadata) -> None:
        count = metadata.nano_pool_entries.numel()
        pools = metadata.nano_pool_entries.to(torch.int64)
        generations = metadata.nano_generations
        prefix = metadata.nano_prefix_lens
        cache = metadata.nano_cache_tokens
        ready = (
            (self.nano_last_generation[pools] == generations)
            & (self.nano_last_cache[pools] == cache)
            & (self.nano_last_prefix[pools] <= prefix)
        )
        self.nano_states[:count].copy_(torch.where(metadata.nano_active, torch.where(ready, -1, -2), -3))
        self.nano_last_generation.scatter_(0, pools, generations)
        self.nano_last_prefix.scatter_(0, pools, prefix)
        self.nano_last_cache.scatter_(0, pools, cache)

    def _nano_select(self, query, weights, indexer, indexer_metadata):
        metadata = self._nano_metadata
        count = metadata.nano_pool_entries.numel()
        tokens = metadata.num_decode_tokens
        self._prepare_nano_lim_state(metadata)
        prefix = metadata.nano_prefix_lens
        cache = metadata.nano_cache_tokens
        index_cache = indexer.k_cache.kv_cache[0].view(-1, 128, 1, 128)
        table = indexer_metadata.block_table[:count].contiguous()
        if table.shape[1] * 128 != self.nano_slot_map.shape[1]:
            # The scheduler may pad its logical block-table width. This is
            # fixed after initialization, before any graph capture.
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("nano indexer block-table layout must be initialized before graph capture")
            self.nano_slot_map = torch.full(
                (self.nano_slot_map.shape[0], table.shape[1] * 128), -(1 << 31), dtype=torch.int32, device=query.device
            )
            self.nano_last_generation.fill_(-1)
            self.nano_states[:count].copy_(torch.where(metadata.nano_active, -2, -3))
        if self.nano_key_scale is None:
            self.nano_key_scale = torch.empty(index_cache.shape[:3], dtype=torch.float32, device=query.device)
            self.nano_query_scale = torch.empty(
                (self.nano_topk_src.shape[0], query.shape[1]), dtype=torch.float32, device=query.device
            )
        torch.ops._C_ascend.npu_fused_li_manage_mtp(
            weights[:tokens].contiguous(),
            self.nano_query_scale[:tokens],
            query[:tokens].contiguous(),
            self.nano_key_scale,
            index_cache,
            table,
            metadata.nano_query_ends,
            metadata.nano_seq_lens,
            prefix,
            cache,
            self.nano_states[:count],
            metadata.nano_pool_entries,
            self.nano_slot_map,
            self.nano_topk_src[:tokens],
            self.nano_topk_dst[:tokens],
            self.nano_topk_misses[:tokens],
            self.nano_miss_src[:count],
            self.nano_miss_dst[:count],
            self.nano_misses[:count],
        )
        return self.nano_topk_src[:tokens]

    def bind_nano_kv_cache(self, manager, layer_name) -> None:
        """Bind immutable layer addresses after cache registration, before capture."""
        layer_id = manager._get_offload_layer_id(layer_name)
        device = manager.topk_buffers_k[layer_id].device
        self.nano_host_bases = torch.tensor(
            [manager.k_caches_cpu[layer_id].data_ptr(), manager.v_caches_cpu[layer_id].data_ptr()],
            dtype=torch.int64,
            device=device,
        ).view(2, 1)
        self.nano_device_bases = torch.tensor(
            [manager.topk_buffers_k[layer_id].data_ptr(), manager.topk_buffers_v[layer_id].data_ptr()],
            dtype=torch.int64,
            device=device,
        ).view(2, 1)

    def _nano_restore_tail(self, metadata, manager, layer_name):
        # Byte offsets and lengths are common to every layer and are already
        # prepared by the metadata builder. Only these two base-address adds
        # remain per layer; there is no per-layer mask/cast/length preparation.
        if self.nano_host_bases is None:
            raise RuntimeError("nano KV base addresses must be bound before graph capture")
        count = metadata.nano_copy_src_offsets.numel()
        torch.add(
            metadata.nano_copy_src_offsets.view(2, -1), self.nano_host_bases, out=self.nano_copy_src[:count].view(2, -1)
        )
        torch.add(
            metadata.nano_copy_dst_offsets.view(2, -1),
            self.nano_device_bases,
            out=self.nano_copy_dst[:count].view(2, -1),
        )
        manager.copy_nano_kv(
            self.nano_copy_src[:count], self.nano_copy_dst[:count], metadata.nano_copy_lengths, metadata.nano_copy_count
        )

    def _prepare_nano_reused_topk(self, topk_indices, metadata):
        """Resolve the proposer's compacted MTP indices without running LIM.

        Later draft steps attend to the exact saved selection, as in regular
        SFA. In particular, they must not append a newly advanced dense tail.
        Usually every source is already resident from draft zero. A prefill
        first step (or an invalidated arena) instead fills a compact TopK
        cache once; subsequent drafts reuse it. Its source-to-slot map stays
        coherent so the next LIM invocation can safely reuse or resize it.
        """
        count = metadata.nano_pool_entries.numel()
        if metadata.num_decode_tokens != count:
            raise RuntimeError("MTP index reuse requires one query per request")
        topk = self.nano_reuse_slots.numel()
        pools = metadata.nano_pool_entries.to(torch.int64)
        sources = topk_indices[:count].reshape(count, topk).to(torch.int32)
        source_capacity = self.nano_slot_map.shape[1]
        valid = (
            metadata.nano_active[:, None]
            & (sources >= 0)
            & (sources < metadata.nano_seq_lens[:, None])
            & (sources < source_capacity)
        )
        source_ids = torch.where(valid, sources, 0).to(torch.int64)
        mapping = self.nano_slot_map[pools]
        cached_slots = mapping.gather(1, source_ids)
        cached_count = self.nano_last_cache[pools]
        ready = (self.nano_last_generation[pools] == metadata.nano_generations) & (cached_count >= topk)
        hits = (cached_slots >= 0) & (cached_slots < cached_count[:, None])
        cold = metadata.nano_active & ~(ready & (hits | ~valid).all(dim=1))
        slots = torch.where(cold[:, None], self.nano_reuse_slots[None], cached_slots)
        cache = torch.where(metadata.nano_active & ~cold, cached_count, topk)
        self.nano_reuse_cache_tokens[:count].copy_(cache)
        self.nano_topk_src[:count, 0].copy_(torch.where(valid, sources, -1))
        self.nano_topk_dst[:count, 0].copy_(torch.where(valid, slots, -1))
        misses = torch.where(cold, topk, 0)
        self.nano_topk_misses[:count].copy_(misses)
        self.nano_misses[:count].copy_(misses)
        # FirstFillScatterCopy requires non-negative entries. Invalid TopK
        # padding copies a harmless source to an unused slot, then attention
        # ignores that slot through its -1 sparse index. Inactive rows copy 0.
        self.nano_miss_src[:count, :topk].copy_(source_ids)
        self.nano_miss_dst[:count, :topk].copy_(self.nano_reuse_slots[None])
        invalid_slot = -(1 << 31)
        mapping = torch.where(cold[:, None], invalid_slot, mapping)
        # Padded indices use source zero. Restore its deterministic value
        # after scatter so duplicate padding cannot overwrite a real entry.
        # scatter_reduce falls back to CPU on some supported torch_npu builds.
        zero_slot = torch.maximum(mapping[:, 0], torch.where(valid & (sources == 0), slots, invalid_slot).amax(dim=1))
        mapping.scatter_(1, source_ids, torch.where(valid, slots, invalid_slot))
        mapping[:, 0].copy_(zero_slot)
        self.nano_slot_map.index_copy_(0, pools, mapping)
        self.nano_last_generation.scatter_(0, pools, metadata.nano_generations)
        self.nano_last_cache.scatter_(0, pools, cache)
        self.nano_last_prefix.scatter_(0, pools, metadata.nano_prefix_lens)
        return self.nano_reuse_cache_tokens[:count]

    def _nano_attention(self, query, query_rope, topk_indices, metadata, manager, layer_name):
        tokens = metadata.num_decode_tokens
        count = metadata.nano_pool_entries.numel()
        owner = self.nano_indexer_owner
        reuse_indices = self.skip_topk and owner is self
        if reuse_indices and not self.has_indexer:
            raise RuntimeError("nano shared indexer owner was not bound during model initialization")
        layer_id = manager._get_offload_layer_id(layer_name)
        hbm_k = manager.topk_buffers_k[layer_id].view(-1, 128, 1, self.kv_lora_rank)
        hbm_v = manager.topk_buffers_v[layer_id].view(-1, 128, 1, self.qk_rope_head_dim)
        host_k = manager.k_caches_cpu[layer_id].view(-1, 128, self.kv_lora_rank)
        host_v = manager.v_caches_cpu[layer_id].view(-1, 128, self.qk_rope_head_dim)
        if reuse_indices:
            # The existing proposer already compacted the model's TopK
            # buffer to one sampled row per request. Reconstruct destinations
            # from those logical IDs, never from stale draft-zero row order.
            cache_tokens = self._prepare_nano_reused_topk(topk_indices, metadata)
            logical_lens = cache_tokens  # exact saved selection; no extra tail
        else:
            self._nano_restore_tail(metadata, manager, layer_name)
            cache_tokens = metadata.nano_cache_tokens
            logical_lens = metadata.nano_logical_lens
        heads = query.shape[1]
        q, qr = prepare_copy_sfa_queries(query[:tokens], query_rope[:tokens])
        out = torch.empty_like(q)
        torch.ops._C_ascend.npu_fused_copy_sfa_mtp(
            qr,
            q,
            metadata.nano_query_ends,
            logical_lens,
            cache_tokens,
            owner.nano_topk_dst[:tokens],
            owner.nano_topk_src[:tokens],
            owner.nano_topk_misses[:tokens],
            owner.nano_miss_src[:count],
            owner.nano_miss_dst[:count],
            owner.nano_misses[:count],
            metadata.nano_hbm_block_table,
            metadata.nano_source_block_table,
            hbm_v,
            hbm_k,
            host_v,
            host_k,
            float(self.scale),
            out,
        )
        return torch.where(metadata.nano_token_active[:tokens, None, None], out[:, :heads], 0).contiguous()

    def _compute_kv_only(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode-only KV generation that never touches an NPU paged cache."""
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        assert self.kv_a_layernorm is not None, "kv_a_layernorm must be initialized"
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        rms_in, rope_in = kv_no_split.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_nope_flat, _ = torch_npu.npu_rms_norm(
            rms_in.view(-1, self.kv_lora_rank),
            self.kv_a_layernorm.weight,
            epsilon=self.kv_a_layernorm.variance_epsilon,
        )
        k_nope = k_nope_flat.view(B, N, S, self.kv_lora_rank)
        k_pe = torch_npu.npu_interleave_rope(
            rope_in,
            cos,
            sin,
        )
        return k_nope, k_pe

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
    ):
        if self._is_decode_only(attn_metadata):
            k_nope, k_pe = self._compute_kv_only(kv_no_split, cos, sin)
            manager = get_sparse_kv_offload_manager()
            layer_name = self._offload_layer_name()
            k_cache_cpu, v_cache_cpu = self._cpu_cache_pair(manager, layer_name)
            if attn_metadata.nano_enabled:
                layer_id = manager._get_offload_layer_id(layer_name)
                device_slots = attn_metadata.nano_device_slots
                for cache_tensor, value in (
                    (manager.topk_buffers_k[layer_id], k_nope),
                    (manager.topk_buffers_v[layer_id], k_pe),
                ):
                    rows = cache_tensor.view(-1, cache_tensor.shape[-1])
                    rows.index_copy_(0, device_slots[: value.shape[0]], value.reshape(value.shape[0], -1))
                slots = torch.where(attn_metadata.nano_token_active[: slots.numel()], slots, -1)
            manager.offload_new_kv(
                layer_name=layer_name,
                slot_mapping=slots,
                k_cache_cpu=k_cache_cpu,
                v_cache_cpu=v_cache_cpu,
                k_cache_npu=None,
                v_cache_npu=None,
                k=k_nope,
                v=k_pe,
                has_prefill=False,
                capturing=self._in_graph_runtime(),
            )
            return k_pe, k_nope

        # Prefill / mixed batch (colocate debug only): stage in the NPU paged
        # main cache as usual, then commit the written rows D2H into the
        # shared CPU pool.
        _check_device_kv_cache_exist()
        result = super().exec_kv(kv_no_split, cos, sin, kv_cache, slots, attn_metadata)
        manager = get_sparse_kv_offload_manager()
        layer_name = self._offload_layer_name()
        k_cache_cpu, v_cache_cpu = self._cpu_cache_pair(manager, layer_name)
        manager.offload_new_kv(
            layer_name=layer_name,
            slot_mapping=slots,
            k_cache_cpu=k_cache_cpu,
            v_cache_cpu=v_cache_cpu,
            k_cache_npu=kv_cache[0],
            v_cache_npu=kv_cache[1],
            k=None,
            v=None,
            has_prefill=True,
            capturing=self._in_graph_runtime(),
        )
        return result

    @staticmethod
    def _flatten_pa_cache(cache: torch.Tensor) -> torch.Tensor:
        if cache.dim() == 3:
            return cache
        if cache.dim() == 4:
            return cache.reshape(cache.shape[0], cache.shape[1], cache.shape[2] * cache.shape[3])
        raise RuntimeError(f"PA cache must be 3D or 4D, got shape={tuple(cache.shape)}")

    @staticmethod
    def _to_int32_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        if tensor.dtype == torch.int32 and tensor.device == device:
            return tensor
        return tensor.to(device=device, dtype=torch.int32)

    @staticmethod
    def _get_optional_custom_op(op_name: str):
        for namespace in (
            getattr(torch.ops, "_C_ascend", None),
            getattr(torch.ops, "custom", None),
            torch_npu,
        ):
            if namespace is None:
                continue
            op = getattr(namespace, op_name, None)
            if op is not None:
                return op
        return None

    def _require_custom_op(self, op_name: str):
        op = self._get_optional_custom_op(op_name)
        if op is None:
            raise RuntimeError(
                f"fused_overlap offload requires custom op {op_name}, but it is not registered "
                "in torch.ops._C_ascend, torch.ops.custom, or torch_npu."
            )
        return op

    def _normalize_fused_overlap_topk_indices(
        self,
        topk_indices: torch.Tensor,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        topk_indices = self._to_int32_device(topk_indices, device)
        if topk_indices.dim() == 2:
            topk_indices = topk_indices.unsqueeze(1)
        elif topk_indices.dim() == 4:
            if topk_indices.shape[0] * topk_indices.shape[1] != num_tokens:
                raise RuntimeError(
                    "fused_overlap BSND topk token dimension mismatch: "
                    f"topk_shape={tuple(topk_indices.shape)} num_tokens={num_tokens}"
                )
            topk_indices = topk_indices.reshape(num_tokens, topk_indices.shape[2], topk_indices.shape[3])
        elif topk_indices.dim() != 3:
            raise RuntimeError(
                f"fused_overlap offload expects topk_indices with dim 2/3/4, got shape={tuple(topk_indices.shape)}"
            )
        if topk_indices.shape[0] != num_tokens:
            raise RuntimeError(
                "fused_overlap topk token dimension mismatch: "
                f"topk_shape={tuple(topk_indices.shape)} num_tokens={num_tokens}"
            )
        if topk_indices.shape[1] <= 0 or topk_indices.shape[2] <= 0:
            raise RuntimeError(f"fused_overlap topk shape is invalid: {tuple(topk_indices.shape)}")
        if self.local_num_heads < topk_indices.shape[1] or self.local_num_heads % topk_indices.shape[1] != 0:
            raise RuntimeError(
                "fused_overlap query heads must be a positive multiple of topk heads: "
                f"query_heads={self.local_num_heads} topk_heads={topk_indices.shape[1]}"
            )
        if topk_indices.shape[2] > self.sfa_sparse_topk:
            raise RuntimeError(
                "fused_overlap topk exceeds configured topk: "
                f"topk={topk_indices.shape[2]} configured={self.sfa_sparse_topk}"
            )
        return topk_indices.contiguous()

    def _flatten_selection_buffer(
        self,
        buffer: torch.Tensor,
        *,
        row_count: int,
        blocks_per_row: int,
        name: str,
    ) -> torch.Tensor:
        if buffer.shape[0] < row_count:
            raise RuntimeError(
                f"fused_overlap {name} row capacity is too small: "
                f"required_rows={row_count} buffer_shape={tuple(buffer.shape)}"
            )
        view = buffer[:row_count]
        if view.dim() == 4:
            if view.shape[1] != self.lru_resident_capacity or view.shape[2] != 1:
                raise RuntimeError(
                    f"fused_overlap {name} expects [row, resident_capacity, 1, dim], got shape={tuple(view.shape)}"
                )
            return view.reshape(row_count * blocks_per_row, self.block_size, view.shape[3])
        if view.dim() == 3:
            if view.shape[1] != self.lru_resident_capacity:
                raise RuntimeError(
                    f"fused_overlap {name} expects resident_capacity in dim1, got shape={tuple(view.shape)}"
                )
            return view.reshape(row_count * blocks_per_row, self.block_size, view.shape[2])
        raise RuntimeError(f"fused_overlap {name} must be 3D or 4D, got shape={tuple(view.shape)}")

    def _ensure_fused_overlap_selection_state(
        self,
        *,
        token_count: int,
        topk_head_count: int,
        topk: int,
        cache_blocks_per_row: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_token_capacity = max(self.max_num_topk_rows, token_count)
        cache_topk_head_capacity = max(topk_head_count, 1)
        cache_topk_capacity = max(self.sfa_sparse_topk, topk)
        row_capacity = cache_token_capacity * cache_topk_head_capacity
        selection_block_count = row_capacity * cache_blocks_per_row
        capacity = (
            cache_token_capacity,
            cache_topk_head_capacity,
            cache_topk_capacity,
            cache_blocks_per_row,
        )
        needs_realloc = (
            self.selection_kv_block_table is None
            or self.selection_kv_block_status is None
            or self.selection_membership_map is None
            or self.fused_overlap_last_req_ids is None
            or self._fused_overlap_selection_capacity is None
            or self._fused_overlap_selection_capacity[0] < cache_token_capacity
            or self._fused_overlap_selection_capacity[1] < cache_topk_head_capacity
            or self._fused_overlap_selection_capacity[2] < cache_topk_capacity
            or self._fused_overlap_selection_capacity[3] < cache_blocks_per_row
            or self.selection_kv_block_table.device != device
            or self.selection_kv_block_status.device != device
            or self.selection_membership_map.device.type != "cpu"
            or self.fused_overlap_last_req_ids.device != device
        )
        if needs_realloc:
            if get_forward_context().capturing:
                raise RuntimeError(
                    "fused_overlap selection state must be preallocated before "
                    "NPUGraph capture: "
                    f"required_capacity={capacity} "
                    f"current_capacity={self._fused_overlap_selection_capacity}"
                )
            self.selection_kv_block_table = torch.arange(
                selection_block_count,
                dtype=torch.int32,
                device=device,
            ).reshape(row_capacity, cache_blocks_per_row)
            self.selection_kv_block_status = torch.full(
                (
                    cache_token_capacity,
                    cache_topk_head_capacity,
                    _fsa_selection_status_stride(cache_topk_capacity),
                ),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self.selection_membership_map = get_sparse_kv_offload_manager().allocate_fused_overlap_membership_map(
                row_capacity
            )
            self.fused_overlap_last_req_ids = torch.full(
                (cache_token_capacity,),
                -1,
                dtype=torch.int64,
                device=device,
            )
            self._fused_overlap_selection_capacity = capacity
        assert self.selection_kv_block_table is not None
        assert self.selection_kv_block_status is not None
        assert self.selection_membership_map is not None
        assert self.fused_overlap_last_req_ids is not None
        return (
            self.selection_kv_block_table[:, :cache_blocks_per_row],
            self.selection_kv_block_status[
                :,
                :cache_topk_head_capacity,
                : _fsa_selection_status_stride(topk),
            ],
            self.selection_membership_map,
            self.fused_overlap_last_req_ids[:token_count],
        )

    def _invalidate_fused_overlap_selection_rows(
        self,
        selection_kv_block_status: torch.Tensor,
        selection_membership_map: torch.Tensor,
        last_req_ids: torch.Tensor,
        attn_metadata: M,
        *,
        num_tokens: int,
        num_reqs: int,
        topk_count: int,
        seq_lens: torch.Tensor,
        cum_query_lens: torch.Tensor,
    ) -> None:
        if attn_metadata.token_to_req is None:
            raise RuntimeError("fused_overlap offload requires token_to_req metadata for selection invalidation")
        if attn_metadata.req_ids_tensor is None:
            raise RuntimeError("fused_overlap offload requires req_ids_tensor metadata for selection invalidation")
        device = last_req_ids.device
        token_to_req = attn_metadata.token_to_req[:num_tokens].to(device=device, dtype=torch.long)
        if not get_forward_context().capturing:
            invalid_req_mapping = (token_to_req < 0) | (token_to_req >= num_reqs)
            if bool(invalid_req_mapping.any().item()):
                raise RuntimeError(
                    "fused_overlap token_to_req contains request indices outside decode request range: "
                    f"num_tokens={num_tokens} num_reqs={num_reqs}"
                )
        req_ids = attn_metadata.req_ids_tensor[:num_reqs].to(device=device, dtype=torch.long)
        current_req_ids = req_ids[token_to_req]
        # Row reuse by a different request: drop the whole selection status row.
        changed_rows = last_req_ids != current_req_ids
        selection_kv_block_status.masked_fill_(changed_rows.view(num_tokens, 1, 1), -1)
        membership_rows = selection_membership_map.reshape(num_tokens, -1, selection_membership_map.shape[-1])
        membership_control = membership_rows[
            ...,
            FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT
            + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
        ]
        membership_control.view(torch.int32).masked_fill_(
            changed_rows.view(num_tokens, 1, 1),
            -1,
        )
        last_req_ids.copy_(current_req_ids)

        # Selection status stores absolute topk token indices. History hits can
        # still be reused under MTP; clear entries that:
        # 1) fall in this step's rewritten window [seq_len - q_len, seq_len)
        #    (newly written / spec-reject rewritable tokens), or
        # 2) are out of range for the current seq_len (>= seq_len).
        # The trailing status slot is actual_seq metadata, not a topk index.
        seq_lens = seq_lens[:num_reqs].to(device=device, dtype=torch.long)
        cum_query_lens = cum_query_lens[:num_reqs].to(device=device, dtype=torch.long)
        query_lens = torch.diff(cum_query_lens, prepend=cum_query_lens.new_zeros(1))
        rewrite_start = (seq_lens - query_lens)[token_to_req].view(num_tokens, 1, 1)
        rewrite_end = seq_lens[token_to_req].view(num_tokens, 1, 1)
        topk_status = selection_kv_block_status[..., :topk_count]
        rewritten_hits = (topk_status >= 0) & (topk_status >= rewrite_start) & (topk_status < rewrite_end)
        oob_hits = (topk_status >= 0) & (topk_status >= rewrite_end)
        invalidated_status = rewritten_hits | oob_hits
        topk_status.masked_fill_(invalidated_status, -1)
        membership_control.view(torch.int32).masked_fill_(
            changed_rows.view(num_tokens, 1, 1) | invalidated_status.any(dim=-1, keepdim=True),
            -1,
        )

    def _validate_fused_overlap_mtp_decode_metadata(
        self,
        attn_metadata: M,
        *,
        num_tokens: int,
        num_reqs: int,
        full_q_actual_seq: torch.Tensor,
        full_kv_actual_seq: torch.Tensor,
    ) -> None:
        if attn_metadata.token_to_req is None:
            raise RuntimeError(
                "fused_overlap offload decode requires token_to_req metadata "
                f"(num_tokens={num_tokens} num_reqs={num_reqs})"
            )
        if full_q_actual_seq.numel() != num_reqs:
            raise RuntimeError(
                "fused_overlap full_q_actual_seq must have one entry per decode "
                f"request: got {full_q_actual_seq.numel()} for num_reqs={num_reqs}"
            )
        if full_kv_actual_seq.numel() != num_reqs:
            raise RuntimeError(
                "fused_overlap full_kv_actual_seq must have one entry per decode "
                f"request: got {full_kv_actual_seq.numel()} for num_reqs={num_reqs}"
            )
        if get_forward_context().capturing:
            return
        token_to_req = attn_metadata.token_to_req[:num_tokens]
        if token_to_req.numel() != num_tokens:
            raise RuntimeError(
                f"fused_overlap token_to_req length mismatch: got {token_to_req.numel()} for num_tokens={num_tokens}"
            )
        invalid_req_mapping = (token_to_req < 0) | (token_to_req >= num_reqs)
        if bool(invalid_req_mapping.any().item()):
            raise RuntimeError(
                "fused_overlap token_to_req contains request indices outside "
                f"decode request range: num_tokens={num_tokens} num_reqs={num_reqs}"
            )
        q_cum_last = int(full_q_actual_seq[-1].item())
        if q_cum_last != num_tokens:
            raise RuntimeError(
                "fused_overlap TND full_q_actual_seq must end at num_tokens: "
                f"full_q_actual_seq[-1]={q_cum_last} num_tokens={num_tokens} "
                f"num_reqs={num_reqs}"
            )

    def _flatten_fused_overlap_mtp_to_token_batch(
        self,
        attn_metadata: M,
        *,
        num_tokens: int,
        num_reqs: int,
        full_q_actual_seq: torch.Tensor,
        full_kv_actual_seq: torch.Tensor,
        full_kv_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert attn_metadata.token_to_req is not None
        device = full_q_actual_seq.device
        token_to_req = attn_metadata.token_to_req[:num_tokens].to(device=device, dtype=torch.long)
        req_cum_q = full_q_actual_seq.to(device=device, dtype=torch.long)
        req_seq_lens = full_kv_actual_seq.to(device=device, dtype=torch.long)
        req_q_lens = torch.diff(req_cum_q, prepend=req_cum_q.new_zeros(1))
        token_starts = torch.zeros(num_reqs, dtype=torch.long, device=device)
        if num_reqs > 1:
            token_starts[1:] = req_cum_q[:-1]
        local_offsets = torch.arange(num_tokens, device=device, dtype=torch.long) - token_starts[token_to_req]
        token_kv_lens = (
            (req_seq_lens[token_to_req] - req_q_lens[token_to_req] + local_offsets + 1)
            .to(dtype=torch.int32)
            .contiguous()
        )
        if not get_forward_context().capturing and bool((token_kv_lens <= 0).any().item()):
            raise RuntimeError(
                "fused_overlap MTP flatten produced non-positive per-token kv lenses: "
                f"token_kv_lens={token_kv_lens.detach().cpu().tolist()}"
            )
        token_q_cum = torch.arange(1, num_tokens + 1, device=device, dtype=torch.int32).contiguous()
        token_block_table = full_kv_block_table[token_to_req].contiguous()
        return token_q_cum, token_kv_lens, token_block_table

    def _prepare_fused_overlap_decode_common_inputs(
        self,
        attn_metadata: M,
        *,
        num_tokens: int,
        num_reqs: int,
        topk_indices_decode: torch.Tensor,
        actual_seq_lengths_query_decode: torch.Tensor,
        actual_seq_lengths_key_decode: torch.Tensor,
    ) -> _FusedOverlapDecodeCommonInputs:
        if attn_metadata.token_to_req is None:
            raise RuntimeError("fused_overlap offload requires token_to_req metadata for topk masking")
        if attn_metadata.req_ids_tensor is None:
            raise RuntimeError("fused_overlap offload requires req_ids_tensor metadata")

        forward_context = get_forward_context()
        cache_key = (
            num_tokens,
            num_reqs,
            topk_indices_decode.ndim,
            topk_indices_decode.dtype,
            topk_indices_decode.device,
            attn_metadata.token_to_req.data_ptr(),
            attn_metadata.req_ids_tensor.data_ptr(),
            attn_metadata.block_table.data_ptr(),
            actual_seq_lengths_query_decode.data_ptr(),
            actual_seq_lengths_key_decode.data_ptr(),
        )
        cache = getattr(
            forward_context,
            "_fsa_offload_decode_common_inputs",
            None,
        )
        if cache is None:
            cache = {}
            forward_context._fsa_offload_decode_common_inputs = cache
        cached_inputs = cache.get(cache_key)
        if cached_inputs is not None:
            return cached_inputs

        device = topk_indices_decode.device
        token_to_req = attn_metadata.token_to_req[:num_tokens].to(
            device=device,
            dtype=torch.long,
        )
        decode_seq_lens = torch.index_select(
            actual_seq_lengths_key_decode[:num_reqs].to(
                device=device,
                dtype=topk_indices_decode.dtype,
            ),
            0,
            token_to_req,
        )
        seq_len_thresholds = decode_seq_lens.view(
            num_tokens,
            *([1] * (topk_indices_decode.ndim - 1)),
        )
        current_req_ids = torch.index_select(
            attn_metadata.req_ids_tensor[:num_reqs].to(
                device=device,
                dtype=torch.int64,
            ),
            0,
            token_to_req,
        )
        decode_cum_query_lens = actual_seq_lengths_query_decode[:num_reqs].to(
            device=device,
            dtype=torch.int32,
        )
        decode_query_lens = torch.diff(
            decode_cum_query_lens,
            prepend=decode_cum_query_lens.new_zeros(1),
        )
        stable_prefix_lens = (
            actual_seq_lengths_key_decode[:num_reqs].to(
                device=device,
                dtype=torch.int32,
            )
            - decode_query_lens
        ).clamp_min_(0)
        decode_stable_prefix_lens = torch.index_select(
            stable_prefix_lens,
            0,
            token_to_req,
        )
        full_kv_block_table = self._to_int32_device(
            attn_metadata.block_table[:num_reqs],
            device,
        ).contiguous()
        full_kv_actual_seq = self._to_int32_device(
            actual_seq_lengths_key_decode,
            device,
        )
        full_q_actual_seq = self._to_int32_device(
            actual_seq_lengths_query_decode,
            device,
        )
        self._validate_fused_overlap_mtp_decode_metadata(
            attn_metadata,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            full_q_actual_seq=full_q_actual_seq,
            full_kv_actual_seq=full_kv_actual_seq,
        )
        if num_tokens != num_reqs:
            full_q_actual_seq, full_kv_actual_seq, full_kv_block_table = self._flatten_fused_overlap_mtp_to_token_batch(
                attn_metadata,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                full_q_actual_seq=full_q_actual_seq,
                full_kv_actual_seq=full_kv_actual_seq,
                full_kv_block_table=full_kv_block_table,
            )
            if full_kv_block_table.size(0) != full_q_actual_seq.numel():
                raise RuntimeError(
                    "fused_overlap native-TND block_table batch mismatch: "
                    f"block_table.size(0)={full_kv_block_table.size(0)} "
                    f"full_q_actual_seq.numel()={full_q_actual_seq.numel()} "
                    f"num_tokens={num_tokens} num_reqs={num_reqs}"
                )
        if full_q_actual_seq.numel() != full_kv_actual_seq.numel():
            raise RuntimeError(
                "fused_overlap native-TND Q/KV actual_seq batch mismatch: "
                f"full_q_actual_seq.numel()={full_q_actual_seq.numel()} "
                f"full_kv_actual_seq.numel()={full_kv_actual_seq.numel()} "
                f"num_tokens={num_tokens} num_reqs={num_reqs}"
            )

        common_inputs = _FusedOverlapDecodeCommonInputs(
            seq_len_thresholds=seq_len_thresholds,
            current_req_ids=current_req_ids,
            stable_prefix_lens=decode_stable_prefix_lens,
            full_kv_block_table=full_kv_block_table,
            full_kv_actual_seq=full_kv_actual_seq,
            full_q_actual_seq=full_q_actual_seq,
        )
        cache[cache_key] = common_inputs
        return common_inputs

    def _execute_fused_overlap_offload_decode(
        self,
        ql_nope_decode: torch.Tensor,
        q_pe_decode: torch.Tensor,
        topk_indices_decode: torch.Tensor,
        attn_metadata: M,
        actual_seq_lengths_query_decode: torch.Tensor,
        actual_seq_lengths_key_decode: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        num_tokens = ql_nope_decode.shape[0]
        num_reqs = int(getattr(attn_metadata, "num_decodes", 0) or 0)
        if num_tokens <= 0 or num_reqs <= 0:
            raise RuntimeError(
                "fused_overlap decode requires positive num_tokens and num_reqs: "
                f"num_tokens={num_tokens} num_reqs={num_reqs}"
            )

        manager = get_sparse_kv_offload_manager()
        fused_op = self._require_custom_op("npu_fused_sparse_attention_overlap")
        topk_indices_decode = self._normalize_fused_overlap_topk_indices(
            topk_indices_decode,
            num_tokens,
            ql_nope_decode.device,
        )
        common_inputs = self._prepare_fused_overlap_decode_common_inputs(
            attn_metadata,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            topk_indices_decode=topk_indices_decode,
            actual_seq_lengths_query_decode=actual_seq_lengths_query_decode,
            actual_seq_lengths_key_decode=actual_seq_lengths_key_decode,
        )
        topk_head_count = topk_indices_decode.shape[1]
        topk = topk_indices_decode.shape[2]
        if topk_head_count != 1:
            raise ValueError(f"external fused_overlap planner requires one TopK head, got {topk_head_count}")
        runtime_blocks_per_row = max(cdiv(topk, self.block_size), 1)
        cache_blocks_per_row = self.lru_resident_capacity // self.block_size
        if runtime_blocks_per_row > cache_blocks_per_row:
            raise RuntimeError(
                "fused_overlap topk exceeds selection buffer capacity: "
                f"topk={topk} runtime_blocks_per_row={runtime_blocks_per_row} "
                f"resident_capacity={self.lru_resident_capacity} block_size={self.block_size}"
            )

        full_kv_cache_cpu, full_k_rope_cpu = manager.get_fused_overlap_cpu_kv_inputs(layer_name)
        full_kv_cache = self._flatten_pa_cache(full_kv_cache_cpu).contiguous()
        full_k_rope = self._flatten_pa_cache(full_k_rope_cpu).contiguous()
        full_kv_block_table = common_inputs.full_kv_block_table
        full_kv_actual_seq = common_inputs.full_kv_actual_seq
        full_q_actual_seq = common_inputs.full_q_actual_seq

        layer_id = manager._get_offload_layer_id(layer_name)
        selection_row_count = max(self.max_num_topk_rows, num_tokens) * topk_head_count
        selection_kv_cache = self._flatten_selection_buffer(
            manager.topk_buffers_k[layer_id],
            row_count=selection_row_count,
            blocks_per_row=cache_blocks_per_row,
            name="selection_kv_cache",
        )
        selection_k_rope = self._flatten_selection_buffer(
            manager.topk_buffers_v[layer_id],
            row_count=selection_row_count,
            blocks_per_row=cache_blocks_per_row,
            name="selection_k_rope",
        )
        (
            selection_block_table,
            selection_block_status,
            selection_membership_map,
            last_req_ids,
        ) = self._ensure_fused_overlap_selection_state(
            token_count=num_tokens,
            topk_head_count=topk_head_count,
            topk=topk,
            cache_blocks_per_row=cache_blocks_per_row,
            device=ql_nope_decode.device,
        )
        external_plan_prepared = manager.prepare_fused_overlap_external_plan(
            layer_name=layer_name,
            num_tokens=num_tokens,
            topk_indices_npu=topk_indices_decode.squeeze(1),
            req_ids_npu=common_inputs.current_req_ids,
            stable_prefix_lens_npu=common_inputs.stable_prefix_lens,
            visible_seq_lens_npu=full_kv_actual_seq,
            selection_membership_map=selection_membership_map,
            capturing=get_forward_context().capturing,
            skip_topk=self.skip_topk,
        )
        if not external_plan_prepared:
            self._invalidate_fused_overlap_selection_rows(
                selection_block_status,
                selection_membership_map,
                last_req_ids,
                attn_metadata,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                topk_count=topk,
                seq_lens=actual_seq_lengths_key_decode,
                cum_query_lens=actual_seq_lengths_query_decode,
            )

        fused_query = torch.cat([ql_nope_decode, q_pe_decode], dim=-1).contiguous()
        fused_inputs = {
            "query": fused_query,
            "selection_k_rope": selection_k_rope,
            "selection_kv_cache": selection_kv_cache,
            "selection_kv_block_table": selection_block_table,
            "selection_kv_block_status": selection_block_status,
            "selection_membership_map": selection_membership_map,
            "selection_topk_indices": topk_indices_decode,
            "full_k_rope": full_k_rope,
            "full_kv_cache": full_kv_cache,
            "full_kv_block_table": full_kv_block_table,
            "full_kv_actual_seq": full_kv_actual_seq,
            "full_q_actual_seq": full_q_actual_seq,
            "scale_value": self.scale,
            "sparse_block_size": 1,
            "selection_topk_block_size": 1,
            "layout_query": "TND",
            "layout_kv": "PA_BSND",
            "sparse_mode": 3,
        }
        manager.inject_current_kv_into_selection(
            layer_name=layer_name,
            num_tokens=num_tokens,
            selection_kv_cache=selection_kv_cache,
            selection_k_rope=selection_k_rope,
            capturing=get_forward_context().capturing,
        )
        attn_output = fused_op(**fused_inputs)
        attn_output = attn_output[..., : ql_nope_decode.shape[-1]].contiguous()
        manager.wait_for_current_kv_writeback(get_forward_context().capturing)
        return attn_output

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        num_decodes = int(getattr(attn_metadata, "num_decodes", 0) or 0)
        num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0) or 0)
        num_prefills = int(getattr(attn_metadata, "num_prefills", 0) or 0)
        manager = get_sparse_kv_offload_manager()
        layer_name = self._offload_layer_name()

        if attn_metadata.nano_enabled:
            result = self._nano_attention(ql_nope, q_pe, topk_indices, attn_metadata, manager, layer_name)
            return self._pad_to_input_tokens(result, ql_nope.shape[0])

        if num_decode_tokens == 0:
            # Pure prefill batch (colocate debug only).
            _check_device_kv_cache_exist()
            return super()._execute_sparse_flash_attention_process(
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table=block_table,
            )

        if attn_metadata.req_ids_tensor is None or attn_metadata.token_to_req is None:
            raise RuntimeError("Sparse KV offload requires req_ids_tensor/token_to_req metadata")

        if self.use_fused_overlap:
            decode_attn_output = self._execute_fused_overlap_offload_decode(
                ql_nope[:num_decode_tokens],
                q_pe[:num_decode_tokens],
                topk_indices[:num_decode_tokens],
                attn_metadata,
                actual_seq_lengths_query[:num_decodes],
                actual_seq_lengths_key[:num_decodes],
                layer_name,
            )
            if num_prefills == 0:
                return self._pad_to_input_tokens(decode_attn_output, ql_nope.shape[0])
            _check_device_kv_cache_exist()
            prefill_query_offset = actual_seq_lengths_query[num_decodes - 1]
            prefill_query_lens = actual_seq_lengths_query[num_decodes:] - prefill_query_offset
            prefill_block_table = attn_metadata.block_table[num_decodes : num_decodes + num_prefills]
            prefill_attn_output = super()._execute_sparse_flash_attention_process(
                ql_nope[num_decode_tokens:],
                q_pe[num_decode_tokens:],
                kv_cache,
                topk_indices[num_decode_tokens:],
                attn_metadata,
                prefill_query_lens,
                actual_seq_lengths_key[num_decodes:],
                block_table=prefill_block_table,
            )
            attn_output = torch.cat([decode_attn_output, prefill_attn_output], dim=0)
            return self._pad_to_input_tokens(attn_output, ql_nope.shape[0])

        token_to_req = attn_metadata.token_to_req[:num_decode_tokens]
        row_to_req = token_to_req.to(dtype=torch.int64)
        decode_seq_lens = torch.index_select(
            actual_seq_lengths_key[:num_decodes],
            0,
            row_to_req,
        )
        decode_cum_query_lens = actual_seq_lengths_query[:num_decodes]
        decode_query_lens = decode_cum_query_lens.clone()
        if num_decodes > 1:
            decode_query_lens[1:] -= decode_cum_query_lens[:-1]
        # Only the query span can be rewritten by the next MTP step.
        stable_prefix_lens = (actual_seq_lengths_key[:num_decodes] - decode_query_lens).clamp_min_(0)
        decode_stable_prefix_lens = torch.index_select(
            stable_prefix_lens,
            0,
            row_to_req,
        )
        decode_topk = topk_indices[:num_decode_tokens]
        seq_len_thresholds = decode_seq_lens.view(
            decode_seq_lens.shape[0],
            *([1] * (decode_topk.ndim - 1)),
        )
        valid_topk = build_valid_topk_mask(decode_topk, seq_len_thresholds)
        decode_topk = torch.where(
            valid_topk,
            decode_topk,
            torch.full_like(decode_topk, -1),
        )
        if decode_topk.ndim == 3 and decode_topk.shape[1] == 1:
            decode_topk = decode_topk.squeeze(1)
        if decode_topk.ndim != 2:
            raise ValueError("Sparse KV offload top-k must have [tokens, topk] shape")

        (
            resident_k,
            resident_v,
            resident_slot_indices,
            resident_block_table,
            resident_query_lens,
            resident_seq_lens,
        ) = self._resident_views(manager, layer_name, num_decode_tokens)
        decode_req_ids = torch.index_select(
            attn_metadata.req_ids_tensor[:num_decodes],
            0,
            row_to_req,
        )
        manager.onload_topk_kv(
            layer_name,
            num_decode_tokens,
            num_decodes,
            attn_metadata.block_table[:num_decodes],
            decode_topk,
            resident_slot_indices,
            decode_req_ids,
            decode_stable_prefix_lens,
            token_to_req,
            capturing=self._in_graph_runtime(),
            skip_topk=self.skip_topk,
        )
        decode_attn_output = DeviceOperator.execute_sparse_flash_attention_process(
            self,
            ql_nope[:num_decode_tokens],
            q_pe[:num_decode_tokens],
            (resident_k, resident_v),
            resident_slot_indices.unsqueeze(1),
            attn_metadata,
            resident_query_lens,
            resident_seq_lens,
            block_table=resident_block_table,
        )
        if num_prefills == 0:
            return self._pad_to_input_tokens(decode_attn_output, ql_nope.shape[0])

        # Mixed batch (colocate debug only): prefill rows still attend the NPU
        # paged cache. The cumulative query lengths are rebased to the first
        # prefill request.
        _check_device_kv_cache_exist()
        prefill_query_offset = actual_seq_lengths_query[num_decodes - 1]
        prefill_query_lens = actual_seq_lengths_query[num_decodes:] - prefill_query_offset
        prefill_block_table = attn_metadata.block_table[num_decodes : num_decodes + num_prefills]
        prefill_attn_output = super()._execute_sparse_flash_attention_process(
            ql_nope[num_decode_tokens:],
            q_pe[num_decode_tokens:],
            kv_cache,
            topk_indices[num_decode_tokens:],
            attn_metadata,
            prefill_query_lens,
            actual_seq_lengths_key[num_decodes:],
            block_table=prefill_block_table,
        )
        attn_output = torch.cat([decode_attn_output, prefill_attn_output], dim=0)
        return self._pad_to_input_tokens(attn_output, ql_nope.shape[0])
