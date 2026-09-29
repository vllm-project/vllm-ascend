# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache metadata and execution backends for the GLM-Next pooled indexer."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_pcp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.attention.context_parallel.common_cp import get_cp_local_query_key_lens
from vllm_ascend.attention.context_parallel.sfa_dcp_utils import (
    build_sfa_dcp_replicated_block_table,
    build_sfa_dcp_replicated_slot_mapping,
    get_sfa_dcp_local_block_table,
    get_sfa_dcp_max_local_block_table_cols,
    get_sfa_pcp_global_metadata,
)
from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendKPoolIndexerCacheSpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
)
from vllm_ascend.device.hardware_profile import AttentionBackendFamily, get_current_hardware_profile
from vllm_ascend.models.glm5next.kv_cache import (
    format_indexer_kpool_slot_mapping,
)
from vllm_ascend.utils import _round_up, enable_dsa_cp

GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE = 128


@dataclass
class AscendIndexerKPoolQueryMetadata:
    """Token-local query addressing over the replicated pool cache."""

    positions: torch.Tensor
    cum_query_lens: torch.Tensor
    num_actual_tokens: int


@dataclass
class AscendIndexerKPoolMetadata:
    """Metadata for compressed indexer cache writes and top-k reads."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor | None
    positions: torch.Tensor
    block_size: int
    compress_ratio: int
    cache_role: str = "indexer"
    cum_query_lens: torch.Tensor | None = None
    raw_seq_lens: torch.Tensor | None = None
    num_actual_tokens: int = 0
    query_metadata: AscendIndexerKPoolQueryMetadata | None = None
    write_slot_mapping: torch.Tensor | None = None
    write_positions: torch.Tensor | None = None
    write_cum_query_lens: torch.Tensor | None = None
    write_raw_seq_lens: torch.Tensor | None = None
    pcp_hidden_restore_indices: torch.Tensor | None = None
    pcp_local_token_count: int = 0


@dataclass
class _AscendIndexerKPoolBuffers:
    """Persistent per-step tensors referenced by captured ACL graphs."""

    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    cum_query_lens: torch.Tensor
    raw_seq_lens: torch.Tensor
    positions: torch.Tensor
    write_slot_mapping: torch.Tensor
    write_cum_query_lens: torch.Tensor
    write_raw_seq_lens: torch.Tensor
    write_positions: torch.Tensor
    select_dcp_block_table: torch.Tensor | None
    write_dcp_block_table: torch.Tensor | None
    select_dcp_token_slots: torch.Tensor | None
    write_dcp_token_slots: torch.Tensor | None


class AscendIndexerKPoolMetadataBuilder(AttentionMetadataBuilder):
    """Build pool-level addressing for the compressed indexer cache."""

    consumes_pcp_context = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        # This cache-only builder still participates in graph capability
        # reduction. Its decode metadata uses persistent buffers refreshed in
        # place, so it must not disable the main model's uniform decode graph.
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: MLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        if not isinstance(kv_cache_spec, AscendKPoolIndexerCacheSpec):
            raise TypeError(
                "Ascend Indexer KPool backend requires "
                f"AscendKPoolIndexerCacheSpec, got {type(kv_cache_spec).__name__}."
            )
        compress_ratio = get_kv_cache_compression_ratio(kv_cache_spec)
        if compress_ratio <= 1:
            raise ValueError(f"Ascend Indexer KPool cache requires compress_ratio > 1, got {compress_ratio}.")
        if not layer_names or any(not name.endswith(".indexer.k_cache") for name in layer_names):
            raise ValueError(f"Invalid Indexer KPool cache layer names: {layer_names}.")
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.logical_block_size = vllm_config.cache_config.block_size
        self.storage_block_size = get_storage_block_size(kv_cache_spec)
        if self.storage_block_size <= 0:
            raise ValueError(f"Indexer KPool storage block size must be positive, got {self.storage_block_size}.")
        self.compress_ratio = compress_ratio
        parallel_config = vllm_config.parallel_config
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        self.pcp_size = parallel_config.prefill_context_parallel_size
        self.dcp_size = kv_cache_spec.dcp_replication_size
        if self.dcp_size != parallel_config.decode_context_parallel_size:
            raise ValueError(
                "KPool cache replication must match decode context parallelism: "
                f"cache={self.dcp_size}, dcp={parallel_config.decode_context_parallel_size}."
            )
        self.use_dcp = self.dcp_size > 1
        if self.logical_block_size % GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE:
            raise ValueError(
                "GLM-Next logical block size must be divisible by the SFA "
                f"kernel block size: logical={self.logical_block_size}, "
                f"kernel={GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE}."
            )
        self.kernel_row_block_size = GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE // self.compress_ratio
        scheduler_config = vllm_config.scheduler_config
        self._max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        self.use_dsa_cp = enable_dsa_cp()
        self.dsa_cp_size = get_tp_group().world_size if self.use_dsa_cp else 1
        self._max_num_batched_tokens = _round_up(self._max_num_batched_tokens, self.dsa_cp_size)
        self._max_num_seqs = scheduler_config.max_num_seqs
        # FULL draft graphs pad the request-shaped metadata to the selected
        # token bucket.  That padded length can exceed max_num_seqs (for
        # example, four requests with K=3 use a 16-token bucket), so request
        # buffers must cover the graph token capacity as well.
        self._max_num_metadata_reqs = max(
            self._max_num_seqs,
            self._max_num_batched_tokens,
        )
        self._max_num_write_tokens = self._max_num_batched_tokens * self.pcp_size
        # Each MTP draft step owns a persistent common slot-mapping tensor. Key
        # derived buffers by that address so capture and runtime rebuilds bind
        # the same storage without different draft steps overwriting each other.
        self._metadata_buffers: dict[int, _AscendIndexerKPoolBuffers] = {}
        self._dsa_cp_query_buffers: dict[int, torch.Tensor] = {}
        if self.use_dcp:
            self.max_local_block_table_cols = get_sfa_dcp_max_local_block_table_cols(
                vllm_config.model_config.max_model_len,
                self.logical_block_size,
                self.dcp_size,
                1,
            )
            max_replicated_cols = self.max_local_block_table_cols * self.dcp_size
            self._replicated_col_idx = torch.arange(
                max_replicated_cols,
                dtype=torch.int32,
                device=device,
            )

    def _build_query_metadata(
        self, common: CommonAttentionMetadata, positions: torch.Tensor
    ) -> AscendIndexerKPoolQueryMetadata | None:
        if not self.use_dsa_cp:
            return None
        local_tokens = positions.shape[0] // self.dsa_cp_size
        local_start = get_tp_group().rank_in_group * local_tokens
        local_end = local_start + local_tokens
        key = common.slot_mapping.data_ptr()
        if key not in self._dsa_cp_query_buffers:
            self._dsa_cp_query_buffers[key] = torch.empty(
                self._max_num_metadata_reqs, dtype=torch.int32, device=self.device
            )
        query_lens = self._dsa_cp_query_buffers[key][: common.num_reqs]
        local_query_lens, _ = get_cp_local_query_key_lens(
            common.query_start_loc,
            common.query_start_loc[1 : common.num_reqs + 1],
            common.seq_lens[: common.num_reqs],
            local_start,
            local_end,
        )
        query_lens.copy_(local_query_lens)
        return AscendIndexerKPoolQueryMetadata(
            positions=positions[local_start:local_end],
            cum_query_lens=query_lens,
            num_actual_tokens=max(min(local_end, common.num_actual_tokens) - local_start, 0),
        )

    def _get_metadata_buffers(self, common_attn_metadata: CommonAttentionMetadata) -> _AscendIndexerKPoolBuffers:
        key = common_attn_metadata.slot_mapping.data_ptr()
        buffers = self._metadata_buffers.get(key)
        if buffers is None:

            def empty_tokens(capacity: int, dtype: torch.dtype) -> torch.Tensor:
                return torch.empty(capacity, dtype=dtype, device=self.device)

            def empty_block_table() -> torch.Tensor | None:
                if not self.use_dcp:
                    return None
                return torch.empty(
                    (self._max_num_metadata_reqs, self.max_local_block_table_cols * self.dcp_size),
                    dtype=torch.int32,
                    device=self.device,
                )

            buffers = _AscendIndexerKPoolBuffers(
                slot_mapping=empty_tokens(self._max_num_batched_tokens, torch.int64),
                seq_lens=empty_tokens(self._max_num_metadata_reqs, torch.int32),
                cum_query_lens=empty_tokens(self._max_num_metadata_reqs, torch.int32),
                raw_seq_lens=empty_tokens(self._max_num_metadata_reqs, torch.int32),
                positions=empty_tokens(self._max_num_batched_tokens, torch.int64),
                write_slot_mapping=empty_tokens(self._max_num_write_tokens, torch.int64),
                write_cum_query_lens=empty_tokens(self._max_num_metadata_reqs, torch.int32),
                write_raw_seq_lens=empty_tokens(self._max_num_metadata_reqs, torch.int32),
                write_positions=empty_tokens(self._max_num_write_tokens, torch.int64),
                select_dcp_block_table=empty_block_table(),
                write_dcp_block_table=empty_block_table(),
                select_dcp_token_slots=(
                    empty_tokens(self._max_num_batched_tokens, torch.int32) if self.use_dcp else None
                ),
                write_dcp_token_slots=(empty_tokens(self._max_num_write_tokens, torch.int32) if self.use_dcp else None),
            )
            self._metadata_buffers[key] = buffers
        return buffers

    def _build_cache_view(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        buffers: _AscendIndexerKPoolBuffers,
        *,
        write: bool,
        token_slots_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_input_tokens
        token_capacity = buffers.write_slot_mapping.shape[0] if write else buffers.slot_mapping.shape[0]
        if num_tokens > token_capacity:
            raise RuntimeError(
                f"KPool {'write' if write else 'select'} metadata needs {num_tokens} "
                f"token slots, but its persistent buffer holds {token_capacity}."
            )
        if self.use_dcp:
            local_table = get_sfa_dcp_local_block_table(
                common_attn_metadata.block_table_tensor,
                num_reqs,
                self.max_local_block_table_cols,
            )
            replicated_cols = local_table.shape[1] * self.dcp_size
            table_buffer = buffers.write_dcp_block_table if write else buffers.select_dcp_block_table
            token_slot_buffer = buffers.write_dcp_token_slots if write else buffers.select_dcp_token_slots
            assert table_buffer is not None and token_slot_buffer is not None
            if table_buffer.shape[0] < num_reqs or table_buffer.shape[1] < replicated_cols:
                raise RuntimeError("KPool replicated block-table buffer is too small.")
            block_table = build_sfa_dcp_replicated_block_table(
                local_table,
                common_attn_metadata.seq_lens,
                table_buffer[:num_reqs, :replicated_cols],
                self._replicated_col_idx[:replicated_cols],
                self.dcp_size,
                1,
            )
            token_slots = token_slot_buffer[:num_tokens]
            build_sfa_dcp_replicated_slot_mapping(
                common_attn_metadata,
                block_table,
                token_slots,
                self.logical_block_size,
                self.device,
            )
        else:
            block_table = common_attn_metadata.block_table_tensor[:num_reqs]
            token_slots = (
                common_attn_metadata.slot_mapping[:num_tokens]
                if token_slots_override is None
                else token_slots_override[:num_tokens]
            )

        compressed_slot_buffer = buffers.write_slot_mapping if write else buffers.slot_mapping
        compressed_slots = compressed_slot_buffer[:num_tokens]
        compressed_slots.copy_(
            format_indexer_kpool_slot_mapping(
                token_slots,
                common_attn_metadata.positions[:num_tokens].long(),
                self.logical_block_size,
                self.compress_ratio,
            )
        )
        return block_table, compressed_slots

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendIndexerKPoolMetadata:
        del common_prefix_len, fast_build
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        num_cache_tokens = _round_up(num_input_tokens, self.dsa_cp_size)
        buffers = self._get_metadata_buffers(common_attn_metadata)
        positions = buffers.positions[:num_cache_tokens]
        positions[num_input_tokens:].zero_()
        positions[:num_input_tokens].copy_(common_attn_metadata.positions[:num_input_tokens])
        block_table, _ = self._build_cache_view(common_attn_metadata, buffers, write=False)
        slot_mapping = buffers.slot_mapping[:num_cache_tokens]
        slot_mapping[num_input_tokens:].fill_(-1)
        seq_lens = buffers.seq_lens[:num_reqs]
        torch.div(
            common_attn_metadata.seq_lens[:num_reqs],
            self.compress_ratio,
            rounding_mode="floor",
            out=seq_lens,
        )
        cum_query_lens = buffers.cum_query_lens[:num_reqs]
        cum_query_lens.copy_(common_attn_metadata.query_start_loc[: num_reqs + 1][1:])
        raw_seq_lens = buffers.raw_seq_lens[:num_reqs]
        raw_seq_lens.copy_(common_attn_metadata.seq_lens[:num_reqs])
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens_cpu = None
        if seq_lens_cpu is not None:
            seq_lens_cpu = torch.div(seq_lens_cpu, self.compress_ratio, rounding_mode="floor")
        write_slot_mapping = slot_mapping
        write_positions = positions
        write_cum_query_lens = cum_query_lens
        write_raw_seq_lens = raw_seq_lens
        pcp_hidden_restore_indices = None
        pcp_local_token_count = 0
        pcp_context = kwargs.get("pcp_context")
        if self.use_pcp and pcp_context is not None and bool(pcp_context.global_batch.is_prefilling_np.any()):
            pcp_cache_group_idx = kwargs.get("pcp_cache_group_idx")
            if pcp_cache_group_idx is None:
                raise RuntimeError("KPool PCP metadata requires the PCP cache-group index.")
            global_metadata = get_sfa_pcp_global_metadata(
                common_attn_metadata,
                pcp_context,
                pcp_cache_group_idx,
            )
            global_num_reqs = global_metadata.num_reqs
            global_num_tokens = global_metadata.num_input_tokens
            global_token_slots = pcp_context.global_slot_mappings[pcp_cache_group_idx, :global_num_tokens]
            _, write_slot_mapping = self._build_cache_view(
                global_metadata,
                buffers,
                write=True,
                token_slots_override=global_token_slots,
            )
            write_positions = buffers.write_positions[:global_num_tokens]
            write_positions.copy_(global_metadata.positions[:global_num_tokens])
            write_cum_query_lens = buffers.write_cum_query_lens[:global_num_reqs]
            write_cum_query_lens.copy_(global_metadata.query_start_loc[1 : global_num_reqs + 1])
            write_raw_seq_lens = buffers.write_raw_seq_lens[:global_num_reqs]
            write_raw_seq_lens.copy_(global_metadata.seq_lens[:global_num_reqs])
            pcp_hidden_restore_indices = pcp_context.hidden_restore_idx[:global_num_tokens]
            if pcp_context.padded_gather_idx is None:
                raise RuntimeError("KPool PCP prefill requires the gathered-token layout.")
            pcp_local_token_count = pcp_context.padded_gather_idx.numel() // self.pcp_size
        return AscendIndexerKPoolMetadata(
            block_table=block_table,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            block_size=self.kernel_row_block_size,
            compress_ratio=self.compress_ratio,
            cum_query_lens=cum_query_lens,
            raw_seq_lens=raw_seq_lens,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_metadata=self._build_query_metadata(common_attn_metadata, positions),
            write_slot_mapping=write_slot_mapping,
            write_positions=write_positions,
            write_cum_query_lens=write_cum_query_lens,
            write_raw_seq_lens=write_raw_seq_lens,
            pcp_hidden_restore_indices=pcp_hidden_restore_indices,
            pcp_local_token_count=pcp_local_token_count,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        **kwargs,
    ) -> AscendIndexerKPoolMetadata:
        return self.build(0, common_attn_metadata, **kwargs)

    def build_for_graph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        attn_state: Any = None,
        **kwargs,
    ) -> AscendIndexerKPoolMetadata:
        del attn_state
        return self.build(0, common_attn_metadata, **kwargs)

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
        **kwargs,
    ) -> AscendIndexerKPoolMetadata:
        del draft_index
        return self.build(0, common_attn_metadata, fast_build=True, **kwargs)


class AscendIndexerKPoolBackend(AttentionBackend):
    """Cache-only backend for the compressed indexer keys."""

    @staticmethod
    def get_impl_cls():
        return None

    @staticmethod
    def get_name() -> str:
        return "ASCEND_INDEXER_KPOOL"

    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The scheduler manages logical token blocks. Triton consumes complete
        # compressed storage pages with their actual size and strides.
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type[AscendIndexerKPoolMetadataBuilder]:
        return AscendIndexerKPoolMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_type, cache_dtype_str
        if num_kv_heads != 1:
            raise ValueError(f"Indexer KPool cache requires one KV head, got {num_kv_heads}.")
        return (num_blocks, block_size, num_kv_heads, head_size)


@dataclass
class AscendIndexerKPoolTailMetadata:
    """Addressing required to update the compressor tail cache."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    write_block_table: torch.Tensor | None = None
    write_slot_mapping: torch.Tensor | None = None


class AscendIndexerKPoolTailMetadataBuilder(AttentionMetadataBuilder):
    """Build independent metadata for the GLM-Next compressor tail."""

    consumes_pcp_context = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        # Full-graph tail writes use the fixed-shape sentinel path. Do not let
        # the base class default NEVER downgrade FULL_DECODE_ONLY for the main
        # model merely because this cache-only builder is in the cache group.
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AscendIndexerKPoolTailSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        if not isinstance(kv_cache_spec, AscendIndexerKPoolTailSpec):
            raise TypeError(
                "Ascend Indexer KPool tail backend requires "
                f"AscendIndexerKPoolTailSpec, got {type(kv_cache_spec).__name__}."
            )
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.use_dsa_cp = enable_dsa_cp()
        self.dsa_cp_size = get_tp_group().world_size if self.use_dsa_cp else 1
        self._max_num_batched_tokens = _round_up(vllm_config.scheduler_config.max_num_batched_tokens, self.dsa_cp_size)
        self._slot_buffers: dict[int, torch.Tensor] = {}

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendIndexerKPoolTailMetadata:
        del common_prefix_len, fast_build
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        if self.use_dsa_cp:
            # Match the gathered K/gate rows, including TP alignment padding.
            key = common_attn_metadata.slot_mapping.data_ptr()
            if key not in self._slot_buffers:
                self._slot_buffers[key] = torch.empty(
                    self._max_num_batched_tokens, dtype=slot_mapping.dtype, device=self.device
                )
            padded_tokens = _round_up(num_input_tokens, self.dsa_cp_size)
            slots = self._slot_buffers[key][:padded_tokens]
            slots[:num_input_tokens].copy_(slot_mapping)
            slots[num_input_tokens:].fill_(-1)
            slot_mapping = slots
        write_block_table = block_table
        write_slot_mapping = slot_mapping
        pcp_context = kwargs.get("pcp_context")
        if pcp_context is not None and bool(pcp_context.global_batch.is_prefilling_np.any()):
            pcp_cache_group_idx = kwargs.get("pcp_cache_group_idx")
            if pcp_cache_group_idx is None:
                raise RuntimeError("KPool tail PCP metadata requires the PCP cache-group index.")
            global_num_reqs = pcp_context.global_batch.num_reqs
            global_num_tokens = pcp_context.global_batch.num_tokens
            write_block_table = pcp_context.global_block_tables[pcp_cache_group_idx][:global_num_reqs]
            write_slot_mapping = pcp_context.global_slot_mappings[pcp_cache_group_idx, :global_num_tokens]
        return AscendIndexerKPoolTailMetadata(
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            write_block_table=write_block_table,
            write_slot_mapping=write_slot_mapping,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        **kwargs,
    ) -> AscendIndexerKPoolTailMetadata:
        return self.build(0, common_attn_metadata, **kwargs)

    def build_for_graph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        attn_state: Any = None,
        **kwargs,
    ) -> AscendIndexerKPoolTailMetadata:
        del attn_state
        return self.build(0, common_attn_metadata, **kwargs)

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
        **kwargs,
    ) -> AscendIndexerKPoolTailMetadata:
        del draft_index
        return self.build(0, common_attn_metadata, fast_build=True, **kwargs)


class AscendIndexerKPoolTailBackend(AttentionBackend):
    """Cache-only backend for the GLM-Next compressor tail."""

    @staticmethod
    def get_impl_cls():
        return None

    @staticmethod
    def get_name() -> str:
        return "ASCEND_INDEXER_KPOOL_TAIL"

    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Ring capacity is independent of the pool size and SFA C128.
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type[AscendIndexerKPoolTailMetadataBuilder]:
        return AscendIndexerKPoolTailMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_type, cache_dtype_str
        if num_kv_heads != 1:
            raise ValueError(f"Indexer KPool tail cache requires one KV head, got {num_kv_heads}.")
        return (num_blocks, 2, block_size, head_size)


class Glm5NextKPoolIndexerBackend(nn.Module):
    """Model-side implementation of the unified seven-argument indexer API.

    The cache-only backends above own the engine-facing attention contracts.
    Keep execution independent of the standard SFA indexer and its operators.
    """

    def __init__(self, vllm_indexer: Any, qk_rope_head_dim: int) -> None:
        super().__init__()
        if qk_rope_head_dim != 0:
            raise ValueError(
                f"GLM-Next KPool indexing supports NoPE queries only, got qk_rope_head_dim={qk_rope_head_dim}."
            )
        parallel_config = vllm_indexer.vllm_config.parallel_config

        if get_current_hardware_profile().attention_backend_family is AttentionBackendFamily.COMPATIBILITY:
            raise NotImplementedError("KPool sparse attention requires Ascend A2, A3 or A5.")

        self.n_head: int = vllm_indexer.n_head
        self.head_dim: int = vllm_indexer.head_dim
        self.topk_tokens: int = vllm_indexer.topk_tokens
        self.q_lora_rank: int = vllm_indexer.q_lora_rank
        self.index_kpool: int = vllm_indexer.index_kpool
        self.wq_b = vllm_indexer.wq_b
        self.wk_weights_proj = vllm_indexer.wk_weights_proj
        self.k_norm = vllm_indexer.k_norm
        self.softmax_scale = vllm_indexer.softmax_scale
        self.index_kpool_compress_ape = vllm_indexer.index_kpool_compress_ape
        self.index_kpool_compress_gate = vllm_indexer.index_kpool_compress_gate
        self.k_cache: Any = vllm_indexer.k_cache
        self.tail_cache: Any = vllm_indexer.tail_cache
        self.topk_indices_buffer: torch.Tensor | None = vllm_indexer.topk_indices_buffer
        self._pcp_active = parallel_config.prefill_context_parallel_size > 1
        # Load KPool operators only when constructing the model-side backend;
        # cache metadata is also imported during engine initialization.
        from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool

        self.indexer_op = SparseAttnIndexerKpool(self.topk_tokens, self.head_dim)
        self.enable_sparse_li_c8 = False
        for name in ("_wk_weight_f32", "_gate_weight_f32", "_norm_weight_f32", "_norm_bias_f32"):
            self.register_buffer(name, None, persistent=False)

    @property
    def topk_output_width(self) -> int:
        return self.topk_tokens + self.index_kpool - 1

    def get_topk_lengths(self, positions: torch.Tensor) -> torch.Tensor:
        visible = (positions + 1).clamp_min(0)
        history = (visible // self.index_kpool * self.index_kpool).clamp(max=self.topk_tokens)
        return history + visible % self.index_kpool

    @property
    def num_cache_tensors(self) -> int:
        return 1

    def process_weights_after_loading(self) -> None:
        self._wk_weight_f32 = self.wk_weights_proj.weight.detach().float()
        self._gate_weight_f32 = self.index_kpool_compress_gate.detach().float()
        self._norm_weight_f32 = self.k_norm.weight.detach().float() if self.k_norm.weight is not None else None
        self._norm_bias_f32 = self.k_norm.bias.detach().float() if self.k_norm.bias is not None else None

    @staticmethod
    def _bound_cache(layer: Any) -> torch.Tensor:
        context = get_forward_context()
        cache = layer.kv_cache
        if isinstance(cache, (list, tuple)):
            virtual_engine = getattr(context, "virtual_engine", 0) or 0
            if virtual_engine >= len(cache):
                raise IndexError(f"Cache virtual engine {virtual_engine} is out of range.")
            cache = cache[virtual_engine]
        if isinstance(cache, (list, tuple)):
            if len(cache) != 1:
                raise TypeError("GLM KPool cache must contain one tensor.")
            cache = cache[0]
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"GLM KPool cache {layer.prefix!r} is not bound to a tensor.")
        return cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k_hidden_states: torch.Tensor,
        indexer_metadata: Any,
        compute_topk: bool = True,
        attn_q_gather_handle: torch.distributed.Work | None = None,
    ) -> torch.Tensor | None:
        if not isinstance(indexer_metadata, AscendIndexerKPoolMetadata):
            raise TypeError("GLM KPool backend requires AscendIndexerKPoolMetadata.")
        context = get_forward_context()
        if not isinstance(context.attn_metadata, dict):
            raise TypeError("GLM KPool backend requires per-layer metadata.")
        tail_metadata = context.attn_metadata[self.tail_cache.prefix]
        if not isinstance(tail_metadata, AscendIndexerKPoolTailMetadata):
            raise TypeError("GLM KPool backend requires tail-cache metadata.")

        num_tokens = hidden_states.shape[0]
        if indexer_metadata.query_metadata is None and context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            num_tokens = min(num_tokens, indexer_metadata.num_actual_tokens)
        if num_tokens > k_hidden_states.shape[0]:
            raise RuntimeError(
                "KPool projection exceeds the local key input: "
                f"required={num_tokens}, k_hidden={k_hidden_states.shape[0]}."
            )
        hidden = hidden_states[:num_tokens]
        k_hidden = k_hidden_states[:num_tokens]
        if self._wk_weight_f32 is None:
            self.process_weights_after_loading()
        assert self._wk_weight_f32 is not None
        hidden_f32 = hidden.float()
        k_hidden_f32 = hidden_f32 if k_hidden_states is hidden_states else k_hidden.float()
        projected = F.linear(k_hidden_f32, self._wk_weight_f32)
        k = F.layer_norm(
            projected[:, : self.head_dim],
            (self.head_dim,),
            self._norm_weight_f32,
            self._norm_bias_f32,
            getattr(self.k_norm, "eps", getattr(self.k_norm, "variance_epsilon", 1e-6)),
        )
        gate_score = F.linear(k_hidden_f32, self._gate_weight_f32)
        if indexer_metadata.query_metadata is not None:
            # Pool compression can cross a TP token boundary. Gather raw K and
            # gates before updating the replicated pool/tail caches.
            gathered = get_tp_group().all_gather(torch.cat((k, gate_score), dim=-1), dim=0)
            k, gate_score = gathered.split(self.head_dim, dim=-1)
        q_values = None
        weights = None
        if compute_topk:
            if isinstance(q_c, tuple):
                raise TypeError("GLM KPool backend requires an unquantized q_c tensor.")
            q_values = self.wq_b(q_c[:num_tokens])[0].view(num_tokens, self.n_head, self.head_dim)
            weights = (
                projected[:num_tokens, self.head_dim :]
                if k_hidden_states is hidden_states
                else F.linear(hidden_f32, self._wk_weight_f32[self.head_dim :])
            )[:num_tokens].to(q_values.dtype)
            weights = weights * (self.softmax_scale * self.n_head**-0.5)

        write_k = k
        write_gate_score = gate_score
        if indexer_metadata.pcp_hidden_restore_indices is not None:
            if not self._pcp_active:
                raise RuntimeError("KPool metadata requested a PCP gather while PCP is disabled.")
            local_count = indexer_metadata.pcp_local_token_count
            gather_k = k[:local_count]
            gather_gate = gate_score[:local_count]
            if local_count > k.shape[0]:
                # PCP pads every rank to the largest local token count. The
                # model may project only this rank's actual token rows.
                padding = local_count - k.shape[0]
                gather_k = F.pad(gather_k, (0, 0, 0, padding))
                gather_gate = F.pad(gather_gate, (0, 0, 0, padding))
            gathered_k = get_pcp_group().all_gather(gather_k.contiguous(), dim=0)
            gathered_gate = get_pcp_group().all_gather(gather_gate.contiguous(), dim=0)
            restore = indexer_metadata.pcp_hidden_restore_indices
            write_k = torch.index_select(gathered_k, 0, restore)
            write_gate_score = torch.index_select(gathered_gate, 0, restore)

        indexer_cache = self._bound_cache(self.k_cache)
        tail_cache = self._bound_cache(self.tail_cache)
        positions = indexer_metadata.positions[: k.shape[0]]
        result = self.indexer_op(
            k,
            q_values,
            weights,
            positions,
            indexer_cache,
            tail_cache,
            indexer_metadata,
            tail_metadata,
            gate_score=gate_score,
            compress_ape=self.index_kpool_compress_ape,
            index_kpool=self.index_kpool,
            max_pool_seq_len=(
                indexer_metadata.block_table.shape[1] * indexer_cache.shape[1]
                if context.cudagraph_runtime_mode == CUDAGraphMode.FULL or indexer_metadata.seq_lens_cpu is None
                else int(indexer_metadata.seq_lens_cpu.max())
                if indexer_metadata.seq_lens_cpu.numel()
                else 0
            ),
            compute_topk=compute_topk,
            output_buffer=self.topk_indices_buffer,
            # FULL graphs retain padded token rows and replay the captured
            # dispatch. Keep paged reads independent of the live query count.
            allow_cache_packing=context.cudagraph_runtime_mode != CUDAGraphMode.FULL,
            query_metadata=indexer_metadata.query_metadata,
            write_k=write_k,
            write_gate_score=write_gate_score,
            write_positions=indexer_metadata.write_positions,
            write_cum_query_lens=indexer_metadata.write_cum_query_lens,
            write_raw_seq_lens=indexer_metadata.write_raw_seq_lens,
            write_indexer_slot_mapping=indexer_metadata.write_slot_mapping,
            write_tail_slot_mapping=tail_metadata.write_slot_mapping,
            write_tail_block_table=tail_metadata.write_block_table,
        )
        return result
