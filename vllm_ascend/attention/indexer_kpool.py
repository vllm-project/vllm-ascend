# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache metadata and execution backends for the GLM-Next pooled indexer."""

from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
)
from vllm_ascend.device.hardware_profile import AttentionBackendFamily, get_current_hardware_profile
from vllm_ascend.utils import _round_up, enable_dsa_cp, vllm_version_is

GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE = 128


@dataclass
class AscendIndexerKPoolQueryMetadata:
    """Token-local query addressing over the replicated pool cache."""

    positions: torch.Tensor
    cum_query_lens: torch.Tensor
    num_actual_tokens: int
    seq_lens: torch.Tensor
    pool_tail: torch.Tensor


@dataclass
class AscendIndexerKPoolMetadata:
    """Metadata for compressed indexer cache writes and top-k reads."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    block_size: int
    compress_ratio: int
    cache_role: str = "indexer"
    cum_query_lens: torch.Tensor | None = None
    raw_seq_lens: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    num_actual_tokens: int = 0
    query_metadata: AscendIndexerKPoolQueryMetadata | None = None
    query_start_loc: torch.Tensor | None = None
    start_pos: torch.Tensor | None = None
    pool_tail: torch.Tensor | None = None
    keypool_state_table: torch.Tensor | None = None
    keypool_start_offsets: torch.Tensor | None = None
    pooled_slot_mapping: torch.Tensor | None = None


@dataclass
class _KPoolMetadataBuffers:
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    cum_query_lens: torch.Tensor
    raw_seq_lens: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    start_pos: torch.Tensor
    pool_tail: torch.Tensor
    start_offsets: torch.Tensor
    token_indices: torch.Tensor
    pooled_slot_mapping: torch.Tensor


class AscendIndexerKPoolMetadataBuilder(AttentionMetadataBuilder):
    """Build pool-level addressing for the compressed indexer cache."""

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
        if not isinstance(kv_cache_spec, MLAAttentionSpec):
            raise TypeError(
                f"Ascend Indexer KPool backend requires MLAAttentionSpec, got {type(kv_cache_spec).__name__}."
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
        if self.logical_block_size % GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE:
            raise ValueError(
                "GLM-Next logical block size must be divisible by the SFA "
                f"kernel block size: logical={self.logical_block_size}, "
                f"kernel={GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE}."
            )
        self.kernel_row_block_size = GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE // self.compress_ratio
        # MTP tables count expanded kernel pages.
        self.kernel_blocks_per_logical_block = self.logical_block_size // GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE
        scheduler_config = vllm_config.scheduler_config
        self.use_dsa_cp = enable_dsa_cp()
        self.dsa_cp_size = get_tp_group().world_size if self.use_dsa_cp else 1
        self.dsa_cp_rank = get_tp_group().rank_in_group if self.use_dsa_cp else 0
        self._max_num_batched_tokens = _round_up(scheduler_config.max_num_batched_tokens, self.dsa_cp_size)
        # FULL draft graphs can pad requests beyond max_num_seqs.
        self._max_num_metadata_reqs = max(scheduler_config.max_num_seqs, self._max_num_batched_tokens)
        # Keep captured storage isolated between MTP draft steps.
        self._metadata_buffers: dict[int, _KPoolMetadataBuffers] = {}
        self._dsa_cp_query_buffers: dict[int, torch.Tensor] = {}
        self._captured_state_tables: dict[tuple[int, int, int], torch.Tensor] = {}
        self._state_table_scratch: dict[int, torch.Tensor] = {}
        self._state_table_shapes: dict[int, tuple[int, int]] = {}

    def _build_query_metadata(
        self, common: CommonAttentionMetadata, positions: torch.Tensor
    ) -> AscendIndexerKPoolQueryMetadata | None:
        if not self.use_dsa_cp:
            return None
        local_tokens = positions.shape[0] // self.dsa_cp_size
        local_start = self.dsa_cp_rank * local_tokens
        local_end = local_start + local_tokens
        key = common.slot_mapping.data_ptr()
        if key not in self._dsa_cp_query_buffers:
            self._dsa_cp_query_buffers[key] = torch.empty(
                3, self._max_num_metadata_reqs, dtype=torch.int64, device=self.device
            )
        query_lens, seq_lens, pool_tail = self._dsa_cp_query_buffers[key][:, : common.num_reqs]
        return AscendIndexerKPoolQueryMetadata(
            positions=positions[local_start:local_end],
            cum_query_lens=query_lens,
            num_actual_tokens=max(min(local_end, common.num_actual_tokens) - local_start, 0),
            seq_lens=seq_lens,
            pool_tail=pool_tail,
        )

    def _get_metadata_buffers(self, common_attn_metadata: CommonAttentionMetadata) -> _KPoolMetadataBuffers:
        key = common_attn_metadata.slot_mapping.data_ptr()
        buffers = self._metadata_buffers.get(key)
        if buffers is None:
            reqs, tokens = self._max_num_metadata_reqs, self._max_num_batched_tokens
            buffers = _KPoolMetadataBuffers(
                slot_mapping=torch.empty(tokens, dtype=torch.int64, device=self.device),
                seq_lens=torch.empty(reqs, dtype=torch.int64, device=self.device),
                cum_query_lens=torch.empty(reqs, dtype=torch.int64, device=self.device),
                raw_seq_lens=torch.empty(reqs, dtype=torch.int32, device=self.device),
                positions=torch.empty(tokens, dtype=torch.int64, device=self.device),
                query_start_loc=torch.empty(reqs + 1, dtype=torch.int32, device=self.device),
                start_pos=torch.empty(reqs, dtype=torch.int32, device=self.device),
                pool_tail=torch.empty(reqs, dtype=torch.int64, device=self.device),
                start_offsets=torch.empty(reqs, dtype=torch.int32, device=self.device),
                token_indices=torch.arange(tokens, dtype=torch.int32, device=self.device),
                pooled_slot_mapping=torch.empty(tokens + 1, dtype=torch.int64, device=self.device),
            )
            self._metadata_buffers[key] = buffers
        return buffers

    def _get_state_table(self, common: CommonAttentionMetadata, graph_capture: bool) -> torch.Tensor:
        # Capture owns each table; eager shapes reuse one growing allocation.
        step = common.slot_mapping.data_ptr()
        width = getattr(common, "max_query_len", common.num_input_tokens)
        cols = 1 + (width + self.compress_ratio - 1) // self.compress_ratio
        shape = (common.num_reqs, cols)
        key = (step, *shape)
        table = self._captured_state_tables.get(key)
        initialize = table is None
        if table is None:
            if graph_capture:
                table = torch.empty(shape, dtype=torch.int32, device=self.device)
                self._captured_state_tables[key] = table
            else:
                size = common.num_reqs * cols
                scratch = self._state_table_scratch.get(step)
                if scratch is None or scratch.numel() < size:
                    scratch = torch.empty(size, dtype=torch.int32, device=self.device)
                    self._state_table_scratch[step] = scratch
                table = scratch[:size].view(shape)
                initialize = self._state_table_shapes.get(step) != shape
                self._state_table_shapes[step] = shape
        if initialize:
            # This table depends only on shape, so initialize it once.
            host = torch.empty(shape, dtype=torch.int32, pin_memory=self.device.type != "cpu")
            rows = torch.arange(1, common.num_reqs + 1, dtype=torch.int32)
            host.copy_((rows + common.num_reqs)[:, None].expand(shape))
            host[:, 0].copy_(rows)
            table.copy_(host, non_blocking=True)
        return table

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendIndexerKPoolMetadata:
        graph_capture = kwargs.pop("_graph_capture", False)
        del common_prefix_len, fast_build, kwargs
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        buffers = self._get_metadata_buffers(common_attn_metadata)
        num_cache_tokens = _round_up(num_input_tokens, self.dsa_cp_size)
        num_source_tokens = min(
            num_input_tokens, common_attn_metadata.positions.shape[0], common_attn_metadata.slot_mapping.shape[0]
        )
        if num_source_tokens < common_attn_metadata.num_actual_tokens:
            raise ValueError("KPool positions and slot mapping must cover every actual token.")
        from vllm_ascend.ops.triton.glm5_next_cann_keypool import build_keypool_metadata  # type: ignore[import-untyped]

        state_table = self._get_state_table(common_attn_metadata, graph_capture)
        query_metadata = self._build_query_metadata(common_attn_metadata, buffers.positions[:num_cache_tokens])
        (
            positions,
            slot_mapping,
            seq_lens,
            raw_seq_lens,
            query_start_loc,
            cum_query_lens,
            start_pos,
            pool_tail,
        ) = build_keypool_metadata(
            common_attn_metadata,
            buffers,
            num_cache_tokens,
            num_source_tokens,
            self.logical_block_size,
            self.compress_ratio,
            query_metadata=query_metadata,
            local_start=self.dsa_cp_rank * (num_cache_tokens // self.dsa_cp_size),
        )
        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        return AscendIndexerKPoolMetadata(
            block_table=block_table,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            positions=positions,
            block_size=self.kernel_row_block_size,
            compress_ratio=self.compress_ratio,
            cum_query_lens=cum_query_lens,
            raw_seq_lens=raw_seq_lens,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=query_start_loc,
            start_pos=start_pos,
            pool_tail=pool_tail,
            pooled_slot_mapping=buffers.pooled_slot_mapping[
                1 : 1 + min(num_cache_tokens, num_cache_tokens // self.compress_ratio + num_reqs)
            ],
            keypool_state_table=state_table,
            keypool_start_offsets=buffers.start_offsets[:num_reqs],
            query_metadata=query_metadata,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata) -> AscendIndexerKPoolMetadata:
        return self.build_for_graph_capture(common_attn_metadata)

    def build_for_graph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        attn_state: Any = None,
        **kwargs,
    ) -> AscendIndexerKPoolMetadata:
        del attn_state
        return self.build(0, common_attn_metadata, _graph_capture=True, **kwargs)

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
    def supports_batch_invariance(cls) -> bool:
        # vLLM #60271 added the batch-invariance support check in
        # `init_attn_backend` on main; the v0.31.0 lane has no such check.
        return not vllm_version_is("0.31.0")

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
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
    cann_inputs: Any = None


class AscendIndexerKPoolTailMetadataBuilder(AttentionMetadataBuilder):
    """Build independent metadata for the GLM-Next compressor tail."""

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
        # Persistent rings are separate from temporary CANN state pages.
        self._max_num_metadata_reqs = max(
            vllm_config.scheduler_config.max_num_seqs, vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._tail_inputs: dict[int, dict[str, torch.Tensor]] = {}
        self._pool_size = kv_cache_spec.compress_ratio
        self._page_rows = kv_cache_spec.page_size_bytes // (2 * kv_cache_spec.head_size * 4)
        self._lookahead = self.block_size > self._pool_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendIndexerKPoolTailMetadata:
        del common_prefix_len, fast_build, kwargs
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        key = common_attn_metadata.slot_mapping.data_ptr()
        block_table = common_attn_metadata.block_table_tensor[:num_reqs, :1]
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        from vllm_ascend.ops.triton.glm5_next_cann_keypool import build_tail_inputs  # type: ignore[import-untyped]

        if key not in self._tail_inputs:
            n, width = self._max_num_metadata_reqs, self.block_size
            self._tail_inputs[key] = {
                "history": torch.empty(n, self._pool_size, dtype=torch.int64, device=self.device),
                **{
                    name: torch.empty(n, width, dtype=torch.int64, device=self.device)
                    for name in ("writes", "state", "retained")
                },
            }
        retained_width = (
            min(self.block_size, getattr(common_attn_metadata, "max_query_len", self.block_size))
            if self._lookahead
            else 0
        )
        cann_inputs = build_tail_inputs(
            common_attn_metadata,
            self._pool_size,
            self.block_size,
            self._page_rows,
            max(1, retained_width) if self._lookahead else 0,
            self._tail_inputs[key],
        )
        return AscendIndexerKPoolTailMetadata(
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            cann_inputs=cann_inputs,
        )

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
    def supports_batch_invariance(cls) -> bool:
        # vLLM #60271 added the batch-invariance support check in
        # `init_attn_backend` on main; the v0.31.0 lane has no such check.
        return not vllm_version_is("0.31.0")

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
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
        return (num_blocks, block_size, 2 * head_size)


class Glm5NextKPoolIndexerBackend(nn.Module):
    """Model-side implementation of the unified seven-argument indexer API.

    The cache-only backends above own the engine-facing attention contracts.
    Keep execution independent of the standard SFA indexer and its operators.
    """

    def __init__(self, vllm_indexer: nn.Module, qk_rope_head_dim: int) -> None:
        super().__init__()
        if qk_rope_head_dim != 0:
            raise ValueError(
                f"GLM-Next KPool indexing supports NoPE queries only, got qk_rope_head_dim={qk_rope_head_dim}."
            )
        parallel_config = vllm_indexer.vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1 or parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("GLM-Next KPool indexing does not support PCP or DCP.")

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
        self._init_cann_ops()
        for name in (
            "_key_weight",
            "_gate_weight",
            "_head_weight_f32",
            "_norm_weight_f32",
            "_norm_bias_f32",
        ):
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
        self._key_weight = self.wk_weights_proj.weight[: self.head_dim].detach().contiguous()
        self._gate_weight = self.index_kpool_compress_gate.detach().to(self._key_weight.dtype).contiguous()
        self._head_weight_f32 = self.wk_weights_proj.weight[self.head_dim :].detach().float().contiguous()
        self._norm_weight_f32 = self.k_norm.weight.detach().float() if self.k_norm.weight is not None else None
        self._norm_bias_f32 = self.k_norm.bias.detach().float() if self.k_norm.bias is not None else None

    @staticmethod
    def _bound_cache(layer: Any) -> torch.Tensor:
        context = get_forward_context()
        cache = layer.kv_cache
        if isinstance(cache, list):
            virtual_engine = getattr(context, "virtual_engine", 0) or 0
            if virtual_engine >= len(cache):
                raise IndexError(f"Cache virtual engine {virtual_engine} is out of range.")
            cache = cache[virtual_engine]
        if isinstance(cache, (list, tuple)):
            if len(cache) != 1:
                raise TypeError("GLM KPool cache requires one key tensor.")
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
        hidden = hidden_states[:num_tokens]
        k_hidden = k_hidden_states[:num_tokens]
        if self._key_weight is None:
            self.process_weights_after_loading()
        assert self._key_weight is not None
        q_values = None
        weights = None
        if compute_topk:
            if isinstance(q_c, tuple):
                raise TypeError("GLM KPool backend requires an unquantized q_c tensor.")
            q_values = self.wq_b(q_c[:num_tokens])[0].view(num_tokens, self.n_head, self.head_dim)
            weights = F.linear(hidden.float(), self._head_weight_f32).to(q_values.dtype)
            weights = weights * (self.softmax_scale * self.n_head**-0.5)

        indexer_cache = self._bound_cache(self.k_cache)
        tail_cache = self._bound_cache(self.tail_cache)
        if indexer_metadata.query_metadata is not None:
            k_hidden = get_tp_group().all_gather(k_hidden, dim=0)
        return self._forward_cann(
            k_hidden,
            q_values,
            weights,
            indexer_cache,
            tail_cache,
            indexer_metadata,
            tail_metadata,
            compute_topk=compute_topk,
        )

    def _init_cann_ops(self) -> None:
        try:
            import_module("cann_ops_transformer")
            adapters = import_module("vllm_ascend.ops.cann_keypool")
            fused = import_module("vllm_ascend.ops.triton.glm5_next_cann_keypool")
            self._gather_retained_hidden = adapters.gather_retained_hidden
            self._pack_keypool_indices = fused.pack_keypool_indices
            self._prepare_keypool_state = adapters.prepare_keypool_state
            self._write_keypool_caches = fused.write_keypool_caches
            self._key_pool = torch.ops.cann_ops_transformer.key_pool
            self._pool_key_indexer = partial(
                adapters.pool_key_indexer_nonempty,
                torch.ops.cann_ops_transformer.pool_key_indexer,
            )
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "GLM-5.3-Flash KPool requires cann_ops_transformer with key_pool and pool_key_indexer, "
                "the matching CANN ops-transformer operator package and Triton cache/index adapters."
            ) from exc

    def _forward_cann(
        self,
        hidden_states: torch.Tensor,
        q_values: torch.Tensor | None,
        weights: torch.Tensor | None,
        indexer_cache: torch.Tensor,
        tail_cache: torch.Tensor,
        indexer_metadata: AscendIndexerKPoolMetadata,
        tail_metadata: AscendIndexerKPoolTailMetadata,
        *,
        compute_topk: bool,
    ) -> torch.Tensor | None:
        query_metadata = indexer_metadata.query_metadata or indexer_metadata
        num_tokens = (
            query_metadata.positions.shape[0] if indexer_metadata.query_metadata is not None else hidden_states.shape[0]
        )
        output_width = self.topk_tokens + self.index_kpool - 1
        if compute_topk and self.topk_indices_buffer is not None:
            if (
                self.topk_indices_buffer.ndim != 2
                or self.topk_indices_buffer.dtype != torch.int32
                or self.topk_indices_buffer.device != hidden_states.device
                or self.topk_indices_buffer.stride(1) != 1
                or (num_tokens > 1 and self.topk_indices_buffer.stride(0) < self.topk_indices_buffer.shape[1])
            ):
                raise ValueError("GLM KPool output buffer requires int32 contiguous columns on the query device.")
            if self.topk_indices_buffer.shape[0] < num_tokens or self.topk_indices_buffer.shape[1] < output_width:
                raise ValueError("GLM KPool output buffer must cover the token rows and top-k width.")
        if self.index_kpool <= 0 or self.topk_tokens <= 0 or self.topk_tokens % self.index_kpool:
            raise ValueError("KPool top-k must be divisible by its positive pool size.")
        if num_tokens == 0:
            if not compute_topk:
                return None
            if self.topk_indices_buffer is not None:
                return self.topk_indices_buffer[:0, :output_width].unsqueeze(1)
            return torch.empty((0, 1, output_width), dtype=torch.int32, device=hidden_states.device)
        if (
            indexer_metadata.cum_query_lens is None
            or indexer_metadata.query_start_loc is None
            or indexer_metadata.start_pos is None
            or indexer_metadata.pool_tail is None
            or indexer_metadata.pooled_slot_mapping is None
            or tail_metadata.cann_inputs is None
        ):
            raise ValueError("GLM KPool metadata requires CANN request boundaries and pooled-key row indices.")
        if indexer_cache.dtype != torch.bfloat16:
            raise TypeError("GLM KPool compressed cache must be bfloat16.")
        if tail_cache.dtype != torch.float32:
            raise TypeError("GLM KPool compressor tail must be float32.")
        if tail_cache.ndim != 3 or tail_cache.shape[1] < self.index_kpool or tail_cache.shape[2] != 2 * self.head_dim:
            raise ValueError("GLM KPool tail requires [blocks, capacity, 2 * head_dim] K/gate storage.")
        if tail_metadata.block_size != tail_cache.shape[1]:
            raise ValueError("GLM KPool tail metadata capacity must match the bound cache.")
        if (
            self.index_kpool_compress_ape.shape != (self.index_kpool, self.head_dim)
            or self.index_kpool_compress_ape.dtype != torch.float32
        ):
            raise ValueError("GLM KPool APE must be FP32 with shape [pool_size, head_dim].")

        if compute_topk and (q_values is None or weights is None):
            raise ValueError("GLM KPool top-k requires query and head weights.")

        # Keep history separate: CANN can save the tail before other cores
        # finish reading history. Whole-pool shifts preserve the pooling result.
        norm_eps = getattr(self.k_norm, "eps", getattr(self.k_norm, "variance_epsilon", 1e-6))
        state = self._prepare_keypool_state(tail_cache, indexer_metadata.start_pos, tail_metadata, self.index_kpool)
        pooled_key = self._key_pool(
            hidden_states,
            self._key_weight,
            self._gate_weight,
            self.index_kpool_compress_ape,
            state,
            indexer_metadata.keypool_state_table,
            indexer_metadata.keypool_start_offsets,
            norm_weight=self._norm_weight_f32,
            norm_bias=self._norm_bias_f32,
            cu_seqlens=indexer_metadata.query_start_loc,
            cmp_ratio=self.index_kpool,
            norm_eps=norm_eps,
        )
        tail_key = tail_gate = None
        if tail_metadata.cann_inputs.retained_indices is not None:
            # Re-project the live MTP rollback window missing from CANN state.
            tail_hidden = self._gather_retained_hidden(hidden_states, tail_metadata.cann_inputs.retained_indices)
            tail_key = F.linear(tail_hidden, self._key_weight).float()
            if self._norm_weight_f32 is not None:
                tail_key = F.layer_norm(
                    tail_key, (self.head_dim,), self._norm_weight_f32, self._norm_bias_f32, norm_eps
                )
            tail_gate = F.linear(tail_hidden, self._gate_weight).float()
        self._write_keypool_caches(
            tail_cache,
            indexer_cache,
            tail_key,
            tail_gate,
            pooled_key,
            indexer_metadata,
            tail_metadata,
            state,
        )
        # Sharing top-k still advances the compressed cache and raw tail.
        if not compute_topk:
            return None
        # Empty DSA-CP shards must update caches but skip CANN's unsafe no-work path.
        # FULL capture retains selection because replay may have live queries.
        if query_metadata.num_actual_tokens == 0 and get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL:
            if self.topk_indices_buffer is not None:
                output = self.topk_indices_buffer[:num_tokens]
                output.fill_(-1)
                return output[:, :output_width].unsqueeze(1)
            return torch.full((num_tokens, 1, output_width), -1, dtype=torch.int32, device=hidden_states.device)
        assert q_values is not None and weights is not None
        query = q_values.to(indexer_cache.dtype)
        indices, _ = self._pool_key_indexer(
            query,
            indexer_cache,
            weights.to(torch.bfloat16),
            query_metadata.pool_tail,
            actual_seq_q=query_metadata.cum_query_lens,
            actual_seq_k=query_metadata.seq_lens,
            block_table=indexer_metadata.block_table,
            layout_q="TND",
            layout_k="PA_BBND",
            topk=self.topk_tokens,
            pool_size=self.index_kpool,
            mask_mode=3,
        )
        return self._pack_keypool_indices(
            indices, query_metadata, self.topk_tokens, self.index_kpool, self.topk_indices_buffer
        )
