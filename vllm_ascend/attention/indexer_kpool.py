# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent pooled-indexer and compressor-state metadata for GLM-Next."""

from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolStateSpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
)
from vllm_ascend.models.glm5next.kv_cache import (
    format_indexer_kpool_slot_mapping,
)

GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE = 128
INDEXER_KPOOL_MAX_BLOCK_SIZE = 1024
INDEXER_KPOOL_BLOCK_ALIGNMENT = 16


def select_indexer_block_size(storage_block_size: int) -> tuple[int, int]:
    """Select a CANN-compatible physical block size for the indexer cache.

    ``pool_key_indexer`` requires the ``PA_BBND`` block dimension to be a
    multiple of 16 and no larger than 1024. GLM-Next's logical attention block
    can be much larger than 1024 after compression, so split one storage block
    into several CANN-compatible sub-blocks when necessary.

    Returns ``(block_size, blocks_per_logical_block)``.
    """
    if storage_block_size <= 0:
        raise ValueError(f"Indexer KPool storage block size must be positive, got {storage_block_size}.")
    if storage_block_size <= INDEXER_KPOOL_MAX_BLOCK_SIZE and storage_block_size % INDEXER_KPOOL_BLOCK_ALIGNMENT == 0:
        return storage_block_size, 1
    max_candidate = min(storage_block_size, INDEXER_KPOOL_MAX_BLOCK_SIZE)
    max_candidate -= max_candidate % INDEXER_KPOOL_BLOCK_ALIGNMENT
    for candidate in range(
        max_candidate,
        INDEXER_KPOOL_BLOCK_ALIGNMENT - 1,
        -INDEXER_KPOOL_BLOCK_ALIGNMENT,
    ):
        if storage_block_size % candidate == 0:
            return candidate, storage_block_size // candidate
    raise ValueError(
        "Indexer KPool storage block size "
        f"{storage_block_size} cannot be split into a block size that is a "
        f"multiple of {INDEXER_KPOOL_BLOCK_ALIGNMENT} and no larger than "
        f"{INDEXER_KPOOL_MAX_BLOCK_SIZE}."
    )


@dataclass
class AscendIndexerKPoolMetadata:
    """Metadata for compressed indexer cache writes and top-k reads."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    positions: torch.Tensor
    block_size: int
    compress_ratio: int
    cache_role: str = "indexer"
    cum_query_lens: torch.Tensor | None = None
    raw_seq_lens: torch.Tensor | None = None
    num_actual_tokens: int = 0


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
        self.logical_block_size = kv_cache_spec.block_size
        self.storage_block_size = get_storage_block_size(kv_cache_spec)
        self.compress_ratio = compress_ratio
        if self.logical_block_size % GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE:
            raise ValueError(
                "GLM-Next logical block size must be divisible by the SFA "
                f"kernel block size: logical={self.logical_block_size}, "
                f"kernel={GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE}."
            )
        self.kernel_blocks_per_logical_block = self.logical_block_size // GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE
        self.indexer_block_size, self.indexer_blocks_per_logical_block = select_indexer_block_size(
            self.storage_block_size
        )
        scheduler_config = vllm_config.scheduler_config
        # ACLGraph replay keeps the addresses captured on the first run. The
        # derived compressed metadata therefore needs persistent storage that
        # is refreshed in place on every builder invocation.
        self._slot_mapping_buffer = torch.empty(
            scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._seq_lens_buffer = torch.empty(
            scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=device,
        )
        self._cum_query_lens_buffer = torch.empty(
            scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=device,
        )
        self._raw_seq_lens_buffer = torch.empty(
            scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=device,
        )
        max_logical_blocks = cdiv(
            vllm_config.model_config.max_model_len,
            self.logical_block_size,
        )
        self._block_table_buffer = torch.empty(
            scheduler_config.max_num_seqs,
            max_logical_blocks * self.indexer_blocks_per_logical_block,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendIndexerKPoolMetadata:
        del common_prefix_len, fast_build, kwargs
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        positions = common_attn_metadata.positions[:num_input_tokens].long()
        slot_mapping = self._slot_mapping_buffer[:num_input_tokens]
        slot_mapping.copy_(
            format_indexer_kpool_slot_mapping(
                common_attn_metadata.slot_mapping[:num_input_tokens],
                positions,
                self.logical_block_size,
                self.compress_ratio,
            )
        )
        seq_lens = self._seq_lens_buffer[:num_reqs]
        torch.div(
            common_attn_metadata.seq_lens[:num_reqs],
            self.compress_ratio,
            rounding_mode="floor",
            out=seq_lens,
        )
        cum_query_lens = self._cum_query_lens_buffer[:num_reqs]
        cum_query_lens.copy_(common_attn_metadata.query_start_loc[: num_reqs + 1][1:])
        raw_seq_lens = self._raw_seq_lens_buffer[:num_reqs]
        raw_seq_lens.copy_(common_attn_metadata.seq_lens[:num_reqs])
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].to("cpu")
        seq_lens_cpu = torch.div(
            seq_lens_cpu,
            self.compress_ratio,
            rounding_mode="floor",
        )
        expanded_block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        split = self.kernel_blocks_per_logical_block
        if expanded_block_table.shape[1] % split:
            raise ValueError(
                "GLM-Next indexer received a partially expanded SFA block "
                f"table: width={expanded_block_table.shape[1]}, split={split}."
            )
        logical_width = expanded_block_table.shape[1] // split
        expanded_width = logical_width * self.indexer_blocks_per_logical_block
        if expanded_width > self._block_table_buffer.shape[1]:
            raise ValueError(
                "GLM-Next indexer block table exceeds its persistent buffer: "
                f"required={expanded_width}, capacity="
                f"{self._block_table_buffer.shape[1]}."
            )
        block_table = self._block_table_buffer[:num_reqs, :expanded_width]
        # The common full-group table is expanded for the C128 SFA kernel:
        # scheduler block N becomes [split*N, ..., split*N+split-1]. The
        # compressed indexer owns one physical page per scheduler block, so it
        # must recover N rather than treating the SFA sub-blocks as pages.
        base = torch.div(
            expanded_block_table[:, ::split],
            split,
            rounding_mode="floor",
        )
        k = self.indexer_blocks_per_logical_block
        if k == 1:
            block_table.copy_(base)
        else:
            base_repeated = base.repeat_interleave(k, dim=1)
            offsets = (
                torch.arange(
                    expanded_width,
                    dtype=torch.int32,
                    device=base.device,
                )
                % k
            )
            valid = base_repeated >= 0
            block_table.copy_(
                torch.where(
                    valid,
                    base_repeated * k + offsets,
                    torch.full_like(base_repeated, -1),
                )
            )
        return AscendIndexerKPoolMetadata(
            block_table=block_table,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            block_size=self.indexer_block_size,
            compress_ratio=self.compress_ratio,
            cum_query_lens=cum_query_lens,
            raw_seq_lens=raw_seq_lens,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
        )


class AscendIndexerKPoolBackend(AttentionBackend):
    """Cache-only backend for the compressed indexer keys."""

    @staticmethod
    def get_impl_cls():
        return None

    @staticmethod
    def get_name() -> str:
        return "ASCEND_INDEXER_KPOOL"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The scheduler manages logical token blocks; the physical cache uses
        # storage_block_size, split into CANN-compatible sub-blocks when passed
        # to pool_key_indexer.
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
    ) -> tuple[int, ...]:
        del cache_type
        if num_kv_heads != 1:
            raise ValueError(f"Indexer KPool cache requires one KV head, got {num_kv_heads}.")
        return (num_blocks, block_size, num_kv_heads, head_size)


@dataclass
class AscendIndexerKPoolStateMetadata:
    """Addressing required to update the compressor state cache."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    cache_role: str


class AscendIndexerKPoolStateMetadataBuilder(AttentionMetadataBuilder):
    """Build independent metadata for the GLM-Next compressor state."""

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        # Full-graph state writes use the fixed-shape sentinel path. Do not let
        # the base class default NEVER downgrade FULL_DECODE_ONLY for the main
        # model merely because this cache-only builder is in the cache group.
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AscendIndexerKPoolStateSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        if not isinstance(kv_cache_spec, AscendIndexerKPoolStateSpec):
            raise TypeError(
                "Ascend Indexer KPool state backend requires "
                f"AscendIndexerKPoolStateSpec, got {type(kv_cache_spec).__name__}."
            )
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.cache_role = kv_cache_spec.cache_role

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendIndexerKPoolStateMetadata:
        del common_prefix_len, fast_build, kwargs
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        return AscendIndexerKPoolStateMetadata(
            block_table=common_attn_metadata.block_table_tensor[:num_reqs],
            slot_mapping=common_attn_metadata.slot_mapping[:num_input_tokens],
            block_size=self.block_size,
            cache_role=self.cache_role,
        )


class AscendIndexerKPoolStateBackend(AttentionBackend):
    """Cache-only backend for the GLM-Next compressor state."""

    @staticmethod
    def get_impl_cls():
        return None

    @staticmethod
    def get_name() -> str:
        return "ASCEND_INDEXER_KPOOL_STATE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The state page follows index_kpool and is independent of SFA C128.
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type[AscendIndexerKPoolStateMetadataBuilder]:
        return AscendIndexerKPoolStateMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        del cache_type
        if num_kv_heads != 1:
            raise ValueError(f"Indexer KPool state cache requires one KV head, got {num_kv_heads}.")
        return (num_blocks, block_size, head_size)
