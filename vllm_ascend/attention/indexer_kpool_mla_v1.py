# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Indexer KPool MLA attention backend for GLM-5 Next."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
)
from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolStateSpec,
    format_indexer_kpool_slot_mapping,
)
from vllm_ascend.device.device_op import DeviceOperator

INDEXER_KPOOL_MLA_SPARSE_ATTN_QUERY_CHUNK_SIZE = 16
INDEXER_KPOOL_MLA_SAS_METADATA_SIZE = 1024
GLM5_SFA_KERNEL_BLOCK_SIZE = 128


@dataclass
class AscendIndexerKPoolMLAMetadata(AscendSFAMetadata):
    """SFA metadata for the main MLA KV cache."""

    cache_role: str = "kv"
    sas_metadata: torch.Tensor | None = None
    sas_sinks: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None


@dataclass
class AscendIndexerKPoolMetadata:
    """Minimal metadata for compressed-indexer-K reads/writes and top-k."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    positions: torch.Tensor
    block_size: int
    compress_ratio: int
    cache_role: str = "indexer"


class AscendIndexerKPoolMetadataBuilder(AttentionMetadataBuilder):
    """Builds pool-granular addressing for the compressed indexer K cache."""

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
                "Ascend Indexer KPool backend requires MLAAttentionSpec, "
                f"got {type(kv_cache_spec).__name__}."
            )
        if _spec_compress_ratio(kv_cache_spec) <= 1:
            raise ValueError(
                "Ascend Indexer KPool cache requires compress_ratio > 1, "
                f"got {_spec_compress_ratio(kv_cache_spec)}."
            )
        if not layer_names or any(not name.endswith(".indexer.k_cache") for name in layer_names):
            raise ValueError(f"Invalid Indexer KPool cache layer names: {layer_names}.")
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.logical_block_size = kv_cache_spec.block_size
        self.storage_block_size = kv_cache_spec.storage_block_size
        self.compress_ratio = _spec_compress_ratio(kv_cache_spec)
        if self.logical_block_size % GLM5_SFA_KERNEL_BLOCK_SIZE:
            raise ValueError(
                "GLM-5 logical block size must be divisible by the SFA "
                f"kernel block size: logical={self.logical_block_size}, "
                f"kernel={GLM5_SFA_KERNEL_BLOCK_SIZE}."
            )
        self.kernel_blocks_per_logical_block = (
            self.logical_block_size // GLM5_SFA_KERNEL_BLOCK_SIZE
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
        # The upstream SFA block table is expanded at kernel-block
        # granularity from SCHEDULER blocks: cdiv(max_len, scheduler_bs) *
        # split, which exceeds cdiv(max_len, kernel_bs) by one whenever
        # max_len is not scheduler-block aligned (e.g. 200000 -> 1564 vs
        # 1563). Size for the larger of the two plus slack so unaligned
        # max_model_len values cannot overflow the persistent buffer.
        max_logical_blocks = (
            max(
                cdiv(
                    vllm_config.model_config.max_model_len,
                    self.logical_block_size,
                ),
                cdiv(
                    vllm_config.model_config.max_model_len,
                    GLM5_SFA_KERNEL_BLOCK_SIZE,
                ),
            )
            + 1
        )
        self._block_table_buffer = torch.empty(
            scheduler_config.max_num_seqs,
            max_logical_blocks,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendIndexerKPoolMetadata:
        del common_prefix_len, fast_build
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
        expanded_block_table = common_attn_metadata.block_table_tensor[
            :num_reqs
        ]
        split = self.kernel_blocks_per_logical_block
        if expanded_block_table.shape[1] % split:
            raise ValueError(
                "GLM-5 indexer received a partially expanded SFA block "
                f"table: width={expanded_block_table.shape[1]}, split={split}."
            )
        logical_width = expanded_block_table.shape[1] // split
        if logical_width > self._block_table_buffer.shape[1]:
            raise ValueError(
                "GLM-5 indexer block table exceeds its persistent buffer: "
                f"required={logical_width}, capacity="
                f"{self._block_table_buffer.shape[1]}."
            )
        block_table = self._block_table_buffer[
            :num_reqs, :logical_width
        ]
        # The common full-group table is expanded for the C128 SFA kernel:
        # scheduler block N becomes [split*N, ..., split*N+split-1]. The
        # compressed indexer owns one physical page per scheduler block, so it
        # must recover N rather than treating the SFA sub-blocks as pages.
        torch.div(
            expanded_block_table[:, ::split],
            split,
            rounding_mode="floor",
            out=block_table,
        )
        return AscendIndexerKPoolMetadata(
            block_table=block_table,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )


class AscendIndexerKPoolBackend(AttentionBackend):
    """Compressed indexer K cache backend (cache management only, no attention)."""

    @staticmethod
    def get_name() -> str:
        return "ASCEND_INDEXER_KPOOL"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The scheduler manages logical token blocks; the physical cache uses
        # storage_block_size directly.
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
    """Addressing info for compressor tail cache reads/writes only."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    cache_role: str


class AscendIndexerKPoolStateMetadataBuilder(AttentionMetadataBuilder):
    """Builds standalone metadata for the GLM-5 compressor tail cache."""

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
    ) -> AscendIndexerKPoolStateMetadata:
        del common_prefix_len, fast_build
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        return AscendIndexerKPoolStateMetadata(
            block_table=common_attn_metadata.block_table_tensor[:num_reqs],
            slot_mapping=common_attn_metadata.slot_mapping[:num_input_tokens],
            block_size=self.block_size,
            cache_role=self.cache_role,
        )


def _spec_compress_ratio(kv_cache_spec) -> int:
    """Resolve the kpool compress ratio from either spec spelling.

    Some Ascend specs carry ``compress_ratio`` directly; the upstream
    MLAAttentionSpec spelling of the same quantity is ``tokens_per_state``.
    """
    cr = getattr(kv_cache_spec, "compress_ratio", None)
    if cr is None:
        cr = getattr(kv_cache_spec, "tokens_per_state", 1)
    return int(cr)


class AscendIndexerKPoolStateBackend(AttentionBackend):
    """GLM-5 compressor tail cache backend (no attention computation)."""

    @staticmethod
    def get_name() -> str:
        return "ASCEND_INDEXER_KPOOL_STATE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The tail page size is set by index_kpool, not constrained by the SFA
        # C128 kernel block.
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


class AscendIndexerKPoolMLAMetadataBuilder(AscendSFAMetadataBuilder):
    """Builds metadata only for the main MLA KV cache that runs SFA."""

    def __init__(
        self,
        kv_cache_spec: MLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        **kwargs,
    ) -> None:
        if not isinstance(kv_cache_spec, MLAAttentionSpec):
            raise TypeError(
                "Ascend Indexer KPool MLA backend requires MLAAttentionSpec, "
                f"got {type(kv_cache_spec).__name__}."
            )
        if _spec_compress_ratio(kv_cache_spec) != 1:
            raise ValueError(
                "Main Indexer KPool MLA cache must be uncompressed, "
                f"got compress_ratio={_spec_compress_ratio(kv_cache_spec)}."
            )
        if not layer_names or any(not name.endswith(".attn") for name in layer_names):
            raise ValueError(f"Invalid main Indexer KPool MLA layer names: {layer_names}.")
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls=AscendIndexerKPoolMLAMetadata,
            **kwargs,
        )
        self._sas_metadata_buffer: torch.Tensor | None = None
        self._spec_sas_metadata_buffers: list[torch.Tensor] | None = None
        self._sas_sinks: torch.Tensor | None = None
        self._seqused_q: torch.Tensor | None = None
        if DeviceOperator.supports_sharedkv_indexer_kpool_mla():
            self._sas_metadata_buffer = torch.zeros(
                INDEXER_KPOOL_MLA_SAS_METADATA_SIZE,
                dtype=torch.int32,
                device=device,
            )
            num_heads_q = (
                self.model_config.hf_text_config.num_attention_heads
                // self.vllm_config.parallel_config.tensor_parallel_size
            )
            self._sas_sinks = torch.ones(
                num_heads_q, dtype=torch.float32, device=device
            )
            self._seqused_q = torch.empty(
                0,
                dtype=torch.int32,
                device=device,
            )
            if self.speculative_config is not None:
                self._spec_sas_metadata_buffers = [
                    torch.zeros(
                        INDEXER_KPOOL_MLA_SAS_METADATA_SIZE,
                        dtype=torch.int32,
                        device=device,
                    )
                    for _ in range(self.speculative_config.num_speculative_tokens)
                ]

    @classmethod
    def get_cudagraph_support(cls, vllm_config: VllmConfig, kv_cache_spec) -> AttentionCGSupport:
        # The graph path uses fixed-shape cache updates and is only valid for
        # uniform decode batches. The GLM MTP proposer is forced to eager mode
        # independently; its presence must not disable the target model graph.
        return AttentionCGSupport.UNIFORM_BATCH

    def _build(self, common_attn_metadata, draft_index: int | None = None):
        metadata = super()._build(common_attn_metadata, draft_index)
        metadata.cache_role = "kv"
        if not DeviceOperator.supports_sharedkv_indexer_kpool_mla():
            return metadata

        if draft_index is None:
            sas_metadata_buffer = self._sas_metadata_buffer
        else:
            if self._spec_sas_metadata_buffers is None:
                raise RuntimeError("Missing GLM-5 speculative sparse-attention metadata buffers.")
            sas_metadata_buffer = self._spec_sas_metadata_buffers[draft_index - 1]
        if sas_metadata_buffer is None or self._seqused_q is None:
            raise RuntimeError("Missing GLM-5 A5 sparse-attention metadata storage.")

        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        metadata.query_start_loc = query_start_loc
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        hf_config = self.model_config.hf_text_config
        metadata_op = DeviceOperator.get_sparse_attention_metadata_op_indexer_kpool_mla()
        generated_metadata = metadata_op(
            **DeviceOperator.get_sparse_attention_metadata_kwargs_indexer_kpool_mla(query_start_loc.device),
            num_heads_q=hf_config.num_attention_heads // self.vllm_config.parallel_config.tensor_parallel_size,
            num_heads_kv=1,
            head_dim=hf_config.kv_lora_rank,
            cu_seqlens_q=query_start_loc,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            seqused_q=self._seqused_q,
            seqused_ori_kv=seq_lens,
            seqused_cmp_kv=None,
            cmp_residual_kv=None,
            max_seqlen_q=query_lens.max(),
            max_seqlen_ori_kv=seq_lens.max(),
            max_seqlen_cmp_kv=0,
            batch_size=num_reqs,
            ori_topk=hf_config.index_topk + hf_config.index_kpool - 1,
            cmp_topk=0,
            cmp_ratio=1,
            ori_mask_mode=3,
            cmp_mask_mode=3,
            ori_win_left=0,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
            has_ori_kv=True,
            has_cmp_kv=False,
        )
        if generated_metadata.numel() != sas_metadata_buffer.numel():
            raise ValueError(
                "GLM-5 sparse-attention metadata must contain "
                f"{sas_metadata_buffer.numel()} int32 values, got "
                f"{generated_metadata.numel()}."
            )
        sas_metadata_buffer.copy_(generated_metadata)
        metadata.sas_metadata = sas_metadata_buffer
        metadata.sas_sinks = self._sas_sinks
        return metadata


class AscendIndexerKPoolMLABackend(AttentionBackend):
    """GLM-5 backend: SFA attention plus a kpool-compressed indexer."""

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_INDEXER_KPOOL_MLA"

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @staticmethod
    def get_builder_cls():
        return AscendIndexerKPoolMLAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls() -> type[AscendIndexerKPoolMLAImpl]:
        return AscendIndexerKPoolMLAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [GLM5_SFA_KERNEL_BLOCK_SIZE]


class AscendIndexerKPoolMLAImpl(AscendSFAImpl):
    """SFA implementation with GLM-5's softmax kpool indexer cache."""

    def __init__(self, *args, **kwargs) -> None:
        indexer_rotary_emb = kwargs.get("indexer_rotary_emb")
        super().__init__(*args, **kwargs)
        if self.indexer is None:
            raise ValueError("GLM-5 Indexer KPool MLA requires an indexer.")
        if get_ascend_config().c8_enable_reshape_optim:
            raise ValueError(
                "GLM-5 Indexer KPool MLA does not support c8_enable_reshape_optim because "
                "its indexer state cache has an independent slot mapping."
            )
        if getattr(self, "enable_dsa_cp", False):
            raise ValueError("GLM-5 Indexer KPool MLA does not yet support DSA context parallelism.")
        # Indexer KPool MLA owns three physical cache roles. Main MLA and the
        # compressed indexer share one scheduler group, while compressor state
        # uses another. SFA's fused cache paths must not reinterpret this layout.
        self.use_sparse_c8_indexer = False
        self.use_sparse_c8_sfa = False
        self.enable_sfa_prolog_v3 = False
        self.enable_mlapo = False
        self.index_kpool = self.indexer.index_kpool
        self.index_topk = self.indexer.topk_tokens
        if self.index_topk % self.index_kpool:
            raise ValueError(
                "GLM-5 Indexer KPool MLA index_topk must be divisible by "
                f"index_kpool, got {self.index_topk} and {self.index_kpool}."
            )
        self.index_kpool_compress_ape = self.indexer.index_kpool_compress_ape
        self.index_kpool_compress_gate = self.indexer.index_kpool_compress_gate
        self.is_rope_neox_style = self.indexer.is_rope_neox_style
        self.indexer_rotary_emb = indexer_rotary_emb
        self._indexer_kpool_mla_metadata: dict[str, AscendIndexerKPoolMLAMetadata] = {}
        self._indexer_kpool_mla_caches: dict[
            str,
            torch.Tensor | tuple[torch.Tensor, ...],
        ] = {}

    @staticmethod
    def _by_role(items) -> dict[str, object]:
        return {item.cache_role: item for item in items}

    def _get_indexer_slot_mapping(self, attn_metadata):
        return self._indexer_kpool_mla_metadata["indexer_state"].slot_mapping

    def _store_indexer_cache(
        self,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        # Sliding-window metadata marks evicted prompt tokens with -1. Those
        # rows are already compressed and must not be scattered into the tail
        # state page during a long/chunked prefill.
        # The state payload may be an as_strided prefix of a padded physical
        # page. Keep the block dimension explicit instead of flattening across
        # page padding with view().
        AscendIndexerKPoolMLAImpl._scatter_paged_cache(
            cache,
            slot_mapping,
            values,
            cache.shape[1],
        )

    def _get_mla_cache_views(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mla_cache = self._indexer_kpool_mla_caches["kv"]
        if isinstance(mla_cache, torch.Tensor):
            return mla_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim],
                dim=-1,
            )
        if isinstance(mla_cache, (tuple, list)) and len(mla_cache) == 2:
            return mla_cache[0], mla_cache[1]
        raise ValueError("Indexer KPool MLA cache must be a packed tensor or contain KV-latent and RoPE tensors.")

    def _pack_mla_cache_values(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Build the MLA cache values to write for this batch.

        NoPE models carry latent KV only.
        """
        kv_values = kv_c[:num_tokens].reshape(num_tokens, self.kv_lora_rank)
        if self.qk_rope_head_dim == 0:
            return kv_values
        rope_values = k_pe[:num_tokens].reshape(num_tokens, self.qk_rope_head_dim)
        return torch.cat([kv_values, rope_values], dim=-1)

    @staticmethod
    def _scatter_paged_cache(
        cache: torch.Tensor,
        slots: torch.Tensor,
        values: torch.Tensor,
        block_size: int,
    ) -> None:
        if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
            # A GLM-5 logical cache may occupy only the payload prefix of a
            # larger physical page. Flattening such an as_strided view would
            # either fail or discard the physical page stride, so address the
            # page and its token offset independently.
            values = values.reshape(values.shape[0], *cache.shape[2:])
            valid = (slots >= 0) & (slots < cache.shape[0] * block_size)
            safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
            block_ids = torch.div(
                safe_slots,
                block_size,
                rounding_mode="floor",
            )
            block_offsets = torch.remainder(safe_slots, block_size)
            row_mask = valid.view(-1, *([1] * (values.ndim - 1)))
            row_zero = cache[0, 0].clone()
            safe_values = torch.where(row_mask, values, row_zero.unsqueeze(0))
            row_zero_mask = valid & (slots == 0)
            update_zero = torch.where(
                row_zero_mask.view(-1, *([1] * (values.ndim - 1))),
                values,
                torch.zeros_like(values),
            ).sum(dim=0)
            expected_zero = torch.where(
                row_zero_mask.any(),
                update_zero,
                row_zero,
            )
            cache[block_ids, block_offsets] = safe_values
            cache[0, 0].copy_(expected_zero)
            return
        valid_rows = (
            (slots >= 0) & (slots < cache.shape[0] * block_size)
        ).nonzero().flatten()
        if valid_rows.numel() == 0:
            return
        valid_slots = slots[valid_rows]
        block_ids = torch.div(
            valid_slots,
            block_size,
            rounding_mode="floor",
        )
        block_offsets = torch.remainder(valid_slots, block_size)
        indices = torch.stack([block_ids, block_offsets], dim=-1)
        torch_npu.npu_scatter_nd_update_(
            cache,
            indices,
            values.reshape(values.shape[0], *cache.shape[2:])[valid_rows],
        )

    def indexer_select_pre_process(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        if self.wk_weights_proj is None or self.k_norm is None:
            raise RuntimeError("GLM-5 Indexer KPool MLA indexer K projection is not initialized.")
        kw, _ = self.wk_weights_proj(x)
        k_li = self.k_norm(kw[:, : self.head_dim]).view(
            -1,
            1,
            self.head_dim,
        )
        if self.qk_rope_head_dim > 0:
            k_pe, k_nope = k_li.split(
                [
                    self.qk_rope_head_dim,
                    self.head_dim - self.qk_rope_head_dim,
                ],
                dim=-1,
            )
            k_pe = torch_npu.npu_rotary_mul(
                k_pe.unsqueeze(2),
                cos,
                sin,
                rotary_mode=("jhalf" if self.is_rope_neox_style else "interleave"),
            ).squeeze(2)
            k_li = torch.cat([k_pe, k_nope], dim=-1)

        # The SFA base stores this tensor before post-processing.  Make that
        # write initialize one complete compressor-state row: [K, empty gate].
        # CANN key_pool requires the state cache to be FP32.
        state_k = k_li.to(torch.bfloat16).view(-1, self.head_dim).unsqueeze(1)
        return torch.cat([state_k, torch.zeros_like(state_k)], dim=-1), None

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: AscendIndexerKPoolMLAMetadata,
    ):
        del slots
        kv_c, k_pe = kv_no_split.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c, _ = torch_npu.npu_rms_norm(
            kv_c,
            self.kv_a_layernorm.weight,
            epsilon=self.kv_a_layernorm.variance_epsilon,
        )
        if self.qk_rope_head_dim > 0:
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
        num_tokens = attn_metadata.num_actual_tokens
        kv_slots = self._indexer_kpool_mla_metadata["kv"].slot_mapping[:num_tokens]
        mla_cache = self._indexer_kpool_mla_caches["kv"]
        if isinstance(mla_cache, torch.Tensor):
            packed_kv = self._pack_mla_cache_values(kv_c, k_pe, num_tokens)
            self._scatter_paged_cache(
                mla_cache,
                kv_slots,
                packed_kv,
                attn_metadata.block_size,
            )
        else:
            kv_latent_cache, rope_cache = self._get_mla_cache_views()
            self._scatter_paged_cache(
                kv_latent_cache,
                kv_slots,
                kv_c[:num_tokens].view(-1, self.kv_lora_rank),
                attn_metadata.block_size,
            )
            if self.qk_rope_head_dim > 0:
                self._scatter_paged_cache(
                    rope_cache,
                    kv_slots,
                    k_pe[:num_tokens].reshape(num_tokens, self.qk_rope_head_dim),
                    attn_metadata.block_size,
                )
        return None, None

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.qk_rope_head_dim == 0:
            return x
        return super().rope_single(x, cos, sin)

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return super().forward(
                layer_name,
                hidden_states,
                kv_cache,
                None,
                need_gather_q_kv,
                output,
            )
        cache_roles = (
            "kv",
            "indexer_state",
            "indexer",
        )
        # The forward context may hand a single per-layer metadata object
        # instead of a per-role tuple; normalize by pulling the
        # indexer/state cache entries from the forward context (they are
        # registered under their own cache-layer prefixes).
        if attn_metadata is not None and not isinstance(attn_metadata, (tuple, list)):
            attn_metadata = (attn_metadata,)
        if kv_cache is not None and not isinstance(kv_cache, (tuple, list)):
            kv_cache = (kv_cache,)
        attn_metadata = tuple(attn_metadata) if attn_metadata is not None else ()
        kv_cache = tuple(kv_cache) if kv_cache is not None else ()
        if tuple(item.cache_role for item in attn_metadata) != cache_roles:
            from vllm.forward_context import get_forward_context as _gfc

            _amd = _gfc().attn_metadata
            _by_role = {item.cache_role: item for item in attn_metadata}
            # Draft (MTP) steps hand this layer metadata built by the wrong
            # per-backend builder (role-tagged 'indexer'); the forward
            # context carries the correct per-role entries for all three
            # caches -- including this layer's own 'kv' metadata. Pull all
            # three from the context so the roles resolve regardless of what
            # the caller passed.
            _role_layers = (
                (layer_name,),
                (self.indexer.state_cache.prefix,),
                (self.indexer.k_cache.prefix,),
            )
            for (_layer,) in _role_layers:
                _md = _amd.get(_layer) if isinstance(_amd, dict) else None
                if _md is not None:
                    _by_role[_md.cache_role] = _md
            attn_metadata = tuple(
                _by_role[r] for r in cache_roles if r in _by_role
            )
        if tuple(item.cache_role for item in attn_metadata) != cache_roles:
            raise ValueError(
                "Indexer KPool MLA metadata roles must match cache order "
                f"{cache_roles}, got {tuple(item.cache_role for item in attn_metadata)}."
            )
        if len(kv_cache) != len(cache_roles):
            from vllm_ascend.ops.glm5_kpool_indexer_ascend import bound_cache as _bc

            # The main MLA layer hands its own cache tuple (e.g. latent +
            # rope views); keep only the first entry as the "kv" role and
            # append the state/indexer caches to match the 3-role layout.
            _kv = list(kv_cache[:1]) if kv_cache else []
            for _layer in (self.indexer.state_cache, self.indexer.k_cache):
                with contextlib.suppress(Exception):
                    _kv.append(_bc(_layer))
            kv_cache = tuple(_kv)
        if len(kv_cache) != len(cache_roles):
            raise ValueError(f"Indexer KPool MLA expects {len(cache_roles)} caches, got {len(kv_cache)}.")
        metadata = self._by_role(attn_metadata)
        self._indexer_kpool_mla_metadata = metadata  # graph-stable objects, refreshed per step
        self._indexer_kpool_mla_caches = dict(zip(cache_roles, kv_cache, strict=True))

        indexer_cache = self._indexer_kpool_mla_caches["indexer"]
        if not isinstance(indexer_cache, torch.Tensor) or indexer_cache.dtype != torch.bfloat16:
            raise ValueError("Indexer KPool MLA indexer cache must be one bfloat16 key tensor.")

        kv_latent_cache, rope_cache = self._get_mla_cache_views()

        # AscendSFAImpl writes the combined [K, empty gate] compressor state to
        # kv_cache[2]. The post-process fills gate and emits compressed BF16 K.
        sfa_cache = (
            kv_latent_cache,
            rope_cache,
            self._indexer_kpool_mla_caches["indexer_state"],
        )
        return super().forward(
            layer_name,
            hidden_states,
            sfa_cache,
            metadata["kv"],
            need_gather_q_kv,
            output,
        )

    def _write_indexer_cache(self, *args, **kwargs):
        # DeepSeek SFA writes raw per-token keys into its per-token indexer
        # cache. GLM-5 kpool instead persists per-token [K, gate] compressor
        # state and pool-compressed BF16 K — both written by the kpool
        # indexer's own forward (the orchestrator), which this impl invokes.
        # The parent's raw write would view the 4-slot pool cache as a flat
        # per-token array and corrupt it.
        return None

    def _compose_sfa_kv_cache(self, kv_cache):
        # The kpool forward already composes the exact SFA tuple
        # (kv_latent, rope, compressor_state); the DeepSeek SFA
        # recomposition expects the main MLA pair only and would reject the
        # 3-tuple, so pass it through unchanged.
        return kv_cache

    @staticmethod
    def _sparse_attention_pytorch(
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        packed_kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        scale: float,
        num_actual_tokens: int,
        query_chunk_size: int = INDEXER_KPOOL_MLA_SPARSE_ATTN_QUERY_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Reference PyTorch path for token-granular paged sparse attention.

        This implements the ``compute_cpu`` semantics used by
        ``npu_sparse_flash_attention`` for Indexer KPool MLA's fixed configuration:

        - query layout: TND;
        - KV layout: PA_BSND with one KV head;
        - sparse block size: one token;
        - sparse mode: causal mode 3;
        - attention mode: latent value is the first ``kv_lora_rank`` values.

        The packed cache is gathered before splitting KV-latent and RoPE, so
        the implementation never passes non-contiguous cache views to an NPU
        operator.
        """
        if ql_nope.ndim != 3 or q_pe.ndim != 3:
            raise ValueError(
                "Indexer KPool MLA sparse attention queries must use TND "
                f"layout, got {ql_nope.shape} and {q_pe.shape}."
            )
        if ql_nope.shape[:2] != q_pe.shape[:2]:
            raise ValueError(
                "Indexer KPool MLA latent and RoPE queries must share T/N "
                f"dimensions, got {ql_nope.shape} and {q_pe.shape}."
            )
        if packed_kv_cache.ndim != 4 or packed_kv_cache.shape[2] != 1:
            raise ValueError(
                "Indexer KPool MLA packed KV cache must have shape "
                f"[blocks, block, 1, dim], got {packed_kv_cache.shape}."
            )
        expected_cache_dim = ql_nope.shape[-1] + q_pe.shape[-1]
        if packed_kv_cache.shape[-1] != expected_cache_dim:
            raise ValueError(
                "Indexer KPool MLA packed cache head dim must be "
                f"{expected_cache_dim}, got {packed_kv_cache.shape[-1]}."
            )
        if topk_indices.ndim == 3:
            if topk_indices.shape[1] != 1:
                raise ValueError(
                    "Indexer KPool MLA sparse indices must have one index "
                    f"head, got {topk_indices.shape}."
                )
            topk_indices = topk_indices.squeeze(1)
        elif topk_indices.ndim != 2:
            raise ValueError(f"Indexer KPool MLA sparse indices must be [T,K] or [T,1,K], got {topk_indices.shape}.")
        if block_table.ndim != 2:
            raise ValueError(f"Indexer KPool MLA block table must be 2-D, got {block_table.shape}.")
        if actual_seq_lengths_query.ndim != 1:
            raise ValueError("actual_seq_lengths_query must be cumulative 1-D.")
        if actual_seq_lengths_key.ndim != 1:
            raise ValueError("actual_seq_lengths_key must be 1-D.")
        if actual_seq_lengths_query.shape != actual_seq_lengths_key.shape:
            raise ValueError(
                "Query and key sequence metadata must have the same request "
                f"count, got {actual_seq_lengths_query.shape} and "
                f"{actual_seq_lengths_key.shape}."
            )
        if block_table.shape[0] != actual_seq_lengths_key.shape[0]:
            raise ValueError(
                "Block-table rows must match the request count, got "
                f"{block_table.shape[0]} and "
                f"{actual_seq_lengths_key.shape[0]}."
            )
        if num_actual_tokens < 0 or num_actual_tokens > ql_nope.shape[0]:
            raise ValueError(f"Invalid num_actual_tokens={num_actual_tokens} for {ql_nope.shape[0]} query rows.")
        if topk_indices.shape[0] < num_actual_tokens:
            raise ValueError(
                f"Sparse-index rows must cover all actual queries, got {topk_indices.shape[0]} and {num_actual_tokens}."
            )
        if query_chunk_size <= 0:
            raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}.")

        output = torch.zeros(
            (
                ql_nope.shape[0],
                ql_nope.shape[1],
                ql_nope.shape[2],
            ),
            dtype=ql_nope.dtype,
            device=ql_nope.device,
        )
        if num_actual_tokens == 0:
            return output

        query_ends = actual_seq_lengths_query
        query_starts = torch.cat(
            [
                torch.zeros_like(query_ends[:1]),
                query_ends[:-1],
            ],
        )
        token_ids = torch.arange(
            num_actual_tokens,
            device=ql_nope.device,
            dtype=query_ends.dtype,
        )
        request_ids = torch.bucketize(token_ids, query_ends, right=True)
        block_size = packed_kv_cache.shape[1]
        num_block_columns = block_table.shape[1]
        score_mask_value = torch.finfo(torch.float32).min

        for query_start in range(
            0,
            num_actual_tokens,
            query_chunk_size,
        ):
            query_end = min(
                query_start + query_chunk_size,
                num_actual_tokens,
            )
            chunk_request_ids = request_ids[query_start:query_end]
            chunk_token_ids = token_ids[query_start:query_end]
            chunk_sparse_indices = topk_indices[query_start:query_end].to(torch.int64)

            query_offsets = chunk_token_ids - query_starts[chunk_request_ids]
            query_lens = query_ends[chunk_request_ids] - query_starts[chunk_request_ids]
            key_lens = actual_seq_lengths_key[chunk_request_ids]
            causal_limits = (key_lens - query_lens + query_offsets + 1).to(torch.int64)

            valid = chunk_sparse_indices >= 0
            valid &= chunk_sparse_indices < causal_limits[:, None]
            safe_token_indices = chunk_sparse_indices.clamp_min(0)
            logical_pages = torch.div(
                safe_token_indices,
                block_size,
                rounding_mode="floor",
            )
            valid &= logical_pages < num_block_columns
            safe_logical_pages = logical_pages.clamp(
                max=num_block_columns - 1,
            )
            page_offsets = torch.remainder(
                safe_token_indices,
                block_size,
            )
            physical_blocks = block_table[
                chunk_request_ids[:, None],
                safe_logical_pages,
            ].to(torch.int64)
            valid &= physical_blocks >= 0
            valid &= physical_blocks < packed_kv_cache.shape[0]
            safe_physical_blocks = physical_blocks.clamp(
                min=0,
                max=packed_kv_cache.shape[0] - 1,
            )

            # Index both page dimensions so padded physical-page strides are
            # preserved. Advanced indexing materializes a contiguous [C,K,D]
            # result without requiring the source cache itself to be contiguous.
            gathered_packed_kv = packed_kv_cache[
                safe_physical_blocks,
                page_offsets,
                0,
            ]
            gathered_kv, gathered_rope = gathered_packed_kv.split(
                [
                    ql_nope.shape[-1],
                    q_pe.shape[-1],
                ],
                dim=-1,
            )

            chunk_q_nope = ql_nope[query_start:query_end]
            latent_scores = torch.matmul(
                chunk_q_nope.unsqueeze(-2),
                gathered_kv.transpose(-1, -2).unsqueeze(1),
            ).squeeze(-2)
            if q_pe.shape[-1] == 0:
                scores = latent_scores.float().mul_(scale)
            else:
                chunk_q_rope = q_pe[query_start:query_end]
                rope_scores = torch.matmul(
                    chunk_q_rope.unsqueeze(-2),
                    gathered_rope.transpose(-1, -2).unsqueeze(1),
                ).squeeze(-2)
                scores = (latent_scores + rope_scores).float().mul_(scale)
            scores = scores.masked_fill(
                ~valid[:, None, :],
                score_mask_value,
            )
            probabilities = torch.softmax(scores, dim=-1)
            probabilities = torch.where(
                valid[:, None, :],
                probabilities,
                torch.zeros_like(probabilities),
            ).to(ql_nope.dtype)

            chunk_output = torch.matmul(
                probabilities.unsqueeze(-2),
                gathered_kv.unsqueeze(1),
            ).squeeze(-2)
            output[query_start:query_end] = chunk_output

        return output

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        topk_indices: torch.Tensor,
        attn_metadata: AscendIndexerKPoolMLAMetadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> torch.Tensor:
        """Run Indexer KPool MLA sparse attention without requiring contiguous split caches."""
        if len(kv_cache) < 2:
            raise ValueError("Indexer KPool MLA sparse attention requires latent and RoPE cache views.")
        if kv_cache[0].shape[-1] != self.kv_lora_rank:
            raise ValueError(
                "Indexer KPool MLA latent cache dim must be "
                f"{self.kv_lora_rank}, got {kv_cache[0].shape[-1]}."
            )
        if kv_cache[1].shape[-1] != self.qk_rope_head_dim:
            raise ValueError(
                "Indexer KPool MLA RoPE cache dim must be "
                f"{self.qk_rope_head_dim}, got {kv_cache[1].shape[-1]}."
            )

        packed_kv_cache = self._indexer_kpool_mla_caches.get("kv")
        if not isinstance(packed_kv_cache, torch.Tensor):
            raise ValueError("Indexer KPool MLA sparse attention requires the packed MLA cache.")
        import os as _os

        if _os.environ.get("GLM53_KPOOL_PYTORCH_ATTN", "1") == "1":
            # 910B: vllm_ascend_C lacks the MLA-absorbed sparse kernel
            # npu_sparse_flash_mla, and npu_sparse_flash_attention cannot
            # take a 256-dim query against the 512-dim latent cache (EZ1008
            # tiling failure). Route through the PyTorch path (gather
            # selected tokens + absorbed attention) instead; set
            # GLM53_KPOOL_PYTORCH_ATTN=0 once a fused MLA kernel is available.
            return self._sparse_attention_pytorch(
                ql_nope,
                q_pe,
                packed_kv_cache,
                topk_indices,
                attn_metadata.block_table,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                self.scale,
                ql_nope.shape[0],
            )
        return DeviceOperator.execute_sparse_attention_indexer_kpool_mla(
            self,
            ql_nope,
            q_pe,
            packed_kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table=attn_metadata.block_table,
            sparse_mode=3,
            return_lse=False,
        )

    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendIndexerKPoolMLAMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> torch.Tensor:
        del (
            kv_cache,
            attn_metadata,
            cos,
            sin,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )
        if isinstance(q_c, tuple):
            q_c = q_c[0]
        compressed_metadata = self._indexer_kpool_mla_metadata["indexer"]
        assert compressed_metadata.positions is not None
        return self.indexer(
            x,
            q_c,
            compressed_metadata.positions,
            self.indexer_rotary_emb,
        )
