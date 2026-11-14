# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend QSA owner for Qwen3.8-Flash-Next."""

from dataclasses import replace
from typing import Protocol, TypeAlias, cast

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.models.qwen4_exp.amd import (
    indexer_qsa as upstream_indexer,
)

# Upstream model and QSA import each other; initialize the model first when
# this submodule is imported directly, independently of lazy package exports.
from vllm.models.qwen4_exp.amd import model as _upstream_model  # noqa: F401
from vllm.models.qwen4_exp.amd import qsa as upstream_qsa
from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.common.qsa_cache import (
    QSAForwardMetadata,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    get_dtype_size,
)
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.attention.utils import (
    maybe_save_kv_layer_to_connector,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window
from vllm_ascend.ops.triton.linearnorm import qsa_rmsnorm_mrope as _qsa_rmsnorm_mrope  # noqa: F401
from vllm_ascend.ops.triton.qwen4_exp.gate import qsa_sigmoid_mul_ as _qsa_sigmoid_mul  # noqa: F401
from vllm_ascend.ops.triton.qwen4_exp.qsa import (
    qsa_compress_groups_with_ratio as qsa_compress_groups_with_ratio_fused,
)
from vllm_ascend.ops.triton.qwen4_exp.qsa import (
    qsa_store_cache_rows as qsa_store_cache_rows_fused,
)
from vllm_ascend.ops.triton.qwen4_exp.qsa import (
    qsa_store_key_position_rows as qsa_store_key_position_rows_fused,
)
from vllm_ascend.ops.triton.qwen4_exp.qsa import (
    qsa_store_kv_rows as qsa_store_kv_rows_fused,
)
from vllm_ascend.utils import npu_stream_switch

from .attention_dispatch import build_qsa_request_plan, dispatch_qsa_attention
from .gbsa import GBSA_PAGE_SIZE, qsa_gbsa_attention
from .lightning_indexer import qsa_select_paged_tokens_lightning_v2

QSAKVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]
_QSA_INDEXER_STREAM: torch.npu.Stream | None = None


def _qsa_indexer_stream() -> torch.npu.Stream:
    global _QSA_INDEXER_STREAM
    if _QSA_INDEXER_STREAM is None:
        _QSA_INDEXER_STREAM = torch.npu.Stream(device=torch.npu.current_device())
    return _QSA_INDEXER_STREAM


def _store_qsa_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store rows with the fused kernel without compacting a strided cache."""
    flat_slots = slot_mapping.reshape(-1)
    qsa_store_cache_rows_fused(cache, flat_slots, rows)


def _compress_qsa_groups(
    raw_keys: torch.Tensor,
    raw_positions: torch.Tensor,
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups with the fused QSA compression kernel."""
    return qsa_compress_groups_with_ratio_fused(
        raw_keys,
        raw_positions,
        compressor_state_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        compress_ratio,
        rope_cache,
    )


class QSAForwardMetadataProtocol(Protocol):
    """Metadata fields consumed by the Ascend sparse QSA implementation."""

    num_actual_tokens: int
    block_table: torch.Tensor


def _qsa_cache_is_bound(kv_cache: QSAKVCache) -> bool:
    return isinstance(kv_cache, tuple) and all(cache.numel() for cache in kv_cache)


def apply_qsa_rope(
    rotary_emb: torch.nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE using the QSA head width rather than the main Q/K width."""
    rotary_dim = int(rotary_emb.rotary_dim)
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    sections = list(getattr(rotary_emb, "mrope_section", []))
    # Keep the materialized normalization/RoPE boundary portable on Ascend.
    # The CUDA/ROCm variants use a Triton MRoPE gather here; advanced indexing
    # has identical semantics and is graph-capturable by the NPU backend.
    use_merged_cache = False
    # Compression emits rows for every scheduled token, including rows whose
    # compressed slot is invalid.  Never let such dead-row metadata reach the
    # NPU Index kernel as an out-of-range address.  This materializes only the
    # tiny position index, not any KV-cache tensor.
    safe_positions = positions.clamp(0, cache.shape[0] - 1).contiguous()
    cos_sin = cache[safe_positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    if positions.ndim == 2 and not use_merged_cache:
        if getattr(rotary_emb, "mrope_interleaved", False):
            merged_cos = cos[0].clone()
            merged_sin = sin[0].clone()
            merged_cos[..., 1 : sections[1] * 3 : 3] = cos[1, ..., 1 : sections[1] * 3 : 3]
            merged_cos[..., 2 : sections[2] * 3 : 3] = cos[2, ..., 2 : sections[2] * 3 : 3]
            merged_sin[..., 1 : sections[1] * 3 : 3] = sin[1, ..., 1 : sections[1] * 3 : 3]
            merged_sin[..., 2 : sections[2] * 3 : 3] = sin[2, ..., 2 : sections[2] * 3 : 3]
            cos, sin = merged_cos, merged_sin
        else:
            cos = torch.cat(
                [part[i] for i, part in enumerate(cos.split(sections, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [part[i] for i, part in enumerate(sin.split(sections, dim=-1))],
                dim=-1,
            )

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary = tensor[..., :rotary_dim]
    if getattr(rotary_emb, "is_neox_style", False):
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin),
            dim=-1,
        )
    else:
        even = rotary[..., ::2]
        odd = rotary[..., 1::2]
        rotated = torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin),
            dim=-1,
        ).flatten(-2)
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


class AscendQSAIndexer(upstream_indexer.QSAIndexer):
    """QSA indexer using NPU-compatible cache, RoPE and top-k operators."""

    def _metadata(
        self,
    ) -> tuple[QSAForwardMetadata, QSAForwardMetadata] | None:
        """Read paired QSA metadata without a graph-breaking host sync."""
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            return None
        raw = cast(QSAForwardMetadata, metadata[self.raw_key_cache.prefix])
        compressed = cast(
            QSAForwardMetadata,
            metadata[self.compressed_key_cache.prefix],
        )
        if raw.num_actual_tokens != compressed.num_actual_tokens:
            raise RuntimeError("QSA side-cache metadata token counts disagree")
        # The common Ascend builder creates the paired logical positions
        # together. Avoid torch.equal here because checking an NPU tensor on
        # the host would synchronize the stream during FULL graph capture.
        return raw, compressed

    def _fused_norm_rope_eligible(
        self,
        tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> bool:
        section = getattr(self.rotary_emb, "mrope_section", None)
        return bool(
            tensor.dtype == torch.bfloat16
            and tensor.ndim == 3
            and tensor.shape[-1] == self.index_head_dim == 128
            and self.k_layernorm.weight.dtype == tensor.dtype
            and getattr(self.rotary_emb, "rotary_dim", None) == 64
            and getattr(self.rotary_emb, "is_neox_style", False)
            and getattr(self.rotary_emb, "mrope_interleaved", False)
            and section is not None
            and list(section) == [11, 11, 10]
            and positions.ndim in (1, 2)
            and (positions.ndim == 1 or positions.shape[0] == 3)
        )

    def _fused_norm_rope(
        self,
        tensor: torch.Tensor,
        norm: torch.nn.Module,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        logger.info_once("QSA indexer is using fused RMSNorm + MRoPE.")
        cache = self.rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
        safe_positions = positions.clamp(0, cache.shape[0] - 1).contiguous()
        cos_sin = cache[safe_positions]
        return torch.ops.vllm.triton_qsa_rmsnorm_mrope(
            tensor=tensor,
            weight=norm.weight,
            cos_sin=cos_sin,
            eps=norm.variance_epsilon,
            mrope_section=list(self.rotary_emb.mrope_section),
            rope_dim=self.rotary_emb.rotary_dim,
        )

    def normalize_compressed_keys(
        self,
        pooled: torch.Tensor,
        first_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize and rotate compressed Indexer keys."""
        if getattr(self.rotary_emb, "mrope_section", None):
            first_positions = first_positions.transpose(0, 1)
        else:
            first_positions = first_positions[:, 0]
        pooled = pooled.reshape(-1, 1, self.index_head_dim)
        if not self._fused_norm_rope_eligible(pooled, first_positions):
            raise ValueError("QSA compressed-key fused norm+RoPE ABI is not satisfied")
        return self._fused_norm_rope(
            pooled,
            self.k_layernorm,
            first_positions,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        main_key: torch.Tensor | None = None,
        main_value: torch.Tensor | None = None,
        main_key_cache: torch.Tensor | None = None,
        main_value_cache: torch.Tensor | None = None,
        main_slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the fused Ascend QSA path."""
        metadata = self._metadata()
        if metadata is None:
            # Preserve step-0 indices when later MTP steps reuse the buffer.
            if self.skip_topk and out is not None:
                return out
            result = torch.full(
                (hidden_states.shape[0], self.output_width),
                -1,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            if out is not None:
                out.copy_(result)
                return out
            return result

        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        hidden_states = hidden_states[:num_tokens]
        positions = positions[..., :num_tokens]
        query, token_k = self.project_qk(hidden_states, positions)
        return self.forward_projected(
            query,
            token_k,
            positions,
            out,
            main_key=main_key,
            main_value=main_value,
            main_key_cache=main_key_cache,
            main_value_cache=main_value_cache,
            main_slot_mapping=main_slot_mapping,
            metadata=metadata,
        )

    def forward_projected(
        self,
        query: torch.Tensor,
        token_k: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        main_key: torch.Tensor | None = None,
        main_value: torch.Tensor | None = None,
        main_key_cache: torch.Tensor | None = None,
        main_value_cache: torch.Tensor | None = None,
        main_slot_mapping: torch.Tensor | None = None,
        metadata: tuple[QSAForwardMetadata, QSAForwardMetadata] | None = None,
    ) -> torch.Tensor:
        """Continue the indexer after projection on an auxiliary stream."""
        if metadata is None:
            metadata = self._metadata()
        if metadata is None:
            if self.skip_topk and out is not None:
                return out
            raise RuntimeError("Projected QSA indexer inputs require metadata")

        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        positions = positions[..., :num_tokens]
        self._update_and_compress(
            token_k,
            positions,
            raw_metadata,
            compressed_metadata,
        )

        if self.skip_topk:
            if out is None:
                raise RuntimeError("QSA top-k reuse requires an output buffer")
            if all(
                tensor is not None
                for tensor in (main_key, main_value, main_key_cache, main_value_cache, main_slot_mapping)
            ):
                assert main_key is not None and main_value is not None
                assert main_key_cache is not None and main_value_cache is not None
                assert main_slot_mapping is not None
                qsa_store_kv_rows_fused(
                    main_key_cache,
                    main_value_cache,
                    main_slot_mapping,
                    main_key[:num_tokens],
                    main_value[:num_tokens],
                )
            return out
        return self._select(
            query,
            compressed_metadata,
            out,
            main_key=main_key,
            main_value=main_value,
            main_key_cache=main_key_cache,
            main_value_cache=main_value_cache,
            main_slot_mapping=main_slot_mapping,
        )

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qk, _ = self.index_qk_proj(hidden_states)
        q_raw, token_k = qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        q = q_raw.reshape(-1, self.index_n_heads, self.index_head_dim)
        if not self._fused_norm_rope_eligible(q, positions):
            raise ValueError("QSA query fused norm+RoPE ABI is not satisfied")
        q = self._fused_norm_rope(q, self.q_layernorm, positions)
        return q, token_k.reshape(-1, 1, self.index_head_dim)

    def _update_and_compress(
        self,
        token_k: torch.Tensor,
        positions: torch.Tensor,
        raw_metadata: QSAForwardMetadata,
        compressed_metadata: QSAForwardMetadata,
    ) -> None:
        num_tokens = raw_metadata.num_actual_tokens
        raw_key_cache = self.raw_key_cache.key_cache
        rope_position_cache = self.raw_key_cache.rope_position_cache
        if rope_position_cache is None:
            position_rows = raw_metadata.logical_positions.view(-1, 1, 1).expand(-1, 1, 3)
        else:
            position_rows = qsa_cache.canonical_qsa_rope_positions(positions)[:num_tokens].to(
                device=raw_key_cache.device
            )
        pooled, first_positions = _compress_qsa_groups(
            token_k[:num_tokens],
            position_rows,
            raw_key_cache,
            raw_metadata.block_table,
            raw_metadata.token_to_req,
            raw_metadata.query_start_loc,
            raw_metadata.logical_positions,
            compressed_metadata.slot_mapping,
            self.compress_ratio,
            rope_position_cache,
        )
        normalized = self.normalize_compressed_keys(pooled, first_positions)
        _store_qsa_cache_rows(
            self.compressed_key_cache.kv_cache,
            compressed_metadata.slot_mapping,
            normalized,
        )
        if rope_position_cache is None:
            _store_qsa_cache_rows(raw_key_cache, raw_metadata.slot_mapping, token_k[:num_tokens])
        else:
            qsa_store_key_position_rows_fused(
                raw_key_cache,
                rope_position_cache,
                raw_metadata.slot_mapping,
                token_k[:num_tokens],
                position_rows,
            )

    def _select(
        self,
        query: torch.Tensor,
        metadata: QSAForwardMetadata,
        out: torch.Tensor | None,
        *,
        main_key: torch.Tensor | None = None,
        main_value: torch.Tensor | None = None,
        main_key_cache: torch.Tensor | None = None,
        main_value_cache: torch.Tensor | None = None,
        main_slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logger.info_once("QSA is using native request-batched CANN LightningIndexerV2.")
        return qsa_select_paged_tokens_lightning_v2(
            query,
            self.compressed_key_cache.kv_cache,
            metadata.block_table,
            metadata.token_to_req,
            metadata.logical_positions,
            metadata.seq_lens,
            metadata.query_start_loc,
            self.token_topk,
            self.compress_ratio,
            out,
            # Host counts are prepared with this batch's QSA metadata.
            num_actual_requests=getattr(metadata, "qsa_num_live_requests", None),
            num_actual_tokens=getattr(metadata, "qsa_num_live_tokens", metadata.num_actual_tokens),
            out_initialized=out is not None,
            prepared_compressed_lengths=getattr(metadata, "qsa_compressed_lengths", None),
            prepared_residual_lengths=getattr(metadata, "qsa_residual_lengths", None),
            prepared_score_weights=getattr(metadata, "qsa_score_weights", None),
            main_key=main_key,
            main_value=main_value,
            main_key_cache=main_key_cache,
            main_value_cache=main_value_cache,
            main_slot_mapping=main_slot_mapping,
        )


class AscendQSAImpl:
    """Minimal QSA attention implementation independent of FlashAttention."""

    supports_dcp = False
    supports_pcp = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        del kwargs
        self.num_heads = cast(int, args[0])
        self.head_size = cast(int, args[1])
        self.scale = cast(float, args[2])
        self.num_kv_heads = cast(int, args[3])
        self.kv_cache_dtype = cast(str, args[6])
        self.attn_type = args[8]
        self.alibi_slopes = None
        self.sinks = None
        self.sliding_window = (-1, -1)
        self.supports_quant_query_input = False

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: QSAKVCache,
        slot_mapping: torch.Tensor,
    ) -> None:
        del layer
        key_cache, value_cache = kv_cache
        _store_qsa_cache_rows(key_cache, slot_mapping, key)
        _store_qsa_cache_rows(value_cache, slot_mapping, value)

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: QSAKVCache,
        attn_metadata: QSAForwardMetadataProtocol,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        logical_positions: torch.Tensor,
        seq_lens: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA output quantization is not supported on Ascend")
        num_tokens = attn_metadata.num_actual_tokens
        if num_tokens == 0:
            return output
        selected_indices = layer.topk_indices_buffer[:num_tokens]
        key_cache, value_cache = kv_cache
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        packed_width = layer.indexer.token_topk // layer.indexer.compress_ratio + layer.indexer.compress_ratio - 1
        compact_indices = selected_indices[:, :packed_width]
        request_plan = getattr(attn_metadata, "qsa_request_plan", None)
        if request_plan is None:
            return qsa_gbsa_attention(
                query[:num_tokens],
                key_cache,
                value_cache,
                compact_indices,
                attn_metadata.block_table,
                token_to_req[:num_tokens],
                logical_positions[:num_tokens],
                seq_lens,
                output[:num_tokens],
                prepared_visible_lengths=getattr(attn_metadata, "qsa_gbsa_visible_lengths", None),
                prepared_cu_query=getattr(attn_metadata, "qsa_gbsa_cu_query", None),
                prepared_row_block_table=getattr(attn_metadata, "qsa_gbsa_row_block_table", None),
            )
        return dispatch_qsa_attention(
            query[:num_tokens],
            key[:num_tokens],
            value[:num_tokens],
            key_cache,
            value_cache,
            compact_indices,
            attn_metadata.block_table,
            token_to_req[:num_tokens],
            logical_positions[:num_tokens],
            seq_lens,
            output[:num_tokens],
            request_plan,
        )

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        """QSA replay parameters live in persistent buffers; nothing to rebind."""


class AscendQSAMetadataBuilder(upstream_qsa.Qwen4ExpQSAMetadataBuilder):
    # FULL_DECODE_ONLY captures a fixed GBSA path. Unlike sequence thresholds,
    # decode phase cannot cross into FIA when a request grows during replay.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    prepares_qsa_step_metadata = True

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        phases = common_attn_metadata.is_prefilling
        if phases is None or phases.device.type != "cpu":
            raise ValueError("QSA phase dispatch requires scheduler CPU is_prefilling metadata")
        metadata.qsa_request_plan = None
        if any(phases.tolist()):
            lengths = common_attn_metadata.seq_lens_cpu_upper_bound
            if lengths is None:
                raise ValueError("QSA prefill dispatch requires scheduler CPU KV lengths")
            metadata.qsa_request_plan = build_qsa_request_plan(
                common_attn_metadata.query_start_loc_cpu,
                lengths,
                common_attn_metadata.num_actual_tokens,
                phases,
            )
        return metadata

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata):
        # Capture dummy metadata need not carry live phase flags. Explicitly
        # bypass the prefill planner for the decode-only graph contract.
        metadata = super().build(0, common_attn_metadata, fast_build=True)
        metadata.qsa_request_plan = None
        return metadata


class AscendQSABackend(upstream_qsa.Qwen4ExpQSAFlashAttentionBackend):
    # QSA pages are represented by descriptor-backed views. Consecutive
    # logical blocks may be separated by other cache owners in the common
    # vLLM pool, while each individual page remains dense.
    uses_descriptor_cache_view = True
    uses_qsa_page_major_kv_view = True

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_ASCEND"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
        return [GBSA_PAGE_SIZE]

    @staticmethod
    def get_impl_cls() -> type[AscendQSAImpl]:
        return AscendQSAImpl

    @staticmethod
    def get_builder_cls() -> type[AscendQSAMetadataBuilder]:
        return AscendQSAMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Store main K/V in two page-major slots of one descriptor page."""
        if spec.head_size != spec.head_size_v:
            raise ValueError("QSA page-major K/V slots require symmetric head sizes")
        slot_bytes = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
        if spec.num_head_slots == 2 and spec.state_content_bytes == slot_bytes:
            return spec
        return replace(
            spec,
            num_head_slots=2,
            state_content_bytes=slot_bytes,
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)


# The upstream owner resolves these names from its module globals during
# construction. Replace only the platform-specific components, leaving model
# structure, cache specifications and weight mapping in upstream vLLM.
upstream_indexer.apply_qsa_rope = apply_qsa_rope
upstream_qsa.QSAIndexer = AscendQSAIndexer
upstream_qsa.Qwen4ExpQSAFlashAttentionImpl = AscendQSAImpl
upstream_qsa.Qwen4ExpQSAFlashAttentionBackend = AscendQSABackend


# Side-cache descriptors use the same block-outermost layout as main QSA KV.
# Their physical block stride is honored by the runner, so no compact copy is
# required before LightningIndexer or the Triton QSA selector consumes them.
qsa_cache.QSAStateBackend.uses_descriptor_cache_view = True


# qsa_cache selects its Triton metadata builder at import time. The upstream
# kernel uses CUDA PDL intrinsics, so Ascend must use the equivalent Torch path.
qsa_cache.build_qsa_metadata = qsa_cache._build_qsa_metadata_torch


class AscendQwen4ExpQSAAttention(upstream_qsa.Qwen4ExpQSAAttention):
    """Qwen4Exp QSA owner bound to the Ascend implementation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # The selector overwrites its compact prefix; the remaining ABI columns
        # stay at the sentinel value for the lifetime of this persistent buffer.
        self.topk_indices_buffer.fill_(-1)

    def _num_actual_tokens(self) -> int:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            return 0
        return cast(QSAForwardMetadata, metadata[self.layer_name]).num_actual_tokens

    def _start_indexer_projection(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Launch indexer projection and query norm/RoPE on an auxiliary stream."""
        metadata = self.indexer._metadata()
        if metadata is None:
            return None, None
        raw_metadata, _ = metadata
        num_tokens = raw_metadata.num_actual_tokens
        main_stream = torch.npu.current_stream()
        aux_stream = _qsa_indexer_stream()
        hidden_ready = main_stream.record_event()
        hidden_states.record_stream(aux_stream)
        positions.record_stream(aux_stream)
        with npu_stream_switch(aux_stream):
            aux_stream.wait_event(hidden_ready)
            index_query, token_k = self.indexer.project_qk(
                hidden_states[:num_tokens],
                positions[..., :num_tokens],
            )
        index_query.record_stream(main_stream)
        token_k.record_stream(main_stream)
        return index_query, token_k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not current_platform.opaque_attention_op():
            return self._forward_impl(positions, hidden_states)
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.qwen4_exp_ascend_qsa_forward_with_output(
            hidden_states,
            positions,
            output,
            upstream_qsa._encode_layer_name(self.layer_name),
        )
        return output

    def _forward_impl(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        index_query, index_token_k = self._start_indexer_projection(hidden_states, positions)
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = hidden_states.shape[0]
        num_actual_tokens = self._num_actual_tokens()
        encoded_layer_name = upstream_qsa._encode_layer_name(self.layer_name)
        if self.use_fused_qsa_prepare:
            attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
            gate = torch.empty_like(attn_output)
            if index_query is not None and index_token_k is not None and current_platform.opaque_attention_op():
                torch.ops.vllm.qwen4_exp_ascend_qsa_prepare_with_output(
                    hidden_states,
                    positions,
                    qkv,
                    index_query,
                    index_token_k,
                    attn_output,
                    gate,
                    encoded_layer_name,
                )
            elif index_query is not None and index_token_k is not None:
                qwen4_exp_ascend_qsa_prepare_with_output(
                    hidden_states,
                    positions,
                    qkv,
                    index_query,
                    index_token_k,
                    attn_output,
                    gate,
                    encoded_layer_name,
                )
            elif current_platform.opaque_attention_op():
                torch.ops.vllm.qwen4_exp_qsa_prepare_with_output(
                    hidden_states,
                    positions,
                    qkv,
                    attn_output,
                    gate,
                    encoded_layer_name,
                )
            else:
                upstream_qsa.qwen4_exp_qsa_prepare_with_output(
                    hidden_states,
                    positions,
                    qkv,
                    attn_output,
                    gate,
                    encoded_layer_name,
                )
            flat_output = attn_output.view(num_tokens, -1)
            flat_gate = gate.view(num_tokens, -1)
            torch.ops.vllm.qwen4_exp_ascend_qsa_sigmoid_mul_(flat_output, flat_gate, num_actual_tokens)
            output, _ = self.o_proj(flat_output)
            return output

        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        if index_query is not None and index_token_k is not None and current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_ascend_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                index_query,
                index_token_k,
                attn_output,
                encoded_layer_name,
            )
        elif index_query is not None and index_token_k is not None:
            qwen4_exp_ascend_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                index_query,
                index_token_k,
                attn_output,
                encoded_layer_name,
            )
        elif current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        else:
            upstream_qsa.qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            torch.ops.vllm.qwen4_exp_ascend_qsa_sigmoid_mul_(flat_output, gate, num_actual_tokens)
        else:
            flat_output[num_actual_tokens:].zero_()
        output, _ = self.o_proj(flat_output)
        return output

    def _main_fused_norm_rope_eligible(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> bool:
        section = getattr(self.rotary_emb, "mrope_section", None)
        return bool(
            self.attn_output_gate
            and qkv.dtype == torch.bfloat16
            and qkv.ndim == 2
            and qkv.shape[1] == 2 * self.q_size + 2 * self.kv_size
            and self.num_heads > 0
            and self.num_kv_heads > 0
            and self.head_dim == 256
            and getattr(self.rotary_emb, "rotary_dim", None) == 64
            and getattr(self.rotary_emb, "is_neox_style", False)
            and getattr(self.rotary_emb, "mrope_interleaved", False)
            and section is not None
            and list(section) == [11, 11, 10]
            and positions.ndim in (1, 2)
            and (positions.ndim == 1 or positions.shape[0] == 3)
        )

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # The Ascend fused MRoPE path gathers a temporary rotary-cache tensor.
        # That temporary is not replay-safe in the MTP FULL graph: after the
        # graph reuses its memory, the source QKV remains finite while the
        # fused result can contain NaNs in Q/K. The draft model has only one
        # QSA layer, so keep the target-model fusion and use the graph-safe
        # decomposition for MTP.
        # ForwardContext has no draft-model flag. The layer name is fixed when
        # the target or MTP model is built and remains stable during graph replay.
        if self.layer_name.startswith("mtp."):
            return super()._project_qkv_gate(qkv, positions)
        if not self._main_fused_norm_rope_eligible(qkv, positions):
            raise ValueError("QSA main-attention fused norm+RoPE ABI is not satisfied")

        # Preserve valid scheduler positions while mapping padded profiling rows
        # to a safe cache entry, matching the QSA indexer path above.
        cache = self.rotary_emb.cos_sin_cache
        safe_positions = positions.clamp(0, cache.shape[0] - 1).contiguous()
        cos_sin = cache[safe_positions]
        if cos_sin.device != qkv.device:
            cos_sin = cos_sin.to(qkv.device)
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(qkv.dtype)
        return torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            cos_sin=cos_sin,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            eps=self.q_norm.variance_epsilon,
            mrope_section=list(self.rotary_emb.mrope_section),
            is_interleaved=self.rotary_emb.mrope_interleaved,
            rope_dim=self.rotary_emb.rotary_dim,
            has_gate=True,
            weight_offset=1.0,
        )

    def bind_kv_cache(self, kv_cache: QSAKVCache) -> None:
        self.kv_cache = kv_cache

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        qkv: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        index_query: torch.Tensor | None = None,
        index_token_k: torch.Tensor | None = None,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            if gate is not None:
                gate.zero_()
            return
        main_metadata = metadata[self.layer_name]
        if not _qsa_cache_is_bound(self.kv_cache):
            raise RuntimeError("QSA main K/V cache is not bound")
        num_tokens = main_metadata.num_actual_tokens
        # Restore main KV and indexer groups before either cache is updated.
        wait_for_kv_layer_from_connector(self.layer_name)
        side_metadata = metadata[self.indexer.raw_key_cache.prefix]
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")

        # Newer vLLM revisions call the owner with the packed projection when
        # ``use_fused_qsa_prepare`` is enabled. The upstream indexer consumes
        # that projection itself; Ascend keeps its independently fused main
        # Q/K/V preparation and indexer paths, so materialize the same outputs
        # here before entering the existing cache/update transaction.
        if qkv is not None:
            q, k, v, projected_gate = self._project_qkv_gate(qkv, positions)
            query = q.view(-1, self.num_heads, self.head_dim)
            key = k.view(-1, self.num_kv_heads, self.head_dim)
            value = v.view(-1, self.num_kv_heads, self.head_dim)
            if gate is None or projected_gate is None:
                raise RuntimeError("Fused QSA preparation requires an output gate")
            gate.copy_(projected_gate.view_as(gate))

        if (index_query is None) != (index_token_k is None):
            raise RuntimeError("QSA indexer projection outputs must be provided together")
        if index_query is not None:
            torch.npu.current_stream().wait_stream(_qsa_indexer_stream())

        main_key_cache, main_value_cache = self.kv_cache
        main_key_cache = canonicalize_singleton_dim_strides(main_key_cache)
        main_value_cache = canonicalize_singleton_dim_strides(main_value_cache)
        can_fuse_main_store = bool(
            key is not None
            and value is not None
            and key.shape[1] == 1
            and value.shape[1] == 1
            and main_key_cache.ndim == 4
            and main_value_cache.shape == main_key_cache.shape
            and main_key_cache.shape[2] == 1
        )
        indexer_kwargs = {
            "main_key": key if can_fuse_main_store else None,
            "main_value": value if can_fuse_main_store else None,
            "main_key_cache": main_key_cache if can_fuse_main_store else None,
            "main_value_cache": main_value_cache if can_fuse_main_store else None,
            "main_slot_mapping": main_metadata.slot_mapping if can_fuse_main_store else None,
        }
        if index_query is not None and index_token_k is not None:
            selected = self.indexer.forward_projected(
                index_query,
                index_token_k,
                positions,
                self.topk_indices_buffer[:num_tokens],
                **indexer_kwargs,
            )
        else:
            selected = self.indexer(
                hidden_states,
                positions,
                self.topk_indices_buffer[:num_tokens],
                **indexer_kwargs,
            )
        if selected.shape != (num_tokens, self.indexer.output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        if query is None or key is None or value is None:
            raise RuntimeError("QSA preparation did not produce Q/K/V tensors")
        if gate is None:
            # Non-gated fallback paths still need deterministic graph padding.
            output.zero_()
        if not can_fuse_main_store:
            self.impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        with attention_transfer_window():
            self.impl.forward_qsa(
                self,
                query,
                key,
                value,
                self.kv_cache,
                main_metadata,
                output,
                token_to_req=side_metadata.token_to_req,
                logical_positions=side_metadata.logical_positions,
                seq_lens=side_metadata.seq_lens,
            )
        maybe_save_kv_layer_to_connector(self.layer_name, list(self.kv_cache))


def _ascend_qsa_layer(layer_name: LayerNameType) -> AscendQwen4ExpQSAAttention:
    resolved = upstream_qsa._resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[resolved]
    if not isinstance(layer, AscendQwen4ExpQSAAttention):
        raise TypeError(f"{resolved} is not an Ascend Qwen4Exp QSA owner")
    return layer


@eager_break_during_capture
def qwen4_exp_ascend_qsa_forward_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Execute QSA eagerly so its auxiliary stream survives graph replay."""
    layer = _ascend_qsa_layer(layer_name)
    output.copy_(layer._forward_impl(positions, hidden_states))


def qwen4_exp_ascend_qsa_forward_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return None


def qwen4_exp_ascend_qsa_prepare_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    qkv: torch.Tensor,
    index_query: torch.Tensor,
    index_token_k: torch.Tensor,
    output: torch.Tensor,
    gate: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    _ascend_qsa_layer(layer_name)._run_qsa(
        hidden_states,
        positions,
        None,
        None,
        None,
        output,
        qkv=qkv,
        gate=gate,
        index_query=index_query,
        index_token_k=index_token_k,
    )


def qwen4_exp_ascend_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_token_k: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    _ascend_qsa_layer(layer_name)._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
        index_query=index_query,
        index_token_k=index_token_k,
    )


direct_register_custom_op(
    op_name="qwen4_exp_ascend_qsa_forward_with_output",
    op_func=qwen4_exp_ascend_qsa_forward_with_output,
    mutates_args=["output"],
    fake_impl=qwen4_exp_ascend_qsa_forward_with_output_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="qwen4_exp_ascend_qsa_prepare_with_output",
    op_func=qwen4_exp_ascend_qsa_prepare_with_output,
    mutates_args=["output", "gate"],
)
direct_register_custom_op(
    op_name="qwen4_exp_ascend_qsa_with_output",
    op_func=qwen4_exp_ascend_qsa_with_output,
    mutates_args=["output"],
)


__all__ = ["AscendQSAIndexer", "AscendQwen4ExpQSAAttention"]
