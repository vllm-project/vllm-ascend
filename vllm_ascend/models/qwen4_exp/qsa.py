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
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    get_dtype_size,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.ops.triton.linearnorm import qsa_rmsnorm_mrope as _qsa_rmsnorm_mrope  # noqa: F401
from vllm_ascend.ops.triton.qwen4_exp.qsa import (
    qsa_compress_groups_with_ratio as qsa_compress_groups_with_ratio_fused,
)
from vllm_ascend.ops.triton.qwen4_exp.qsa import (
    qsa_store_cache_rows as qsa_store_cache_rows_fused,
)

from .gbsa import GBSA_PAGE_SIZE, qsa_gbsa_attention
from .lightning_indexer import qsa_select_paged_tokens_lightning

QSAKVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]


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
        self._update_and_compress(
            token_k,
            positions,
            raw_metadata,
            compressed_metadata,
        )

        if self.skip_topk:
            if out is None:
                raise RuntimeError("QSA top-k reuse requires an output buffer")
            return out
        return self._select(query, compressed_metadata, out)

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
        _store_qsa_cache_rows(raw_key_cache, raw_metadata.slot_mapping, token_k[:num_tokens])
        if rope_position_cache is not None:
            _store_qsa_cache_rows(
                rope_position_cache,
                raw_metadata.slot_mapping,
                position_rows,
            )

    def _select(
        self,
        query: torch.Tensor,
        metadata: QSAForwardMetadata,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        logger.info_once("QSA is using the CANN Lightning Indexer v2 backend.")
        return qsa_select_paged_tokens_lightning(
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
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA output quantization is not supported on Ascend")
        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output
        selected_indices = layer.topk_indices_buffer[:num_tokens]
        key_cache, value_cache = kv_cache
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        packed_width = layer.indexer.token_topk // layer.indexer.compress_ratio + layer.indexer.compress_ratio - 1
        compact_indices = selected_indices[:, :packed_width]
        logger.info_once("QSA attention is using the CANN GBSA backend.")
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

        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (num_tokens, self.indexer.output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        if query is None or key is None or value is None:
            raise RuntimeError("QSA preparation did not produce Q/K/V tensors")
        self.impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
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


__all__ = ["AscendQSAIndexer", "AscendQwen4ExpQSAAttention"]
