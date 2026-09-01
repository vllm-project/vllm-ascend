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

from typing import cast

import torch
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

from vllm_ascend import envs

from vllm_ascend.ops.triton.qwen4_exp.qsa import (
    qsa_sparse_paged_attention,
    qsa_store_cache_rows,
    qsa_select_paged_tokens as qsa_select_paged_tokens_triton,
)

from .common import qsa_cache
from .common.qsa_cache import QSAForwardMetadata
from .nvidia import indexer_qsa as upstream_indexer
from .nvidia import qsa as upstream_qsa
from .ops import (
    qsa_compress_groups_with_ratio,
    qsa_select_paged_tokens as qsa_select_paged_tokens_reference,
    reshape_and_cache_qsa,
)

QSAKVCache = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def _split_qsa_kv_cache(kv_cache: QSAKVCache, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, tuple):
        if len(kv_cache) != 2:
            raise ValueError("QSA packed cache must contain K and V views")
        return kv_cache
    return kv_cache.transpose(1, 2).split(head_size, dim=-1)


def _qsa_cache_is_bound(kv_cache: QSAKVCache) -> bool:
    if isinstance(kv_cache, tuple):
        return len(kv_cache) == 2 and all(cache.numel() for cache in kv_cache)
    return bool(kv_cache.numel())


def apply_qsa_rope(
    rotary_emb: torch.nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE using the QSA head width rather than the main Q/K width."""
    rotary_dim = int(rotary_emb.rotary_dim)
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    if positions.ndim == 2:
        sections = list(rotary_emb.mrope_section)
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

    def normalize_compressed_keys(
        self,
        pooled: torch.Tensor,
        first_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the reference K normalization and RoPE to compressed keys."""
        compressed_keys = upstream_indexer._gemma_rmsnorm(
            pooled.reshape(-1, self.index_head_dim),
            self.k_layernorm.weight,
            self.k_layernorm.variance_epsilon,
        ).reshape(-1, 1, self.index_head_dim)
        if getattr(self.rotary_emb, "mrope_section", None):
            first_positions = first_positions.transpose(0, 1)
        else:
            first_positions = first_positions[:, 0]
        return apply_qsa_rope(
            self.rotary_emb,
            first_positions,
            compressed_keys,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the portable Ascend path instead of upstream Triton kernels."""
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
        q = self.q_layernorm(q.reshape(-1, self.index_head_dim)).reshape_as(q)
        q = apply_qsa_rope(self.rotary_emb, positions, q)
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
        pooled, first_positions = qsa_compress_groups_with_ratio(
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
        qsa_store_cache_rows(
            self.compressed_key_cache.kv_cache,
            compressed_metadata.slot_mapping,
            normalized,
        )
        qsa_store_cache_rows(raw_key_cache, raw_metadata.slot_mapping, token_k[:num_tokens])
        if rope_position_cache is not None:
            qsa_store_cache_rows(
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
        selector = (
            qsa_select_paged_tokens_reference
            if envs.VLLM_ASCEND_FORCE_QSA_REFERENCE
            else qsa_select_paged_tokens_triton
        )
        return selector(
            query,
            self.compressed_key_cache.kv_cache,
            metadata.block_table,
            metadata.token_to_req,
            metadata.logical_positions,
            metadata.seq_lens,
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
        if isinstance(kv_cache, tuple):
            key_cache, value_cache = _split_qsa_kv_cache(kv_cache, self.head_size)
            qsa_store_cache_rows(key_cache, slot_mapping, key)
            qsa_store_cache_rows(value_cache, slot_mapping, value)
        else:
            reshape_and_cache_qsa(key, value, kv_cache, slot_mapping, self.head_size)

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: QSAKVCache,
        attn_metadata: object,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
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
        logical_indices = layer.topk_indices_buffer[:num_tokens]
        key_cache, value_cache = _split_qsa_kv_cache(kv_cache, self.head_size)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        return qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req[:num_tokens],
            output[:num_tokens],
        )


class AscendQSABackend(upstream_qsa.Qwen4ExpQSAFlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_ASCEND"

    @staticmethod
    def get_impl_cls() -> type[AscendQSAImpl]:
        return AscendQSAImpl


# The upstream owner resolves these names from its module globals during
# construction. Replace only the platform-specific components, leaving model
# structure, cache specifications and weight mapping in upstream vLLM.
upstream_indexer.apply_qsa_rope = apply_qsa_rope
upstream_qsa.QSAIndexer = AscendQSAIndexer
upstream_qsa.Qwen4ExpQSAFlashAttentionImpl = AscendQSAImpl
upstream_qsa.Qwen4ExpQSAFlashAttentionBackend = AscendQSABackend

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
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = metadata[self.layer_name]
        if not _qsa_cache_is_bound(self.kv_cache):
            raise RuntimeError("QSA main K/V cache is not bound")
        num_tokens = main_metadata.num_actual_tokens
        side_metadata = metadata[self.indexer.raw_key_cache.prefix]
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (num_tokens, self.indexer.output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
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
        )


__all__ = ["AscendQSAIndexer", "AscendQwen4ExpQSAAttention"]
