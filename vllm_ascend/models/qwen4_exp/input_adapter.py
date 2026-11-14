# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp-only preparation moved out of the captured model forward.

The common MRV2 runner intentionally remains unaware of QSA.  This adapter is
owned by :class:`AscendQwen4ExpModelState` and prepares fixed-address inputs for
the three Qwen-specific operators that otherwise run once per layer/replay:
GBSA tiling metadata, LightningIndexerV2 tiling metadata and the fused QSA
MRoPE cache gather.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm_ascend.ops.gbsa import generic_block_sparse_attention_metadata
from vllm_ascend.ops.lightning_indexer_v2 import lightning_indexer_v2_metadata

from .gbsa import GBSA_COMPRESS_RATIO, GBSA_PAGE_SIZE


def _tensor_storage_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    """Identify the exact tensor view that supplies dynamic metadata values."""
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


class _Qwen4ExpPreparedRotaryEmbedding(torch.nn.Module):
    """Consume prepared RoPE rows without replacing upstream QKV projection."""

    def __init__(self, delegate: torch.nn.Module) -> None:
        super().__init__()
        self.delegate = delegate
        self._prepared_cos_sin: torch.Tensor | None = None

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            delegate = super().__getattr__("delegate")
            return getattr(delegate, name)

    def set_prepared_cos_sin(self, cos_sin: torch.Tensor) -> None:
        self._prepared_cos_sin = cos_sin

    def _can_use_prepared(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        offsets: torch.Tensor | None,
    ) -> bool:
        from vllm.triton_utils import HAS_TRITON

        prepared = self._prepared_cos_sin
        if not HAS_TRITON or prepared is None or key is None or offsets is not None:
            return False
        if positions.ndim not in (1, 2):
            return False
        if positions.ndim == 2 and (
            not getattr(self.delegate, "mrope_interleaved", False)
            or getattr(self.delegate, "mrope_section", None) is None
        ):
            return False
        expected_shape = (*positions.shape, self.delegate.cos_sin_cache.shape[-1])
        num_tokens = positions.shape[-1]
        return bool(
            tuple(prepared.shape) == expected_shape
            and prepared.device == query.device
            and prepared.dtype == query.dtype
            and key.device == query.device
            and key.dtype == query.dtype
            and query.shape[0] == num_tokens
            and key.shape[0] == num_tokens
        )

    def _forward_prepared(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepared = self._prepared_cos_sin
        if prepared is None:
            raise RuntimeError("Qwen4Exp prepared RoPE rows are unavailable")
        cos, sin = prepared.chunk(2, dim=-1)
        query_shape = query.shape
        key_shape = key.shape
        if positions.ndim == 2:
            from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope

            query, key = triton_mrope(
                query,
                key,
                cos.contiguous(),
                sin.contiguous(),
                self.delegate.mrope_section,
                self.delegate.head_size,
                self.delegate.rotary_dim,
                self.delegate.mrope_interleaved,
                self.delegate.is_neox_style,
            )
        else:
            from vllm_ascend.ops.triton.rope import rope_forward_triton

            num_tokens = positions.shape[0]
            query, key = rope_forward_triton(
                query.view(num_tokens, -1, self.delegate.head_size),
                key.view(num_tokens, -1, self.delegate.head_size),
                cos=cos,
                sin=sin,
                rope_dim=self.delegate.rotary_dim,
                is_neox_style=self.delegate.is_neox_style,
            )
        return query.reshape(query_shape), key.reshape(key_shape)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._can_use_prepared(positions, query, key, offsets):
            if key is None:
                raise RuntimeError("Qwen4Exp prepared RoPE requires a key tensor")
            return self._forward_prepared(positions, query, key)
        if offsets is None:
            return self.delegate(positions, query, key)
        return self.delegate(positions, query, key, offsets)


class Qwen4ExpPreparedInputAdapter:
    """Prepare model-specific replay inputs without changing public MRV2."""

    def __init__(self, model: torch.nn.Module) -> None:
        self._layers = tuple(
            module for module in model.modules() if module.__class__.__name__ == "AscendQwen4ExpQSAAttention"
        )
        self._rope_buffers: dict[tuple[int, tuple[int, ...], torch.dtype, torch.device], torch.Tensor] = {}
        self._metadata_buffers: dict[tuple[object, ...], torch.Tensor] = {}
        self._prepare_gbsa_enabled = os.getenv("VLLM_ASCEND_QWEN4_PREPARE_GBSA", "1") != "0"
        self._prepare_lightning_enabled = os.getenv("VLLM_ASCEND_QWEN4_PREPARE_LIGHTNING", "1") != "0"
        self._prepare_rope_enabled = os.getenv("VLLM_ASCEND_QWEN4_PREPARE_ROPE", "1") != "0"
        if self._prepare_rope_enabled:
            for layer in self._layers:
                rotary = layer.rotary_emb
                if (
                    not layer.layer_name.startswith("mtp.")
                    and isinstance(rotary, torch.nn.Module)
                    and not isinstance(rotary, _Qwen4ExpPreparedRotaryEmbedding)
                ):
                    layer.rotary_emb = _Qwen4ExpPreparedRotaryEmbedding(rotary)

    def prepare_inputs(self, input_batch: Any) -> None:
        """Gather QSA MRoPE rows into graph-stable buffers before forward."""
        self.prepare_positions(
            input_batch.positions,
            input_batch.num_tokens_after_padding,
        )

    def prepare_positions(self, positions: torch.Tensor, num_tokens: int) -> None:
        """Prepare one- or three-axis MRoPE positions for capture/replay."""
        if not self._prepare_rope_enabled:
            return
        if positions.ndim == 1:
            positions = positions[:num_tokens]
        elif positions.ndim == 2:
            positions = positions[:, :num_tokens]
        else:
            raise ValueError("QSA positions must be one- or two-dimensional")
        prepared_by_cache: dict[tuple[object, ...], torch.Tensor] = {}
        for layer in self._layers:
            if layer.layer_name.startswith("mtp."):
                continue
            cache = layer.rotary_emb.cos_sin_cache
            cache_key = (
                cache.data_ptr(),
                tuple(positions.shape),
                layer.q_norm.weight.dtype,
                layer.q_norm.weight.device,
            )
            prepared = prepared_by_cache.get(cache_key)
            if prepared is not None:
                if isinstance(layer.rotary_emb, _Qwen4ExpPreparedRotaryEmbedding):
                    layer.rotary_emb.set_prepared_cos_sin(prepared)
                else:
                    layer._qsa_prepared_cos_sin = prepared
                continue
            safe_positions = positions.clamp(0, cache.shape[0] - 1).contiguous()
            gathered = cache[safe_positions]
            target_dtype = layer.q_norm.weight.dtype
            if gathered.device != layer.q_norm.weight.device:
                gathered = gathered.to(layer.q_norm.weight.device)
            if gathered.dtype != target_dtype:
                gathered = gathered.to(target_dtype)
            key = (cache.data_ptr(), tuple(gathered.shape), gathered.dtype, gathered.device)
            buffer = self._rope_buffers.get(key)
            if buffer is None:
                buffer = torch.empty_like(gathered)
                self._rope_buffers[key] = buffer
            buffer.copy_(gathered)
            if isinstance(layer.rotary_emb, _Qwen4ExpPreparedRotaryEmbedding):
                layer.rotary_emb.set_prepared_cos_sin(buffer)
            else:
                layer._qsa_prepared_cos_sin = buffer
            prepared_by_cache[cache_key] = buffer

    def prepare_attention_metadata(
        self,
        metadata: dict[str, Any],
        *,
        full_graph: bool,
    ) -> None:
        """Attach QSA schedules to model metadata before graph replay."""
        prepared_this_step: dict[tuple[object, ...], torch.Tensor] = {}
        for layer in self._layers:
            main = metadata.get(layer.layer_name)
            raw = metadata.get(layer.indexer.raw_key_cache.prefix)
            compressed = metadata.get(layer.indexer.compressed_key_cache.prefix)
            if main is None or raw is None or compressed is None:
                continue
            if self._prepare_lightning_enabled:
                self._prepare_lightning(
                    layer,
                    compressed,
                    full_graph=full_graph,
                    prepared_this_step=prepared_this_step,
                )
            # Mixed Prefill/Decode dispatch slices GBSA rows dynamically.  Keep
            # its existing per-span fallback; the hot decode-only path has no
            # host request plan and can safely reuse one prepared schedule.
            if self._prepare_gbsa_enabled and getattr(main, "qsa_request_plan", None) is None:
                self._prepare_gbsa(
                    layer,
                    main,
                    raw,
                    full_graph=full_graph,
                    prepared_this_step=prepared_this_step,
                )

    def _persistent_metadata(
        self,
        signature: tuple[object, ...],
        generated: torch.Tensor,
    ) -> torch.Tensor:
        kind = str(signature[0]) if signature else "operator"
        key = (*signature, generated.dtype, generated.device)
        buffer = self._metadata_buffers.get(key)
        if buffer is None:
            buffer = torch.empty_like(generated)
            self._metadata_buffers[key] = buffer
        elif buffer.shape != generated.shape:
            raise RuntimeError(f"QSA {kind} metadata shape changed for a fixed graph signature")
        buffer.copy_(generated)
        return buffer

    def _prepare_lightning(
        self,
        layer: Any,
        metadata: Any,
        *,
        full_graph: bool,
        prepared_this_step: dict[tuple[object, ...], torch.Tensor],
    ) -> None:
        live_requests = int(getattr(metadata, "qsa_num_live_requests", metadata.seq_lens.shape[0]))
        requests = int(metadata.seq_lens.shape[0]) if full_graph else live_requests
        lengths = metadata.seq_lens[:requests]
        query_start_loc = metadata.query_start_loc[: requests + 1]
        compressed_lengths = lengths.floor_divide(GBSA_COMPRESS_RATIO)
        residual_lengths = lengths.remainder(GBSA_COMPRESS_RATIO)
        common = dict(
            cu_seqlens_q=query_start_loc,
            seqused_k=compressed_lengths,
            cmp_residual_k=residual_lengths,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=GBSA_COMPRESS_RATIO,
        )
        head_dim = int(layer.indexer.index_head_dim)
        block_topk = int(layer.indexer.token_topk // layer.indexer.compress_ratio)
        if int(layer.indexer.compress_ratio) != GBSA_COMPRESS_RATIO:
            raise ValueError("QSA prepared metadata requires compression ratio 4")
        signature = (
            "lightning_v2",
            int(layer.indexer.index_n_heads),
            head_dim,
            block_topk,
            requests,
            int(metadata.token_to_req.shape[0]),
            _tensor_storage_signature(lengths),
            _tensor_storage_signature(query_start_loc),
        )
        prepared = prepared_this_step.get(signature)
        if prepared is not None:
            metadata.qsa_lightning_v2_metadata = prepared
            return
        generated = lightning_indexer_v2_metadata(
            layer.indexer.index_n_heads,
            1,
            head_dim,
            block_topk,
            batch_size=requests,
            **common,
        )
        metadata.qsa_lightning_v2_metadata = self._persistent_metadata(
            signature,
            generated,
        )
        prepared_this_step[signature] = metadata.qsa_lightning_v2_metadata

    def _prepare_gbsa(
        self,
        layer: Any,
        main: Any,
        raw: Any,
        *,
        full_graph: bool,
        prepared_this_step: dict[tuple[object, ...], torch.Tensor],
    ) -> None:
        live_rows = int(main.num_actual_tokens)
        rows = int(raw.token_to_req.shape[0]) if full_graph else live_rows
        requests = int(raw.seq_lens.shape[0])
        token_to_req = raw.token_to_req[:rows].to(torch.int64)
        positions = raw.logical_positions[:rows]
        # Match both Python indexing and the Triton preparation kernel: a
        # negative request index addresses from the end, so graph-padding -1
        # rows use the final request rather than request zero.
        visible = torch.minimum(
            positions.add(1).clamp_min(0),
            raw.seq_lens[token_to_req],
        ).to(torch.int32)
        counts = visible.add(GBSA_COMPRESS_RATIO - 1).floor_divide(GBSA_COMPRESS_RATIO)
        max_groups = int(layer.indexer.token_topk // layer.indexer.compress_ratio)
        counts = counts.clamp_max(max_groups)
        num_kv_heads = int(layer.num_kv_heads)
        signature = (
            "gbsa",
            int(layer.num_heads),
            num_kv_heads,
            int(layer.head_dim),
            max_groups,
            requests,
            rows,
            int(main.block_table.shape[1]),
            _tensor_storage_signature(raw.token_to_req[:rows]),
            _tensor_storage_signature(positions),
            _tensor_storage_signature(raw.seq_lens),
        )
        prepared = prepared_this_step.get(signature)
        if prepared is not None:
            main.qsa_gbsa_metadata = prepared
            return
        sparse_count = counts.unsqueeze(0).expand(num_kv_heads, -1).contiguous()
        # Metadata generation consumes the GBSA geometry and counts.  Selected
        # indices are produced later by LI and passed to the attention operator.
        sparse_idx = torch.zeros(
            (num_kv_heads, rows, max_groups),
            dtype=torch.int32,
            device=positions.device,
        )
        cu_q = torch.arange(rows + 1, dtype=torch.int64, device=positions.device)
        common = dict(
            cu_seqlens_q=cu_q,
            seqused_kv=visible,
            layout_q="TND",
            layout_kv="PA_BBND",
            layout_sparse_pattern=4,
            mask_mode=1,
            quant_mode=0,
            softmax_precision=0,
            residual_block_mode=0,
            is_consistent_topk=False,
        )
        generated = generic_block_sparse_attention_metadata(
            sparse_idx,
            sparse_count,
            int(layer.num_heads),
            num_kv_heads,
            int(layer.head_dim),
            [1, GBSA_COMPRESS_RATIO],
            max_seqlen_q=1,
            max_seqlen_kv=int(main.block_table.shape[1] * GBSA_PAGE_SIZE),
            **common,
        )
        main.qsa_gbsa_metadata = self._persistent_metadata(
            signature,
            generated,
        )
        prepared_this_step[signature] = main.qsa_gbsa_metadata


__all__ = ["Qwen4ExpPreparedInputAdapter"]
