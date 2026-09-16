#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Ascend implementation of upstream :class:`MMEncoderAttention`.

Eager and ACL-graph capture both use Fused Infer Attention (``npu_fused_infer_attention_score``)
with ``graph_task_group_begin/end`` so replay-time host metadata can be rebound from the update stream,
matching the LLM full-graph pattern in :mod:`vllm_ascend.attention.attention_v1`.
"""

from __future__ import annotations

import threading
import weakref
from collections import Counter
from collections.abc import Sequence

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from vllm.logger import logger
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention  # type: ignore
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.worker.encoder_acl_graph import (
    get_encoder_forward_context,
    get_encoder_graph_params,
    maybe_compute_actual_seq_lengths,
    update_encoder_graph_workspace,
)

from vllm_ascend import envs

MIN_PAD_SIZE: int = 64
MAX_PAD_SIZE: int = 128
SWA_INT_MAX: int = 2147483647
FIA_BLOCK_SIZE: int = 128

_FUSION_STATS_LOG_INTERVAL: int = 1000

# Host-side FIA ``actual_seq_lengths`` per ``cu_seqlens`` tensor. The vision
# encoder passes the same tensor to every attention layer of one forward, so
# one device-to-host conversion (or an entry-point prime) is shared by all
# layers instead of repeating the per-layer ``.cpu()`` sync of the generic
# eager path. Window-attention and full-attention tensors are cached
# independently because they are distinct tensor objects.
#
# torch.Tensor cannot key a WeakKeyDictionary (weakref equality delegates to
# the element-wise tensor ``__eq__``), so the cache is keyed by ``id`` with a
# weakref guard and a GC callback that evicts entries when the tensor dies.
_CU_SEQLENS_HOST_CACHE: dict[int, tuple[weakref.ref, tuple[int, ...]]] = {}
_CACHE_LOCK = threading.RLock()

_FUSION_STATS_LOCK = threading.Lock()
_FUSION_STATS: Counter[str] = Counter()


def _cu_seqlens_cache_evictor(cache_key: int):
    def evict(_ref) -> None:
        try:
            with _CACHE_LOCK:
                _CU_SEQLENS_HOST_CACHE.pop(cache_key, None)
        except Exception:  # pragma: no cover - never fail during GC
            pass

    return evict


def prime_cu_seqlens_host_lengths(cu_seqlens: torch.Tensor, lengths: Sequence[int]) -> None:
    """Pre-populate the shared host-length cache for ``cu_seqlens``.

    The vision-encoder entry point knows the sequence boundaries on the host
    (e.g. from the numpy metadata), so it can prime the cache without the
    device-to-host sync the lazy conversion would pay on the first layer.
    """
    try:
        tensor_ref = weakref.ref(cu_seqlens, _cu_seqlens_cache_evictor(id(cu_seqlens)))
    except TypeError:
        # Non-weakref-able tensor: fall back to lazy conversion.
        return
    with _CACHE_LOCK:
        _CU_SEQLENS_HOST_CACHE[id(cu_seqlens)] = (
            tensor_ref,
            tuple(int(v) for v in lengths),
        )


def peek_cu_seqlens_host_lengths(cu_seqlens: torch.Tensor) -> tuple[int, ...] | None:
    """Return the cached host lengths for ``cu_seqlens`` without converting."""
    with _CACHE_LOCK:
        entry = _CU_SEQLENS_HOST_CACHE.get(id(cu_seqlens))
    if entry is None:
        return None
    tensor_ref, lengths = entry
    if tensor_ref() is not cu_seqlens:
        return None
    return lengths


def reset_vit_fusion_stats() -> Counter[str]:
    """Clear and return the fused vision path counters (for tests/ops)."""
    with _FUSION_STATS_LOCK:
        snapshot = _FUSION_STATS.copy()
        _FUSION_STATS.clear()
    return snapshot


def _record_vit_fusion_stat(reason: str | None) -> None:
    enabled = envs.VLLM_ASCEND_VIT_FUSED_QKV_ROPE_PAD_STATS
    with _FUSION_STATS_LOCK:
        _FUSION_STATS["calls"] += 1
        if reason is None:
            _FUSION_STATS["fused"] += 1
        else:
            _FUSION_STATS[f"fallback:{reason}"] += 1
        total = _FUSION_STATS["calls"]
        if enabled and total % _FUSION_STATS_LOG_INTERVAL == 0:
            snapshot = _FUSION_STATS.copy()
        else:
            snapshot = None
    if snapshot is not None:
        detail = ", ".join(f"{key}={value}" for key, value in sorted(snapshot.items()) if key != "calls")
        logger.info(
            "vit fused QKV/RoPE/pad stats: calls=%d fused=%d (%.1f%%) %s",
            snapshot["calls"],
            snapshot["fused"],
            100.0 * snapshot["fused"] / max(1, snapshot["calls"]),
            detail,
        )


def _get_or_convert_cu_seqlens_host_lengths(
    cu_seqlens: torch.Tensor,
    token_count: int,
) -> tuple[int, ...] | None:
    """Return FIA host lengths for ``cu_seqlens`` or ``None`` if unusable.

    Mirrors the generic eager conversion (``flat[1:]`` without filtering) and
    additionally validates that the boundaries cover ``token_count``. The
    result is cached on the tensor so the remaining layers of the same vision
    forward reuse it without another device-to-host sync.
    """
    cached = peek_cu_seqlens_host_lengths(cu_seqlens)
    if cached is not None:
        return cached

    flat = cu_seqlens.detach().cpu().view(-1).tolist()
    lengths = tuple(flat[1:]) if flat else ()
    if not lengths or lengths[-1] != token_count:
        return None
    previous = 0
    for end in lengths:
        if end < previous:
            return None
        previous = end

    prime_cu_seqlens_host_lengths(cu_seqlens, lengths)
    return lengths


class AscendMMEncoderAttention(MMEncoderAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MMEncoderAttention.
            multimodal_config: configs for multi-modal.
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

        self.enable_pad = self.head_size > MIN_PAD_SIZE and self.head_size < MAX_PAD_SIZE

    @classmethod
    def maybe_recompute_cu_seqlens(
        cls,
        attn_backend: AttentionBackendEnum,
        cu_seqlens: np.ndarray,
        hidden_size: int,
        tp_size: int,
        device: torch.device,
        fp8_padded_hidden_size: int | None = None,
    ) -> torch.Tensor:
        """Build the device ``cu_seqlens`` and prime the shared host cache.

        The vision-encoder entry point calls this once per forward group with
        the host numpy boundaries. Priming the cache here lets every layer of
        the group reuse the FIA host lengths without a per-layer (or even a
        single) device-to-host sync. FlashInfer rescales the boundaries, so it
        is excluded from priming.
        """
        result = super().maybe_recompute_cu_seqlens(
            attn_backend,
            cu_seqlens,
            hidden_size,
            tp_size,
            device,
            fp8_padded_hidden_size=fp8_padded_hidden_size,
        )
        if (
            envs.VLLM_ASCEND_ENABLE_VIT_FUSED_QKV_ROPE_PAD
            and attn_backend != AttentionBackendEnum.FLASHINFER
            and isinstance(result, torch.Tensor)
            and isinstance(cu_seqlens, np.ndarray)
            and len(cu_seqlens) >= 2
        ):
            prime_cu_seqlens_host_lengths(result, cu_seqlens[1:].tolist())
        return result

    def _reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 3D tensors:
        (batch_size * seq_len, num_heads, head_size)
        """
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

    def _maybe_pad_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
        if not self.enable_pad:
            return q, k, v, None
        origin_head_dim = q.shape[-1]
        pad_len = MAX_PAD_SIZE - origin_head_dim
        q = F.pad(q, (0, pad_len), mode="constant", value=0)
        k = F.pad(k, (0, pad_len), mode="constant", value=0)
        v = F.pad(v, (0, pad_len), mode="constant", value=0)
        return q, k, v, origin_head_dim

    def _maybe_compute_cu_seqlens(
        self,
        bsz: int,
        q_len: int,
        cu_seqlens: torch.Tensor | None,
        *,
        is_capturing: bool = False,
    ) -> torch.Tensor:
        # In the eager path, if cu_seqlens is provided by the model we use it; if it is not provided, we create a
        # default one assuming all sequences have the same length. This is used by models such as Hunyuan-OCR, which
        # always pass None as cu_seqlens and rely on the operator to compute it internally.
        # In the capture path, we always create the default cu_seqlens on CPU instead of using the model tensor, to
        # avoid a device-to-host sync (.cpu()).
        if is_capturing or cu_seqlens is None:
            cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device="cpu")
            return cu_seqlens

        return cu_seqlens.cpu()

    @staticmethod
    def _maybe_unpad_output(
        context_layer: torch.Tensor,
        origin_head_dim: int | None,
    ) -> torch.Tensor:
        if origin_head_dim is not None:
            return context_layer[..., :origin_head_dim]
        return context_layer

    @staticmethod
    def _restore_batch_layout(
        context_layer: torch.Tensor,
        *,
        bsz: int,
        q_len: int,
        is_reshaped: bool,
    ) -> torch.Tensor:
        if is_reshaped:
            return einops.rearrange(context_layer, "(b s) h d -> b s h d", b=bsz, s=q_len).contiguous()
        return einops.rearrange(context_layer, "(b s) h d -> b s (h d)", b=bsz, s=q_len).contiguous()

    def _run_vit_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        actual_seq_lengths_q: list[int],
        actual_seq_lengths_kv: list[int],
        *,
        out: torch.Tensor | None = None,
        softmax_lse: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fia_kwargs = dict(
            query=query,
            key=key,
            value=value,
            atten_mask=None,
            block_table=None,
            input_layout="TND",
            block_size=FIA_BLOCK_SIZE,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=0,
            pre_tokens=SWA_INT_MAX,
            next_tokens=SWA_INT_MAX,
        )
        if out is None:
            context_layer, _ = torch_npu.npu_fused_infer_attention_score(**fia_kwargs)
            return context_layer
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**fia_kwargs)
        if softmax_lse is None:
            softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        torch_npu.npu_fused_infer_attention_score.out(
            workspace=workspace,
            out=[out, softmax_lse],
            **fia_kwargs,
        )
        return out

    def _check_fused_qkv_rope_pad_fia(
        self,
        qkv_proj: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor | None,
        rotary_pos_emb_sin: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
    ) -> str | None:
        """Validate the specialized Qwen3.5-VL eager preprocessing contract.

        Returns ``None`` when the fused producer accepts the input, otherwise a
        short fallback reason for the hit-rate statistics.
        """
        if not envs.VLLM_ASCEND_ENABLE_VIT_FUSED_QKV_ROPE_PAD:
            return "env_disabled"
        if not HAS_TRITON:
            return "no_triton"
        # ACL-graph capture owns the preallocated FIA buffers and remains on
        # the established path until this producer has capture-safe buffers.
        if get_encoder_forward_context().capturing:
            return "capturing"
        if self.head_size != 72 or self.num_heads != 8 or self.num_kv_heads != 8:
            return "head_config"
        if cu_seqlens is None or rotary_pos_emb_cos is None or rotary_pos_emb_sin is None:
            return "missing_inputs"
        if qkv_proj.ndim != 3 or qkv_proj.shape[1] != 1:
            return "qkv_layout"
        # Packed multi-sequence inputs (multi-image, multi-frame video and
        # window attention) stay on the fused path: QKV/RoPE/padding act per
        # token and FIA separates sequences through the host lengths derived
        # from cu_seqlens.
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            return "cu_seqlens_layout"
        token_count = qkv_proj.shape[0]
        if qkv_proj.shape[2] != 3 * self.num_heads * self.head_size:
            return "qkv_layout"
        if rotary_pos_emb_cos.shape != (token_count, self.head_size // 2):
            return "rope_shape"
        if rotary_pos_emb_sin.shape != rotary_pos_emb_cos.shape:
            return "rope_shape"
        if qkv_proj.dtype != torch.bfloat16:
            return "dtype"
        if rotary_pos_emb_cos.dtype != qkv_proj.dtype or rotary_pos_emb_sin.dtype != qkv_proj.dtype:
            return "dtype"
        if qkv_proj.device.type != "npu":
            return "device"
        if rotary_pos_emb_cos.device != qkv_proj.device or rotary_pos_emb_sin.device != qkv_proj.device:
            return "device"
        if not (qkv_proj.is_contiguous() and rotary_pos_emb_cos.is_contiguous() and rotary_pos_emb_sin.is_contiguous()):
            return "contiguity"
        return None

    def _can_use_fused_qkv_rope_pad_fia(
        self,
        qkv_proj: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor | None,
        rotary_pos_emb_sin: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
    ) -> bool:
        """Check the specialized Qwen3.5-VL eager preprocessing contract."""
        return (
            self._check_fused_qkv_rope_pad_fia(
                qkv_proj,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
                cu_seqlens,
            )
            is None
        )

    def forward_qkv_rope_pad_fia(
        self,
        qkv_proj: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor | None,
        rotary_pos_emb_sin: torch.Tensor | None,
        *,
        cu_seqlens: torch.Tensor | None,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return fused eager FIA output as ``[1, T, H, 72]``, or ``None``.

        The Triton producer replaces QKV layout extraction, Q/K RoPE, and
        three pad-to-128 operations. Unsupported inputs deliberately return
        ``None`` so that the vLLM model keeps its existing implementation.
        """
        del sequence_lengths  # The existing FIA eager path also consumes cu_seqlens.
        reason = self._check_fused_qkv_rope_pad_fia(qkv_proj, rotary_pos_emb_cos, rotary_pos_emb_sin, cu_seqlens)
        if reason is not None:
            _record_vit_fusion_stat(reason)
            return None

        token_count = qkv_proj.shape[0]
        assert cu_seqlens is not None  # guaranteed by _check_fused_qkv_rope_pad_fia
        if cu_seqlens.numel() == 2:
            # A single packed sequence [0, T] needs no host conversion.
            actual_seq_lengths: Sequence[int] = (token_count,)
        else:
            actual_seq_lengths = _get_or_convert_cu_seqlens_host_lengths(cu_seqlens, token_count)
            if actual_seq_lengths is None:
                _record_vit_fusion_stat("cu_seqlens_values")
                return None

        from vllm_ascend.ops.triton.vision_qkv_rope_pad import vision_qkv_rope_pad

        q_pad, k_pad, v_pad = vision_qkv_rope_pad(qkv_proj[:, 0, :], rotary_pos_emb_cos, rotary_pos_emb_sin)
        context = self._run_vit_fia(q_pad, k_pad, v_pad, list(actual_seq_lengths), list(actual_seq_lengths))
        _record_vit_fusion_stat(None)
        return self._maybe_unpad_output(context, self.head_size).unsqueeze(0)

    def _forward_eager_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        is_reshaped: bool,
        bsz: int,
        q_len: int,
    ) -> torch.Tensor:
        actual_seq_lengths_q, actual_seq_lengths_kv = maybe_compute_actual_seq_lengths(
            self._maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens),
            query.shape[0],
            key.shape[0],
            cudagraph_mm_encoder=False,
        )
        q, k, v, origin_head_dim = self._maybe_pad_qkv(query, key, value)
        context_layer = self._run_vit_fia(q, k, v, actual_seq_lengths_q, actual_seq_lengths_kv)
        context_layer = self._maybe_unpad_output(context_layer, origin_head_dim)
        return self._restore_batch_layout(
            context_layer,
            bsz=bsz,
            q_len=q_len,
            is_reshaped=is_reshaped,
        )

    def _forward_capture_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        is_reshaped: bool,
        bsz: int,
        q_len: int,
    ) -> torch.Tensor:
        context = get_encoder_forward_context()
        token_budget = context.token_budget
        is_capturing = context.capturing
        params = get_encoder_graph_params()
        if token_budget is None or params is None:
            raise RuntimeError("Encoder graph capture state was not initialized (missing token_budget).")

        actual_seq_lengths_q, actual_seq_lengths_kv = maybe_compute_actual_seq_lengths(
            self._maybe_compute_cu_seqlens(bsz, q_len, cu_seqlens, is_capturing=is_capturing),
            query.shape[0],
            key.shape[0],
            cudagraph_mm_encoder=True,
        )
        q, k, v, origin_head_dim = self._maybe_pad_qkv(query, key, value)

        out = torch.empty_like(q)
        softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

        workspace = params.workspaces.get(token_budget)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=q,
                key=k,
                value=v,
                atten_mask=None,
                block_table=None,
                input_layout="TND",
                block_size=FIA_BLOCK_SIZE,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=0,
                scale=self.scale,
                pre_tokens=SWA_INT_MAX,
                next_tokens=SWA_INT_MAX,
            )
            update_encoder_graph_workspace(token_budget, workspace)

        stream = torch_npu.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)

        torch.npu.graph_task_group_begin(stream)
        self._run_vit_fia(
            q,
            k,
            v,
            actual_seq_lengths_q,
            actual_seq_lengths_kv,
            out=out,
            softmax_lse=softmax_lse,
            workspace=workspace,
        )
        handle = torch.npu.graph_task_group_end(stream)

        packed = (
            weak_ref_tensors(q),
            weak_ref_tensors(k),
            weak_ref_tensors(v),
            None,
            None,
            FIA_BLOCK_SIZE,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(out),
            weak_ref_tensors(softmax_lse),
        )
        params.attn_params[token_budget].append(packed)
        params.events[token_budget].append(event)
        params.handles[token_budget].append(handle)

        context_layer = self._maybe_unpad_output(out, origin_head_dim)
        return self._restore_batch_layout(
            context_layer,
            bsz=bsz,
            q_len=q_len,
            is_reshaped=is_reshaped,
        )

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        q, k, v = self._reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)

        if get_encoder_forward_context().capturing:
            return self._forward_capture_fia(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                is_reshaped=is_reshaped,
                bsz=bsz,
                q_len=q_len,
            )

        return self._forward_eager_fia(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            is_reshaped=is_reshaped,
            bsz=bsz,
            q_len=q_len,
        )
