#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""A3/A2 QFA INT8-GQA attention (quant_mode=12).

Two paths share this file:

1. Prefill (``VLLM_ASCEND_QFA_A3_PREFILL=1``, no KV cache): Q/K quantized
   per-token-head and V per-head to INT8, fed to the TND-packed operator.
   Only PrefillNoCache steps qualify; the KV cache stays BF16.

2. Decode (``--kv-cache-dtype int8_qfa``, paged KV cache): the decode
   operator contract (PA_NZ paged K/V, TND Q with NT q_descale,
   per-token K descale stored per page, per-head static V descale).
   One QFA call per step covers decode and prefill alike.
"""

import torch
import torch_npu
from vllm.v1.attention.backend import AttentionType  # type: ignore[import-not-found]

from vllm_ascend import envs
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendC8AttentionBackendImpl,
    AscendMetadata,
)
from vllm_ascend.utils import logger

QFA_INT8_GQA_QUANT_MODE = 12
QFA_INT8_SCALE_LEVEL = 127.0
QFA_MASK_NO_MASK = 0
QFA_MASK_CAUSAL = 3
QFA_ABSMAX_EPS = 1e-12
QFA_INT8_NZ_DIM_FRAG = 32


def _quantize_per_token_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[T, H, D] float -> INT8 payload + FP32 descale [T, H]."""
    descale = x.abs().amax(dim=-1).clamp_min(QFA_ABSMAX_EPS).float() / QFA_INT8_SCALE_LEVEL
    payload = (x.float() / descale.unsqueeze(-1)).round().clamp(-QFA_INT8_SCALE_LEVEL, QFA_INT8_SCALE_LEVEL)
    return payload.to(torch.int8), descale


def _quantize_per_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[T, H, D] float -> INT8 payload + FP32 descale [H] shared over tokens."""
    descale = x.abs().amax(dim=(0, 2)).clamp_min(QFA_ABSMAX_EPS).float() / QFA_INT8_SCALE_LEVEL
    payload = (x.float() / descale.view(1, -1, 1)).round().clamp(-QFA_INT8_SCALE_LEVEL, QFA_INT8_SCALE_LEVEL)
    return payload.to(torch.int8), descale


def qfa_int8_packet_section_sizes(
    num_kv_heads: int, block_size: int, head_dim: int
) -> tuple[int, int, int]:
    """Element counts per paged packet in physical order: K, K_descale, V.

    K/V payloads live in the operator's PA_NZ 5-D shape
    ``(Bn, N, D//32, Bs, 32)`` (INT8, one byte per element); the K descale is
    the measured ``(Bn, N, Bs)`` FP32 section (probe_panz_descale candidate A,
    rel_fro=0.00%). All three sections are strided views over one raw
    allocation per layer.
    """
    if head_dim % QFA_INT8_NZ_DIM_FRAG:
        raise ValueError(
            f"QFA INT8 packet requires head_dim divisible by {QFA_INT8_NZ_DIM_FRAG}, got {head_dim}."
        )
    if block_size <= 0:
        raise ValueError("QFA INT8 packet requires a positive block size.")
    kv_elems = num_kv_heads * block_size * head_dim
    kd_elems = num_kv_heads * block_size
    return kv_elems, kd_elems, kv_elems


def qfa_int8_page_size_bytes(
    num_kv_heads: int, block_size: int, head_dim: int
) -> int:
    """Bytes per paged packet: K + V INT8 payloads plus the FP32 K descale."""
    k_elems, kd_elems, v_elems = qfa_int8_packet_section_sizes(
        num_kv_heads, block_size, head_dim
    )
    # INT8 payload = 1 byte/elem; FP32 descale = 4 bytes/elem.
    return k_elems + v_elems + 4 * kd_elems


def qfa_int8_cache_spec(spec: "FullAttentionSpec") -> "FullAttentionSpec":
    """Budget the K descale into the page so the planner sizes the pool.

    Mirrors mxfp_cache_spec: keep the real D and the scheduler block size,
    raise ``state_content_bytes`` so the packet's extra FP32 section is
    accounted for in page_size_bytes.
    """
    from dataclasses import replace

    real_bytes = spec.head_size * 2  # K + V payload bytes per token per head
    packet_bytes = qfa_int8_page_size_bytes(spec.num_kv_heads, spec.block_size, spec.head_size)
    content_bytes = packet_bytes // (spec.block_size * spec.num_kv_heads)
    return replace(spec, dtype=torch.int8, state_content_bytes=content_bytes)


def qfa_int8_paged_cache_views(
    raw_tensor: torch.Tensor,
    num_kernel_blocks: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    *,
    page_stride_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three strided views over one raw INT8 allocation, PA_NZ contract.

    Returns ``(k_nz, k_descale, v_nz)``:
    - ``k_nz``/``v_nz``: ``(Bn, N, D//32, Bs, 32)`` INT8 (QFA layout_kv PA_NZ)
    - ``k_descale``: ``(Bn, N, Bs)`` FP32 (probe-verified section shape)
    """
    if raw_tensor.ndim != 1 or raw_tensor.dtype != torch.int8 or raw_tensor.stride(0) != 1:
        raise ValueError("QFA INT8 raw cache must be a contiguous 1-D int8 view.")
    if num_kernel_blocks <= 0:
        raise ValueError("QFA INT8 cache requires at least one kernel block.")

    k_elems, kd_elems, v_elems = qfa_int8_packet_section_sizes(
        num_kv_heads, block_size, head_dim
    )
    payload_bytes = k_elems + 4 * kd_elems + v_elems
    pitch = payload_bytes if page_stride_bytes is None else page_stride_bytes
    if pitch < payload_bytes or raw_tensor.numel() < num_kernel_blocks * pitch:
        raise ValueError("QFA INT8 raw cache or physical page pitch is smaller than the packet.")
    # FP32 section alignment: k_elems is a multiple of head_dim (>=32), and the
    # pitch is a sum of 4-byte-multiple sections, so both divide by 4 exactly.
    if k_elems % 4 or pitch % 4:
        raise ValueError("QFA INT8 packet sections must be 4-byte aligned for the FP32 descale view.")
    # The raw cache can be a narrow view into a bigger group-packed storage
    # (vLLM #51718 packs all group layers into one tensor), so the base offset
    # may be nonzero; the FP32 descale view needs it 4-byte aligned.
    base_offset = raw_tensor.storage_offset()
    if base_offset % 4:
        raise ValueError("QFA INT8 raw cache base offset must be 4-byte aligned for the FP32 descale view.")

    def _strided_view(shape, offset_bytes, elem_dtype, elem_size):
        # Per-axis strides in elements: dims 1.. contiguous, dim0 = packet pitch.
        tail = []
        inner = 1
        for dim in reversed(shape[1:]):
            tail.append(inner)
            inner *= dim
        tail.reverse()
        strides = (pitch // elem_size, *tail)
        # ``storage_offset()`` counts 1-byte elements; the dtype views re-
        # express it in ``elem_dtype`` units, and ``as_strided`` takes an
        # absolute offset -- so convert base + section together. Adding the
        # base in byte units to the section in elem units (the previous form)
        # silently worked for offset-zero raw caches (CPU tests) and exploded
        # on the TP-aligned group-packed ones (A3 bring-up).
        storage_offset = (base_offset + offset_bytes) // elem_size
        return torch.as_strided(
            raw_tensor.view(torch.uint8).view(elem_dtype),
            shape,
            strides,
            storage_offset=storage_offset,
        )

    nz_shape = (
        num_kernel_blocks,
        num_kv_heads,
        head_dim // QFA_INT8_NZ_DIM_FRAG,
        block_size,
        QFA_INT8_NZ_DIM_FRAG,
    )
    kd_shape = (num_kernel_blocks, num_kv_heads, block_size)
    return (
        _strided_view(nz_shape, 0, torch.int8, 1),                    # K payload
        _strided_view(kd_shape, k_elems, torch.float32, 4),           # K descale
        _strided_view(nz_shape, k_elems + 4 * kd_elems, torch.int8, 1),  # V payload
    )


def scatter_qfa_pa_nz_payloads(
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    k_nz: torch.Tensor,
    v_nz: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Fused scatter of quantized K/V payloads into the sectioned PA_NZ cache.

    ``index_put_`` on the per-page packet views made torch_npu materialize
    the whole per-layer pool per write (AsStrided + ViewCopy, ~1.14 GB per
    call; 8.47 ms/layer, 58.5% of decode device time -- profile
    2026-11-18). The fused ScatterPaKvCache kernel writes only the
    addressed slots (the BF16 side runs the same operator at ~5.5 us per
    step).

    On-device contract (probe 2026-11-18): the operator computes every
    address from the data pointer plus the shape's inner product -- it
    ignores the cache view's dim0 stride, so a strided cache target is
    silently miswritten (its doc declares non-contiguous inputs
    unsupported). The sectioned cache layout
    (:func:`qfa_int8_sectioned_cache_views`) provides contiguous K/V
    views; this helper then hands the operator the 4-D scenario-1 form
    ``[num_blocks, num_head * head_size // 32, block_size, 32]`` -- a
    zero-copy reshape of the contiguous 5-D PA_NZ view, so the bytes land
    in the layout the QFA read path expects. ``head_size`` bytes per head
    is a multiple of 32 as the scenario requires; negative (PAD) slots
    are outside the documented value range but the service never passes
    them (the ``num_actual_tokens`` slice excludes padding). Field-by-field
    verification against the index_put golden: ``probe_scatter_qfa*.py``.
    """
    if not (k_nz.is_contiguous() and v_nz.is_contiguous()):
        raise RuntimeError(
            "ScatterPaKvCache requires contiguous K/V cache views "
            "(sectioned layout); strided per-page packet views are not "
            "writable by the operator (probe 2026-11-18)."
        )
    num_blocks, num_heads, num_frags, block_size, _ = k_nz.shape
    torch_npu.npu_scatter_pa_kv_cache(
        key=k_int8,
        value=v_int8,
        key_cache=k_nz.reshape(
            num_blocks, num_heads * num_frags, block_size, QFA_INT8_NZ_DIM_FRAG
        ),
        value_cache=v_nz.reshape(
            num_blocks, num_heads * num_frags, block_size, QFA_INT8_NZ_DIM_FRAG
        ),
        slot_mapping=slot_mapping,
        cache_mode="PA_NZ",
    )


def qfa_int8_sectioned_cache_views(
    raw_tensor: torch.Tensor,
    num_kernel_blocks: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three contiguous views over one raw INT8 allocation, sectioned layout.

    The write path's fused ScatterPaKvCache kernel derives every address as
    ``data_ptr + page * inner_product(shape[1:])`` and ignores the cache
    view's dim0 stride, so the per-page packet layout (``[K|kd|V]`` per
    page, dim0 stride = packet pitch) is not writable by the operator
    (probe 2026-11-18: 5D and 4D targets, cache_mode notwithstanding; the
    op doc declares non-contiguous inputs unsupported). This layout keeps
    the single per-layer allocation but places the three sections
    globally, each internally contiguous::

        [ K blocks (Bn,N,Df,Bs,32) INT8 | K descale (Bn,N,Bs) FP32
        | V blocks (Bn,N,Df,Bs,32) INT8 ]

    Consequences:
    - ScatterPaKvCache writes correctly (dim0 stride == inner product).
    - The index_put_ reference path (``VLLM_ASCEND_QFA_SCATTER_WRITE=0``)
      loses its whole-view materialization penalty (that needed a strided
      target), so the A/B comparison stays fair.
    - The QFA read path is unaffected: PA_NZ reading honors the view's
      dim0 stride and contiguous is the canonical paged layout.
    - Per-layer bytes are unchanged from the packet layout (the scheduler
      sees the same ``state_content_bytes``); a padded physical pitch just
      leaves its tail bytes unused.
    """
    if raw_tensor.ndim != 1 or raw_tensor.dtype != torch.int8 or raw_tensor.stride(0) != 1:
        raise ValueError("QFA INT8 raw cache must be a contiguous 1-D int8 view.")
    if num_kernel_blocks <= 0:
        raise ValueError("QFA INT8 cache requires at least one kernel block.")
    k_elems, kd_elems, v_elems = qfa_int8_packet_section_sizes(
        num_kv_heads, block_size, head_dim
    )
    total_elems = num_kernel_blocks * (k_elems + 4 * kd_elems + v_elems)
    if raw_tensor.numel() < total_elems:
        raise ValueError(
            f"QFA INT8 raw cache holds {raw_tensor.numel()} bytes, "
            f"sectioned layout needs {total_elems}."
        )
    # The FP32 descale section reinterprets bytes from the group-packed
    # storage, so the base offset (narrow strip, vLLM #51718) must stay
    # 4-byte aligned -- same requirement as the packet layout.
    if raw_tensor.storage_offset() % 4:
        raise ValueError("QFA INT8 raw cache base offset must be 4-byte aligned for the FP32 descale view.")

    nz_shape = (
        num_kernel_blocks,
        num_kv_heads,
        head_dim // QFA_INT8_NZ_DIM_FRAG,
        block_size,
        QFA_INT8_NZ_DIM_FRAG,
    )
    kd_shape = (num_kernel_blocks, num_kv_heads, block_size)
    k_bytes = num_kernel_blocks * k_elems
    kd_bytes = 4 * num_kernel_blocks * kd_elems
    return (
        raw_tensor[:k_bytes].view(nz_shape),                                            # K payload
        raw_tensor[k_bytes : k_bytes + kd_bytes].view(torch.float32).view(kd_shape),   # K descale
        raw_tensor[k_bytes + kd_bytes : k_bytes + kd_bytes + num_kernel_blocks * v_elems].view(nz_shape),  # V payload
    )


def _build_int8_cu_seqlens(attn_metadata) -> torch.Tensor:
    """Real per-request cumulative query lengths into the operator's INT32 [B+1].

    ``actual_seq_lengths_q`` carries the true batch (no padded tail) and is
    what the FIA prefill paths already consume; prefix the mandatory 0.
    """
    device = attn_metadata.query_start_loc_gpu.device
    return torch.tensor(
        [0, *attn_metadata.actual_seq_lengths_q],
        dtype=torch.int32,
        device=device,
    )


class AscendQFA3PrefillBackendImpl(AscendAttentionBackendImpl):
    """Stock backend impl with the PrefillNoCache branch routed to QFA INT8-GQA."""

    _enable_logged = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.head_size > 256:
            raise ValueError(
                f"A3 QFA INT8-GQA supports head_dim <= 256, got {self.head_size}."
            )
        if not AscendQFA3PrefillBackendImpl._enable_logged:
            logger.info(
                "A3 QFA INT8-GQA prefill attention enabled "
                "(VLLM_ASCEND_QFA_A3_PREFILL=1); decode and chunked/prefix-hit "
                "prefill keep the stock FIA path."
            )
            AscendQFA3PrefillBackendImpl._enable_logged = True

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata,
        output: torch.Tensor,
    ):
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.sliding_window is None
            and self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            return self._forward_qfa_int8_prefill(query, key, value, attn_metadata, output)
        return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)

    def _forward_qfa_int8_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        from cann_ops_transformer.ops import quant_flash_attn

        num_tokens = attn_metadata.num_actual_tokens
        # Q/K: fused per-token-head quantization (1 kernel each).
        q_int8, q_descale = torch_npu.npu_dynamic_quant(query[:num_tokens], dst_type=torch.int8)
        k_int8, k_descale = torch_npu.npu_dynamic_quant(key[:num_tokens], dst_type=torch.int8)
        # V: per-head over all tokens; no fused kernel available (npu_quantize
        # rejects BF16+FP32 scale), keep the manual chain.
        v_int8, v_descale = _quantize_per_head(value[:num_tokens])
        p_scale = torch.tensor([QFA_INT8_SCALE_LEVEL], dtype=torch.float32, device=query.device)

        # PrefillNoCache: this step's K/V is the whole history, so the query
        # prefix sums serve both sides.
        cu_seqlens = _build_int8_cu_seqlens(attn_metadata)
        max_seqlen = attn_metadata.max_query_len or num_tokens

        attn_out, _ = quant_flash_attn(
            q_int8,
            k_int8,
            v_int8,
            q_descale,
            k_descale,
            v_descale,
            QFA_INT8_GQA_QUANT_MODE,
            p_scale=p_scale,
            metadata=None,
            layout_q="TND",
            layout_q_descale="TND",
            layout_kv="TND",
            layout_out="TND",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            mask_mode=QFA_MASK_CAUSAL if attn_metadata.causal else QFA_MASK_NO_MASK,
            win_left=-1,
            win_right=-1,
            softmax_scale=self.scale,
            return_softmax_lse=False,
        )
        # attn_out is [T, Hq, D] and the runner's output buffer shares that layout.
        output[:num_tokens] = attn_out
        return output


class AscendQFA3Int8Impl(AscendAttentionBackendImpl):
    """INT8 KV cache backend computed by the A3 QFA paged operator.

    Activated by ``--kv-cache-dtype int8_qfa`` alone (no recipe gate, the
    mxfp8 precedent). Structure mirrors the C8-MXFP backend with INT8
    semantics:

    - Cache: one raw INT8 allocation per layer, sectioned
      ``[K blocks | K_descale(FP32, (Bn,N,Bs)) | V blocks]`` by
      :func:`qfa_int8_sectioned_cache_views` (each section contiguous so
      the fused scatter can write; per-layer bytes identical to the
      retired per-page packet layout).
    - Write path (reshape_and_cache): Q/K per-token-head dynamic
      quantization (``npu_dynamic_quant``), V static per-head
      (layer.v_cache_scale, the fa_v.scale calibration channel);
      payloads written via the fused PA_NZ scatter kernel
      (``VLLM_ASCEND_QFA_SCATTER_WRITE``, default on) or index_put
      (env 0: A/B reference -- strided index_put_ materializes the whole
      per-layer pool per call, 8.47 ms/layer).
    - Read path: one QFA call per step for the whole batch (decode and
      prefill alike; the causal mask covers decode rows too).

    Precondition violations fail fast (unsupported model shapes must not
    silently detour to a different numerics path).
    """

    _logged = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not AscendQFA3Int8Impl._logged:
            logger.info(
                "A3 QFA INT8 KV cache attention enabled "
                "(--kv-cache-dtype int8_qfa); one QFA call per step for "
                "decode and prefill alike."
            )
            AscendQFA3Int8Impl._logged = True
        # Capture-safe constants: created lazily on first use (the eager
        # warmup run always precedes graph capture) and then reused at a
        # stable address -- replaying a graph must never repeat a
        # host->device copy (aclrtMemcpy is rejected in GLOBAL capture).
        self._p_scale = None
        self._idx_n = None
        self._idx_df = None
        self._ones_v_scale = None

    def _capture_safe_constants(self, device):
        """p_scale / write-path index constants, built once per device."""
        if self._p_scale is None or self._p_scale.device != device:
            self._p_scale = torch.tensor(
                [QFA_INT8_SCALE_LEVEL], dtype=torch.float32, device=device
            )
            self._idx_n = torch.arange(self.num_kv_heads, device=device).view(1, self.num_kv_heads, 1)
            num_frags = self.head_size // QFA_INT8_NZ_DIM_FRAG
            self._idx_df = torch.arange(num_frags, device=device).view(1, 1, num_frags)
        return self._p_scale, self._idx_n, self._idx_df

    # ------------------------------------------------------------------
    # per-step derivations (graph-replay safe; shared across layers via
    # the step cache, c8_mxfp pattern)
    # ------------------------------------------------------------------
    def _step_cache(self, attn_metadata: AscendMetadata) -> dict:
        cache = getattr(attn_metadata, "qfa_int8_step_cache", None)
        if cache is None:
            cache = {}
            attn_metadata.qfa_int8_step_cache = cache
        return cache

    def _step_lengths(self, attn_metadata: AscendMetadata, num_tokens: int):
        """(cu_seqlens_q, seqused_kv) derived once per step on device.

        The clamp/cummax sanitizes the padded tail (stale -1 entries from
        larger earlier steps) the same way the C8-MXFP backend does.
        """
        cache = self._step_cache(attn_metadata)
        lengths = cache.get("lengths")
        if lengths is None:
            lengths = (
                attn_metadata.query_start_loc_gpu.clamp(min=0, max=num_tokens).cummax(dim=0).values.to(torch.int32),
                attn_metadata.seq_lens_gpu.clamp(min=1).to(torch.int32),
            )
            cache["lengths"] = lengths
        return lengths

    def _step_slot_index(self, attn_metadata: AscendMetadata, slot_mapping: torch.Tensor):
        """slot -> (page, pos) index tensors for the packet writes, once."""
        cache = self._step_cache(attn_metadata)
        slot_index = cache.get("slot_index")
        if slot_index is None:
            safe = slot_mapping.to(torch.long).clamp(min=0)
            block_size = self._kernel_block_size()
            slot_index = (safe // block_size, safe % block_size)
            cache["slot_index"] = slot_index
        return slot_index

    def _kernel_block_size(self) -> int:
        if self.key_cache is None:
            raise RuntimeError("A3 QFA INT8 requires bound caches before slot indexing.")
        return self.key_cache.shape[3]  # NZ view (Bn, N, Df, Bs, 32)

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------
    def reshape_and_cache(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize and scatter K/V (+ K descale) into the packet cache."""
        # The stock call site passes ``output`` as the 6th positional argument
        # (``self.reshape_and_cache(q, k, v, kv_cache, meta, output)``), so it
        # lands in ``*args`` here, not ``kwargs``.
        output = args[0] if args else kwargs.get("output", None)
        num_actual_tokens = attn_metadata.num_actual_tokens
        if key is None or value is None or num_actual_tokens == 0:
            return query, key, value, output

        if len(kv_cache) != 3:
            raise RuntimeError(
                "A3 QFA INT8 requires a (k, k_descale, v) packet cache tuple, "
                f"got length {len(kv_cache)}."
            )
        k_nz, k_descale, v_nz = kv_cache

        # Per-token-head dynamic K quantization: payload + FP32 descale [T, N].
        k_int8, k_descale_tok = torch_npu.npu_dynamic_quant(
            key[:num_actual_tokens], dst_type=torch.int8
        )
        # V static per-head scale: the fa_v.scale channel (ones until the
        # calibrated weights land, Step 3).
        v_scale = self._layer_v_scale(attn_metadata)
        v_int8 = torch.clamp(
            torch.round(value[:num_actual_tokens].float() / v_scale.view(1, -1, 1)),
            -QFA_INT8_SCALE_LEVEL,
            QFA_INT8_SCALE_LEVEL,
        ).to(torch.int8)

        slot_mapping = attn_metadata.slot_mapping[:num_actual_tokens]
        pages, pos = self._step_slot_index(attn_metadata, slot_mapping)
        num_tokens = pages.shape[0]
        n_heads = self.num_kv_heads
        _, idx_n, idx_df = self._capture_safe_constants(pages.device)

        if envs.VLLM_ASCEND_QFA_SCATTER_WRITE:
            # Fused AIV scatter (default): one kernel writes both payloads
            # into the packet views, addressing only the target slots.
            scatter_qfa_pa_nz_payloads(k_int8, v_int8, k_nz, v_nz, slot_mapping)
        else:
            # Reference path (A/B measurement / rollback,
            # VLLM_ASCEND_QFA_SCATTER_WRITE=0). A token's K/V payload is
            # the [Df, 32] tail at (page, n, :, pos, :): index the leading
            # four dims with (page, n, df, pos) tensors broadcast to
            # [T, N, Df] and leave the 32-wide fragment axis implicit. The
            # 3-index form leaves (Bs, 32) as the trailing dims and
            # mismatches the [T, N, Df, 32] payload (probe v2 verified
            # index_put on the 4-D BNBD layout, not this 5-D NZ one). The
            # index constants come from the cached capture-safe set
            # (torch.arange inside capture may host-generate). This
            # strided index_put_ materializes the whole per-layer pool per
            # call (AsStrided + ViewCopy, ~1.14 GB, 8.47 ms/layer =
            # 58.5% of decode device time, profile 2026-11-18).
            nz_last = QFA_INT8_NZ_DIM_FRAG
            num_frags = idx_df.numel()  # head_dim // 32, kept in the cached set
            nz_index = (
                pages.view(num_tokens, 1, 1).expand(num_tokens, n_heads, num_frags),
                idx_n.expand(num_tokens, n_heads, num_frags),
                idx_df.expand(num_tokens, n_heads, num_frags),
                pos.view(num_tokens, 1, 1).expand(num_tokens, n_heads, num_frags),
            )
            payload_src = k_int8.view(num_tokens, n_heads, num_frags, nz_last)
            k_nz.index_put_(nz_index, payload_src)
            payload_src_v = v_int8.view(num_tokens, n_heads, num_frags, nz_last)
            v_nz.index_put_(nz_index, payload_src_v)
        # K descale [T, N] -> (Bn, N, Bs): the 3-D write stays on index_put
        # in both modes -- its strided materialization touches only the
        # ~18 MB descale view (~24 us per step, profile 2026-11-18).
        idx_page = pages.view(num_tokens, 1).expand(num_tokens, n_heads)
        idx_n_2d = idx_n.view(1, n_heads).expand(num_tokens, n_heads)
        idx_pos = pos.view(num_tokens, 1).expand(num_tokens, n_heads)
        k_descale.index_put_((idx_page, idx_n_2d, idx_pos), k_descale_tok)

        from vllm_ascend.attention.attention_v1 import notify_kv_cache_written

        notify_kv_cache_written()
        return query, key, value, output

    def _layer_v_scale(self, attn_metadata: AscendMetadata) -> torch.Tensor:
        """Static per-head V scale for the current layer.

        Reads layer.v_cache_scale when present (fa_v.scale channel; per-head
        [Hkv] after Step 3's weights) and falls back to a ones tensor when
        the checkpoint carries no calibrated keys. The fallback tensor is
        cached (capture-safe); with the fa_v weights in place this branch
        never runs.
        """
        layer = getattr(attn_metadata, "qfa_int8_current_layer", None)
        scale = getattr(layer, "v_cache_scale", None) if layer is not None else None
        if scale is not None and scale.numel() == self.num_kv_heads:
            return scale.data.to(torch.float32)
        device = scale.device if scale is not None else torch.npu.current_device()
        if self._ones_v_scale is None or self._ones_v_scale.device != device:
            self._ones_v_scale = torch.ones(self.num_kv_heads, dtype=torch.float32, device=device)
        return self._ones_v_scale

    # ------------------------------------------------------------------
    # read path: one QFA call per step for the whole batch
    # ------------------------------------------------------------------
    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for AscendQFA3Int8Impl"
            )
        if attn_metadata is None:
            return output.fill_(0)
        if self.sliding_window is not None:
            raise NotImplementedError(
                "A3 QFA INT8 does not support sliding-window attention."
            )
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER_DECODER):
            raise NotImplementedError(
                f"A3 QFA INT8 does not support attn_type={self.attn_type}."
            )
        if len(kv_cache) != 3:
            raise RuntimeError(
                "A3 QFA INT8 requires a (k, k_descale, v) packet cache tuple, "
                f"got length {len(kv_cache)}."
            )

        # Expose the layer for the write path's V scale resolution.
        attn_metadata.qfa_int8_current_layer = layer

        num_tokens = attn_metadata.num_actual_tokens
        if num_tokens <= 0:
            return output

        k_nz, k_descale, v_nz = kv_cache
        self.key_cache = k_nz
        self.value_cache = v_nz

        # Direct-write mode (c8_mxfp pattern): quantize and scatter K/V plus
        # the K descale through this impl's own reshape_and_cache. The stock
        # forward cannot be reused here -- past its cache-update hook it runs
        # the BF16 FIA/paged-attention kernels against the int8 packet views.
        # KV-sharing consumer layers reuse another layer's cache; writing
        # their (dummy) K/V would corrupt the shared slots.
        if key is not None and value is not None and self.kv_sharing_target_layer_name is None:
            self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)

        # Attention on the quantized cache.
        from cann_ops_transformer.ops import quant_flash_attn

        q_int8, q_descale = torch_npu.npu_dynamic_quant(
            query[:num_tokens], dst_type=torch.int8
        )
        q_descale_nt = q_descale.t().contiguous()  # NT: (Q_N, Q_T)

        v_scale = self._layer_v_scale(attn_metadata)
        # Cached capture-safe constant: torch.tensor(..., device=npu) is a
        # host->device copy, rejected under aclgraph capture (GLOBAL mode);
        # the eager warmup run that always precedes capture creates it once.
        p_scale, _, _ = self._capture_safe_constants(query.device)

        cu_seqlens_q, seqused_kv = self._step_lengths(attn_metadata, num_tokens)
        max_seqlen_q = attn_metadata.max_query_len or num_tokens
        mask_mode = QFA_MASK_NO_MASK if max_seqlen_q == 1 else QFA_MASK_CAUSAL
        batch_size = cu_seqlens_q.shape[0] - 1

        attn_out, _ = quant_flash_attn(
            q_int8,
            k_nz,
            v_nz,
            q_descale_nt,
            k_descale,
            v_scale,
            QFA_INT8_GQA_QUANT_MODE,
            block_table=self._qfa_block_table(attn_metadata, batch_size),
            cu_seqlens_q=cu_seqlens_q[: batch_size + 1],
            seqused_kv=seqused_kv[:batch_size],
            p_scale=p_scale,
            metadata=None,
            softmax_scale=self.scale,
            mask_mode=mask_mode,
            win_left=-1,
            win_right=-1,
            max_seqlen_q=max_seqlen_q,
            # Contract: an int upper bound on the real max KV length, used
            # for tiling planning ("not smaller than the actual max"). The
            # -1 default produces corrupted numerics on this build (probe
            # 2026-11-17: rel_fro 1588%) and a per-step .item() is a D2H
            # sync that graph capture rejects -- so pin the model-length
            # constant, which the contract admits and capture freezes
            # safely.
            max_seqlen_kv=int(self.vllm_config.model_config.max_model_len),
            layout_q="TND",
            layout_q_descale="NT",
            layout_kv="PA_NZ",
            layout_out="TND",
            return_softmax_lse=False,
        )
        output[:num_tokens] = attn_out.view(num_tokens, self.num_heads, self.head_size)
        return output

    def _qfa_block_table(self, attn_metadata, batch_size: int) -> torch.Tensor:
        """block_table at kernel-block granularity.

        vLLM encodes block_tables/slot_mapping at the KV-cache storage
        granularity ("Setting kv cache block size to 128 for CUSTOM
        backends"), which already equals the QFA kernel block: the write
        path's slot//kernel_block and the block_tables values share the
        same 128 grain (verified on-device 2026-11-17: pages[0] ==
        block_tables[0][0] == 13). Do NOT re-expand by cache_config
        .block_size // kernel_block_size (1664/128 = 13) -- that re-scales
        already-correct block ids (13 -> 169) and points the operator at
        untouched zero pages, making every attention output exactly zero.
        Re-verify this encoding whenever the vLLM base is upgraded.
        """
        return attn_metadata.block_tables[:batch_size].to(torch.int32)
