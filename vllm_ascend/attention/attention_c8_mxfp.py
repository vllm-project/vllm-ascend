#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""C8-MXFP (MXFP8) KV cache attention backend (QFA dual-operator interface).

Installed per layer by the ModelSlim C8 quantization method
(quantization/methods/kv_cache/mxfp_c8.py) instead of being resolved through
the backend registry: create_weights assigns ``layer.attn_backend`` and swaps
``layer.impl.__class__``, so importing this module is only required on the
C8 path. Also carries the KV cache layout primitives (scale-cache shapes,
PA_NZ scatter/fill, hybrid buffer split) shared by the backend and the model
runner.
"""

import math
from typing import Any

import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (  # type: ignore
    AttentionCGSupport,
    AttentionLayer,
)

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.utils import enable_dcp, enable_pcp, notify_kv_cache_written
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    record_attention_compute_start,
)
from vllm_ascend.utils import uses_mooncake_connector

# KV cache MXFP8 scale layouts, PA_NZ flavor (QFA layout_kv="PA_NZ"; golden
# reference: quant_flash_attn_golden.py "PA_NZ: fp8=[Bn,N,D//32,Bs,32],
# Kscale=[Bn,N,Bs//16,D//64,16,2], Vscale=[Bn,N,D//16,Bs//64,16,2]"). The K/V
# caches themselves keep the natural [num_blocks, block_size, num_kv_heads,
# head_dim] storage and are viewed as (num_blocks, num_kv_heads,
# head_dim // 32, block_size, 32) at the operator boundary, so allocation,
# hybrid partitioning, PD transfer and prefix-cache CoW are all
# layout-agnostic. The trailing 2 packs the even/odd 32-element MX scale
# groups (the native output format of npu_dynamic_mx_quant).
# K scale token:  [num_tokens, num_kv_heads, head_dim // 64, 2]
# K scale cache:  [num_blocks, num_kv_heads, block_size // 16, head_dim // 64, 16, 2]
# V scale token (axis=0 quant): [cdiv(num_tokens, 64), num_kv_heads, head_dim, 2]
# V scale cache:  [num_blocks, num_kv_heads, head_dim // 16, block_size // 64, 16, 2]
MXFP_KV_SCALE_GROUP_SIZE = 64
MXFP_KV_SCALE_VALUES_PER_GROUP = 2
MXFP_KV_NZ_DIM_FRAG = 32
MXFP_K_SCALE_NZ_TOKEN_FRAG = 16
MXFP_V_SCALE_NZ_DIM_FRAG = 16
# Unified per-block scale bytes: num_kv_heads * block_size * head_dim / MXFP8_GROUP_SIZE (K and V).
MXFP8_GROUP_SIZE = 32
# E8M0 scale elements are always 1 byte in KV cache budgeting.
MXFP_SCALE_DTYPE_SIZE = 1


def validate_mxfp_k_scale_head_dim(head_dim: int) -> None:
    if head_dim % MXFP_KV_SCALE_GROUP_SIZE != 0:
        raise ValueError(
            f"C8_MXFP K scale cache requires head_dim divisible by {MXFP_KV_SCALE_GROUP_SIZE}, got {head_dim}."
        )


def validate_mxfp_v_scale_block_size(block_size: int) -> None:
    if block_size % MXFP_KV_SCALE_GROUP_SIZE != 0:
        raise ValueError(
            f"C8_MXFP V scale cache requires block_size divisible by {MXFP_KV_SCALE_GROUP_SIZE}, got {block_size}."
        )


def mxfp_kv_scale_groups(head_dim: int) -> int:
    validate_mxfp_k_scale_head_dim(head_dim)
    return head_dim // MXFP_KV_SCALE_GROUP_SIZE


def mxfp_kv_block_scale_groups(block_size: int) -> int:
    validate_mxfp_v_scale_block_size(block_size)
    return block_size // MXFP_KV_SCALE_GROUP_SIZE


def mxfp_k_scale_page_bytes(num_kv_heads: int, block_size: int, head_dim: int) -> int:
    """Bytes per block for k_scale cache."""
    validate_mxfp_k_scale_head_dim(head_dim)
    return num_kv_heads * block_size * head_dim // MXFP8_GROUP_SIZE


def mxfp_v_scale_page_bytes(num_kv_heads: int, block_size: int, head_dim: int) -> int:
    """Bytes per block for v_scale cache."""
    validate_mxfp_v_scale_block_size(block_size)
    return num_kv_heads * block_size * head_dim // MXFP8_GROUP_SIZE


def mxfp_k_scale_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int, int, int, int]:
    return (
        num_blocks,
        num_kv_heads,
        block_size // MXFP_K_SCALE_NZ_TOKEN_FRAG,
        mxfp_kv_scale_groups(head_dim),
        MXFP_K_SCALE_NZ_TOKEN_FRAG,
        MXFP_KV_SCALE_VALUES_PER_GROUP,
    )


def mxfp_v_scale_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int, int, int, int]:
    return (
        num_blocks,
        num_kv_heads,
        head_dim // MXFP_V_SCALE_NZ_DIM_FRAG,
        mxfp_kv_block_scale_groups(block_size),
        MXFP_V_SCALE_NZ_DIM_FRAG,
        MXFP_KV_SCALE_VALUES_PER_GROUP,
    )


def mxfp_kv_page_size_bytes(
    block_size: int,
    num_kv_heads: int,
    k_dim: int,
    v_dim: int,
    kv_dtype_size: int,
) -> int:
    """Bytes per KV cache page for C8_MXFP (FP8 K/V tensors + E8M0 scale caches)."""
    kv_bytes = block_size * num_kv_heads * (k_dim + v_dim) * kv_dtype_size
    scale_bytes = (
        mxfp_k_scale_page_bytes(num_kv_heads, block_size, k_dim)
        + mxfp_v_scale_page_bytes(num_kv_heads, block_size, v_dim)
    ) * MXFP_SCALE_DTYPE_SIZE
    return kv_bytes + scale_bytes


def scatter_mxfp_pa_nz_kv_cache(
    quant_key: torch.Tensor,
    quant_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter quantized K/V into the paged caches in QFA's PA_NZ layout.

    Scenario 1 of the ScatterPaKvCache contract (the operator's native PA_NZ
    mode)::

        key/value   [batch * seq_len, num_head, head_size]
        key/valueCache
                    [num_blocks, num_head, head_size // last_dim, block_size, last_dim]
        slotMapping [batch * seq_len]
        cacheMode   "PA_NZ"
        last_dim = 32 / sizeof(dtype)     -> 32 for any 1-byte dtype
        (head_size * sizeof(dtype)) % 32 == 0

    ``cache_mode`` selects that contract; at the default the operator reads
    scenario 2 ("Norm") and rejects the 5-D view. Both sides go through int8
    views: the operator accepts FLOAT8_E4M3FN only on 950PR/950DT, while
    INT8 is accepted everywhere and writes identical bytes. Negative slots
    (PAD_SLOT_ID) are skipped by the operator itself, keeping the shapes
    static for graph capture.
    """
    if slot_mapping.numel() == 0:
        return

    num_kv_heads, head_dim = quant_key.shape[1], quant_key.shape[2]

    def _as_bytes(t: torch.Tensor) -> torch.Tensor:
        return t if t.dtype == torch.int8 else t.view(torch.int8)

    def _nz_view(cache: torch.Tensor) -> torch.Tensor:
        return _as_bytes(cache).view(
            -1,
            num_kv_heads,
            head_dim // MXFP_KV_NZ_DIM_FRAG,
            block_size,
            MXFP_KV_NZ_DIM_FRAG,
        )

    torch_npu.npu_scatter_pa_kv_cache(
        key=_as_bytes(quant_key),
        value=_as_bytes(quant_value),
        key_cache=_nz_view(key_cache),
        value_cache=_nz_view(value_cache),
        slot_mapping=slot_mapping,
        cache_mode="PA_NZ",
    )


def fill_mxfp_v_scale_cache(value_scale: torch.Tensor, value_scale_cache: torch.Tensor) -> None:
    """Broadcast V's static per-channel E8M0 scale over its whole paged cache.

    ``value_scale`` is the checkpoint's flat ``(num_kv_heads * head_dim,)``
    E8M0 byte vector, ``value_scale_cache`` the PA_NZ 6-D cache
    ``[num_blocks, num_kv_heads, head_dim // 16, block_size // 64, 16, 2]``.
    Reshaping the source to ``(num_kv_heads, head_dim // 16, 1, 16, 1)`` lines
    its channel axis up with the cache's fragment split and lets the block,
    token-group and even/odd axes broadcast. Head count and V head dim are
    read from the cache so a model whose V head dim differs from Q/K's stays
    correct. A one-time fill (run once the caches exist, before any request,
    capture or replay) rather than a per-step scatter.
    """
    num_kv_heads = value_scale_cache.shape[1]
    v_dim_frags = value_scale_cache.shape[2]
    v_dim_frag_size = value_scale_cache.shape[4]
    value_scale_cache.copy_(value_scale.view(num_kv_heads, v_dim_frags, 1, v_dim_frag_size, 1))


def mxfp_k_scale_slot_index(
    slot_mapping: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose slots into the index tensors the K-scale scatter indexes with.

    Returns ``(block_ids, seg_ids, frag_ids)`` for the PA_NZ K-scale cache,
    where a token at in-block offset ``o`` lands at
    ``[block, n, o // 16, :, o % 16, :]``. Derived once per step and shared
    across layers (capture-safe for the same reason as the other per-step
    quantities; see ``_qfa_k_scale_slot_index``).

    Padded rows (slot -1) are clamped to slot 0, keeping the shapes static
    for graph capture; slot 0 is the null block's, see
    scatter_mxfp_k_scale_cache.
    """
    safe_slots = slot_mapping.to(torch.long).clamp(min=0)
    block_ids = safe_slots // block_size
    offsets = safe_slots % block_size
    return block_ids, offsets // MXFP_K_SCALE_NZ_TOKEN_FRAG, offsets % MXFP_K_SCALE_NZ_TOKEN_FRAG


def scatter_mxfp_k_scale_cache(
    key_scale: torch.Tensor,
    key_scale_cache: torch.Tensor,
    slot_index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Scatter per-token K scales into the paged K-scale cache.

    ``key_scale`` shape: ``[num_tokens, num_kv_heads, head_dim // 64, 2]``
    (any 1-byte dtype; callers pass a uint8 view of the E8M0 scale).
    ``key_scale_cache`` shape (PA_NZ): ``[num_blocks, num_kv_heads,
    block_size // 16, head_dim // 64, 16, 2]``.

    ``slot_index`` comes from :func:`mxfp_k_scale_slot_index` and is shared
    across the step's layers; only the write below is per-layer. Capture
    safe: no host-device sync and no data-dependent shapes.

    Padded rows arrive clamped to slot 0 and are simply written there: slot 0
    belongs to block 0, the null block BlockPool never hands to a request, so
    only dummy padding requests (whose output is discarded) read it back.
    """
    block_ids, seg_ids, frag_ids = slot_index
    if block_ids.numel() == 0:
        return
    key_scale_cache[block_ids, :, seg_ids, :, frag_ids, :] = key_scale


def split_hybrid_c8_mxfp_cache_buffer(
    raw_tensor: torch.Tensor,
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a hybrid Mamba/attention buffer into C8 MXFP payloads.

    Hybrid cache groups allocate one padded raw buffer that is shared by
    Mamba and full-attention layers. Mamba state views start at the front
    of that buffer, while attention payloads are placed at the end. C8
    MXFP needs four payloads instead of the regular K/V pair.
    """
    num_blocks, block_size, num_kv_heads, k_dim = k_shape
    if len(v_shape) != 4:
        raise ValueError(f"Expected a four-dimensional V cache shape, got {v_shape}.")
    if v_shape[:3] != k_shape[:3]:
        raise ValueError(
            "C8_MXFP hybrid K/V cache shapes must share block and head dimensions, "
            f"got k_shape={k_shape}, v_shape={v_shape}."
        )

    v_dim = v_shape[3]
    k_scale_shape = mxfp_k_scale_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        k_dim,
    )
    v_scale_shape = mxfp_v_scale_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        v_dim,
    )
    payload_sizes = [
        math.prod(k_shape),
        math.prod(v_shape),
        math.prod(k_scale_shape),
        math.prod(v_scale_shape),
    ]
    payload_size = sum(payload_sizes)
    if raw_tensor.numel() < payload_size:
        raise ValueError(
            "C8_MXFP hybrid cache buffer is too small: "
            f"raw_numel={raw_tensor.numel()}, payload_numel={payload_size}, "
            f"k_shape={k_shape}, v_shape={v_shape}."
        )

    payload = raw_tensor[raw_tensor.numel() - payload_size :]
    raw_k, raw_v, raw_k_scale, raw_v_scale = torch.split(payload, payload_sizes)
    return raw_k, raw_v, raw_k_scale, raw_v_scale


class AscendC8MXFPAttentionBackend(AscendAttentionBackend):
    """Backend for C8-MXFP KV cache layers (QFA dual-operator interface).

    C8-MXFP QFA requires 512-token pages (the D=256 requirement doc allows
    512/1024). This must not be advertised by the generic backend because
    hybrid BF16 models (e.g. Qwen3.5/3.6 linear-attention mixes) use its
    128-token logical block layout when reshaping their KV cache.
    """

    @staticmethod
    def get_impl_cls() -> type["AscendC8MXFPAttentionBackendImpl"]:
        if enable_pcp() or enable_dcp():
            raise NotImplementedError("C8_MXFP attention does not support PCP/DCP yet.")
        return AscendC8MXFPAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendC8MXFPMetadataBuilder"]:
        if enable_pcp() or enable_dcp():
            raise NotImplementedError("C8_MXFP attention does not support PCP/DCP yet.")
        return AscendC8MXFPMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [512]


class AscendC8MXFPMetadataBuilder(AscendAttentionMetadataBuilder):
    """Metadata builder for the C8-MXFP (QFA) backend.

    The generic Ascend builder sizes its block-table width with the
    128-token generic block; this backend uses 512-token kernel blocks.

    Cudagraph support (final form, no .out wrapper variant needed):
    - PIECEWISE: QFA executes outside the compiled region as a plain call.
    - FULL / FULL_DECODE_ONLY: npugraph_ex captures the allocating QFA
      wrapper plus the in-graph metadata op natively (golden-test
      GRAPH_PATH=7 methodology). Replay correctness relies on the model
      runner's persistent length buffers (query_start_loc_gpu /
      seq_lens_gpu), which the impl derives QFA's cu_seqlens/seqused from
      through captured device-side ops, so every replay re-reads the
      current step's lengths.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        mode = vllm_config.compilation_config.cudagraph_mode
        if mode.has_piecewise_cudagraphs() and not mode.has_full_cudagraphs():
            # PIECEWISE: QFA runs outside the compiled region as a plain
            # call (validated on-device). UNIFORM_BATCH is the support
            # level the other piecewise-capable Ascend backends report
            # (e.g. mla_v1); AttentionCGSupport has no PARTIAL member.
            return AttentionCGSupport.UNIFORM_BATCH
        # FULL (incl. FULL_DECODE_ONLY): npugraph_ex captures the
        # allocating QFA wrapper natively (golden-test GRAPH_PATH=7).
        # Spec decode (MTP) included: the draft graph is a plain
        # torch.npu.graph ACLGraphWrapper capture whose metadata goes
        # through the same builder.build() (field passthrough) and whose
        # per-step lengths live in the proposer's persistent buffers,
        # refreshed in place before each draft replay -- the same
        # stable-address contract as the main-model graphs. During draft
        # capture _EXTRA_CTX.capturing is set by the ACLGraphWrapper
        # (shared forward-context object), so the QFA metadata op is
        # still executed inline inside the captured region.
        return AttentionCGSupport.ALWAYS


QFA_QUANT_MODE_MXFP8 = 1
QFA_MASK_MODE_NO_MASK = 0
QFA_MASK_MODE_CAUSAL = 3
QFA_LAYOUT_TND = "TND"
QFA_LAYOUT_N2TGD = "N2TGD"
QFA_LAYOUT_PA_NZ = "PA_NZ"
# MXFP8 takes the q scale in one of two layouts, and the choice selects the
# kernel: TND (Q_T, Q_N, D/64, 2) compiles the prefill template, N2TGD
# (KV_N, Q_T, G, D/64, 2) the decode one, which merges the S1 and G axes.
# The operator doc puts the boundary at G*Q_S, recommending N2TGD at or below
# this value. Both layouts hold the same scales, so the split is a performance
# choice, not a correctness one.
QFA_QSCALE_N2TGD_MAX_G_TIMES_QS = 80


def _build_qfa_cu_seqlens(cumulative_seq_lengths: list[int], device: torch.device) -> torch.Tensor:
    """Build QFA ``cu_seqlens``: int32 (B+1,) with a leading 0.

    FIA-style ``actual_seq_lengths`` are B cumulative entries; QFA expects the
    cumulative sums prefixed with 0 so batch i spans
    ``[cu_seqlens[i], cu_seqlens[i + 1])`` (QFA requirement doc, sequence
    length conversion rules).
    """
    return torch.tensor([0, *cumulative_seq_lengths], dtype=torch.int32, device=device)


# Resolved lazily and cached: (main_op, metadata_op), delivered through the
# cann_ops_transformer package shipped with the CANN toolkit. Captured
# directly by npugraph_ex (internal at::empty allocations land in the graph's
# private pool); no task_group/update machinery is needed on our side.
_QFA_OPS: tuple[Any, Any] | None = None


def _get_qfa_ops() -> tuple[Any, Any]:
    global _QFA_OPS
    if _QFA_OPS is None:
        try:
            from cann_ops_transformer.ops import quant_flash_attn as main_op  # type: ignore[import-not-found]
            from cann_ops_transformer.ops import (  # type: ignore[import-not-found]
                quant_flash_attn_metadata as metadata_op,
            )
        except ImportError:
            raise RuntimeError(
                "C8_MXFP requires the QFA dual operators delivered in the "
                "cann_ops_transformer package (shipped with the CANN toolkit): "
                "cann_ops_transformer.ops.quant_flash_attn(_metadata) could not "
                "be imported in this environment."
            ) from None
        _QFA_OPS = (main_op, metadata_op)
    return _QFA_OPS


class AscendC8MXFPAttentionBackendImpl(AscendAttentionBackendImpl):
    """MXFP8 KV cache backend computed by the QFA dual-operator interface.

    forward() quantizes Q/K dynamically (``npu_dynamic_mx_quant``, FP8 E4M3 +
    per-token-group E8M0 scales) and V statically (the checkpoint's
    per-channel E8M0 scale), scatters the quantized K/V plus their scale
    caches into the paged cache, and calls the QFA metadata + main operators
    directly on the paged cache.

    Layout: PA_NZ. The K/V caches keep the natural
    ``[num_blocks, block_size, num_kv_heads, head_dim]`` storage; the scatter
    and QFA both work through the NZ 5-D view
    ``(num_blocks, num_kv_heads, head_dim//32, block_size, 32)``. The E8M0
    scale caches are allocated directly in the PA_NZ 6-D shapes.

    One QFA call per step for the whole batch (decode and prefill alike); a
    step whose every query is one row long drops the mask (NO_MASK), which is
    equivalent there and picks a cheaper kernel.

    Graph capture: handled natively by npugraph_ex (golden-test GRAPH_PATH=7).
    Replay safety relies on every per-step input being a stable-address
    tensor refreshed outside Python: block_table / slot_mapping come from the
    runner's persistent buffers, and cu_seqlens_q / seqused_kv are derived
    in-graph from the runner's persistent query_start_loc / seq_lens buffers.
    Speculative decoding (MTP) uses the same derivation chain.

    NOTE: the QFA wrapper's signature (verified on-device) differs from the
    requirement doc's torch_npu example: positional
    q_descale/k_descale/v_descale/quant_mode, p_scale instead of
    quant_scale_p, an extra layout_q_descale, no pa_block_size, and a
    required v_descale placeholder on the metadata call for quant_mode=1.
    """

    # Installed via ``layer.impl.__class__`` assignment, which does not call
    # this subclass's constructor. Class-level defaults are therefore
    # required for objects that predate the class swap.
    enable_hamming_sparse: bool = False

    @property
    def main_op(self) -> Any:
        """The QFA main operator, resolved through the module-level cache.

        A property rather than an __init__ attribute: this impl is installed
        by ``layer.impl.__class__`` assignment, which never calls the
        subclass constructor.
        """
        return _get_qfa_ops()[0]

    @property
    def metadata_op(self) -> Any:
        """The QFA metadata (AICPU planning) operator."""
        return _get_qfa_ops()[1]

    def _nz_5d_view(self, cache: torch.Tensor, block_size: int) -> torch.Tensor:
        """View a natural (num_blocks, block_size, num_kv_heads, head_dim) C8
        MXFP cache tensor in the PA_NZ layout QFA reads:
        (num_blocks, num_kv_heads, head_dim//32, block_size, 32). Head counts
        are derived from the cache itself so models whose V head dim differs
        from Q/K stay correct."""
        num_kv_heads = cache.shape[2]
        head_dim = cache.shape[3]
        return cache.view(
            -1,
            num_kv_heads,
            head_dim // MXFP_KV_NZ_DIM_FRAG,
            block_size,
            MXFP_KV_NZ_DIM_FRAG,
        )

    def _qfa_step_cache(self, attn_metadata: AscendMetadata) -> dict:
        cache = getattr(attn_metadata, "qfa_metadata_cache", None)
        if cache is None:
            cache = {}
            attn_metadata.qfa_metadata_cache = cache
        return cache

    def _qfa_step_lengths(self, attn_metadata: AscendMetadata, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this step's (cu_seqlens_q, seqused_kv), derived once.

        Derived in-graph by the first full-attention layer; every other layer
        consumes the recorded tensors, and each replay re-executes that one
        derivation against the refreshed buffers.
        """
        cache = self._qfa_step_cache(attn_metadata)
        lengths = cache.get("lengths")
        if lengths is None:
            # Clamp sanitizes the tail: unused query_start_loc slots carry -1
            # (FIA padding convention) and may hold stale entries from larger
            # earlier steps; cummax restores monotonicity, turning the tail
            # into zero-length requests. seq_lens clamp(min=1) matches the
            # dummy-request convention (block 0, one token).
            lengths = (
                attn_metadata.query_start_loc_gpu.clamp(min=0, max=num_tokens).cummax(dim=0).values,
                attn_metadata.seq_lens_gpu.clamp(min=1),
            )
            cache["lengths"] = lengths
        return lengths

    def _qfa_k_scale_slot_index(
        self, attn_metadata: AscendMetadata, slot_mapping: torch.Tensor, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return this step's K-scale slot decomposition, derived once.

        Same story as _qfa_step_lengths: six device ops over slot_mapping
        that every layer used to repeat, cached inside the captured region so
        replay re-derives them exactly once.
        """
        cache = self._qfa_step_cache(attn_metadata)
        slot_index = cache.get("k_scale_slots")
        if slot_index is None:
            slot_index = mxfp_k_scale_slot_index(slot_mapping, block_size)
            cache["k_scale_slots"] = slot_index
        return slot_index

    @staticmethod
    def _qfa_v_descale_placeholder(value_scale_cache: torch.Tensor) -> torch.Tensor:
        """The v_descale the metadata op insists on, without allocating one.

        quant_mode=1 refuses a null v_descale at the aclnn entry, but under
        PA_NZ nothing reads it. An allocation inside the captured region
        would record its zero-fill as a graph node replayed every step; a
        view of the layer's own V scale cache launches nothing and has an
        address that predates every capture.

        NOTE: torch_npu.float8_e8m0fnu is the integer dtype ID (293) on this
        torch_npu build, not a torch.dtype; bitcast with the stock torch
        dtype instead.
        """
        return value_scale_cache.view(-1)[:2].view(1, 1, 1, 1, 1, 2).view(torch.float8_e8m0fnu)

    def _get_qfa_metadata(
        self,
        attn_metadata: AscendMetadata,
        *,
        cu_seqlens_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        value_scale_cache: torch.Tensor,
        max_seqlen_q: int,
        mask_mode: int,
        layout_q_descale: str,
    ):
        """Return the QFA metadata plan (AICPU op output), derived once a step.

        The plan's inputs are all non-layer-specific (head counts, lengths,
        mask mode, layouts), so one plan serves every full-attention layer of
        a step. Sharing is safe under graph capture for the same reason as
        _qfa_step_lengths: the deriving call runs inside the captured region,
        ahead of every QFA call that reads its output, and the main operator
        treats ``metadata`` as a read-only input.
        """
        cache = self._qfa_step_cache(attn_metadata)
        plan_key = (
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            max_seqlen_q,
            mask_mode,
            layout_q_descale,
        )
        metadata = cache.get(plan_key)
        if metadata is None:
            # TND + PA: pass cu_seqlens_q only; the KV side is addressed via
            # block_table + seqused_kv. batch_size must NOT be passed with a
            # TND layout_q (the checker rejects it).
            metadata = self.metadata_op(
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                QFA_QUANT_MODE_MXFP8,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=None,
                seqused_q=None,
                seqused_kv=seqused_kv,
                v_descale=self._qfa_v_descale_placeholder(value_scale_cache),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=-1,
                mask_mode=mask_mode,
                win_left=-1,
                win_right=-1,
                layout_q=QFA_LAYOUT_TND,
                layout_q_descale=layout_q_descale,
                layout_kv=QFA_LAYOUT_PA_NZ,
                layout_out=QFA_LAYOUT_TND,
            )
            cache[plan_key] = metadata
        return metadata

    def _qfa_int8_mask(self, attn_metadata: AscendMetadata) -> torch.Tensor | None:
        """QFA's attn_mask is INT8/UINT8/bool; the shared builder already
        hands out an int8 causal mask, so only convert other sources."""
        if attn_metadata.attn_mask is None:
            return None
        if attn_metadata.attn_mask.dtype == torch.int8:
            return attn_metadata.attn_mask
        return attn_metadata.attn_mask.to(torch.int8)

    def _qfa_query_scale_for_layout(self, query_scale: torch.Tensor, max_seqlen_q: int) -> tuple[torch.Tensor, str]:
        """Return the q scale in the layout that selects the right QFA kernel.

        TND ``(Q_T, Q_N, D/64, 2)`` compiles the prefill template; N2TGD
        ``(KV_N, Q_T, G, D/64, 2)`` the decode one (the query-head axis
        split into ``(KV_N, G)``, KV-head half hoisted in front of the token
        axis; query heads are GQA-contiguous, so the split is a reshape and
        a permute). The split is a throughput choice -- both layouts carry
        identical scales -- following the operator doc's G*Q_S boundary.

        The decision reads the query shape, not the scheduler state, so MTP
        verify steps take the decode layout too.
        """
        # Head counts that do not split into whole kv-head groups (possible
        # on MTP draft layers) cannot be reshaped; keep TND instead.
        if self.num_kv_heads == 0 or query_scale.shape[1] % self.num_kv_heads != 0:
            return query_scale, QFA_LAYOUT_TND
        group_size = query_scale.shape[1] // self.num_kv_heads
        if group_size * max_seqlen_q > QFA_QSCALE_N2TGD_MAX_G_TIMES_QS:
            return query_scale, QFA_LAYOUT_TND
        # Permute through a uint8 byte view: float8 transpose/contiguous fall
        # back to AICPU. _run_qfa bitcasts back to E8M0 at the call boundary.
        scale_bytes = query_scale.view(torch.uint8)
        num_tokens = scale_bytes.shape[0]
        n2tgd = (
            scale_bytes.view(num_tokens, self.num_kv_heads, group_size, *scale_bytes.shape[2:])
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        return n2tgd, QFA_LAYOUT_N2TGD

    def _run_qfa(
        self,
        quant_query: torch.Tensor,
        query_scale: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        *,
        cu_seqlens_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        qfa_metadata,
        max_seqlen_q: int,
        mask_mode: int,
        layout_q_descale: str,
        num_tokens: int,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, key_scale, value_scale = kv_cache
        # The K/V caches stay in natural storage; QFA reads them through the
        # PA_NZ 5-D view. The scale caches are stored as raw uint8/int8
        # (index_put_ on float8 falls back to AICPU); QFA's checker wants
        # E8M0, so bitcast at the call boundary (1:1 byte reinterpretation
        # with the torch dtype -- torch_npu.float8_e8m0fnu is an integer ID
        # here, not a dtype).
        key = self._nz_5d_view(key, key.shape[1])
        value = self._nz_5d_view(value, value.shape[1])
        key_scale = key_scale.view(torch.float8_e8m0fnu)
        value_scale = value_scale.view(torch.float8_e8m0fnu)
        query_scale = query_scale.view(torch.float8_e8m0fnu)
        main_op = self.main_op
        # cann_ops_transformer delivery signature (verified on-device): the
        # allocating wrapper is capture-safe under npugraph_ex (ops-transformer
        # golden tests capture exactly this call, GRAPH_PATH=7).
        result = main_op(
            quant_query,
            key,
            value,
            query_scale,
            key_scale,
            value_scale,
            QFA_QUANT_MODE_MXFP8,
            block_table=attn_metadata.block_tables,
            p_scale=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            sinks=None,
            # The doc forbids attn_mask under NO_MASK and requires it under
            # CAUSAL, so the two travel together.
            attn_mask=(None if mask_mode == QFA_MASK_MODE_NO_MASK else self._qfa_int8_mask(attn_metadata)),
            metadata=qfa_metadata,
            softmax_scale=self.scale,
            mask_mode=mask_mode,
            win_left=-1,
            win_right=-1,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=-1,
            layout_q=QFA_LAYOUT_TND,
            layout_q_descale=layout_q_descale,
            layout_kv=QFA_LAYOUT_PA_NZ,
            layout_out=QFA_LAYOUT_TND,
            return_softmax_lse=False,
        )
        # return_softmax_lse=False yields an empty LSE tensor in the
        # cann_ops flavor; tolerate both tuple and single-tensor returns.
        attn_output = result[0] if isinstance(result, tuple) else result
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output

    def _forward_mxfp8_attention(
        self,
        quant_query: torch.Tensor,
        query_scale: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """One QFA call per step for the whole batch (decode + prefill alike).

        The causal mask covers decode rows as well, so the per-subset split of
        the FIA-based design is unnecessary. PrefillNoCache is included:
        reshape_and_cache has already written this step's K/V into the paged
        cache before attention runs.

        Graph capture (npugraph_ex, GRAPH_PATH=7): cu_seqlens_q / seqused_kv
        are derived on device from the runner's persistent buffers, so every
        replay re-executes the derivation and sees the current step's lengths.
        """
        if not attn_metadata.causal:
            raise NotImplementedError("C8_MXFP attention does not support non-causal attention yet.")
        if self.sliding_window is not None:
            raise NotImplementedError("C8_MXFP attention does not support sliding window attention yet.")

        # T as QFA sees it: the query rows actually fed to the operator
        # (forward slices query[:num_actual_tokens] before quantizing), so
        # the TND constraint cu_seqlens_q[-1] == T, the view/output row
        # counts and the sanitize clamp bound below all share one source.
        # At graph capture this is the bucket size; eagerly it is the real
        # token count -- both stay clean where actual_seq_lengths_q[-1]
        # may carry stale tail entries (padded batches).
        num_tokens = quant_query.shape[0]
        if num_tokens <= 0:
            return output

        qsl_gpu = attn_metadata.query_start_loc_gpu
        seq_lens_gpu = attn_metadata.seq_lens_gpu
        if qsl_gpu is None or seq_lens_gpu is None:
            raise RuntimeError(
                "C8_MXFP attention requires the GPU-side length sources "
                "(query_start_loc_gpu / seq_lens_gpu) on AscendMetadata."
            )
        cu_seqlens_q, seqused_kv = self._qfa_step_lengths(attn_metadata, num_tokens)
        # The longest single query in the batch -- NOT the batch token total.
        # The metadata op seeds querySeqSize with this attr and only ever
        # raises it, so an inflated value is never walked back.
        max_seqlen_q = attn_metadata.max_query_len or num_tokens

        # Drop the mask entirely when every request contributes a single query
        # row. hasAttenMask is one of the tiling-key axes, so NO_MASK selects
        # a kernel template that never reads the mask; at Q_S == 1 that is
        # exactly equivalent to CAUSAL (the KV range is bounded by seqused_kv).
        # MTP verify steps (Q_S = 1 + num_spec) keep CAUSAL.
        mask_mode = QFA_MASK_MODE_NO_MASK if max_seqlen_q == 1 else QFA_MASK_MODE_CAUSAL

        # Both operators have to agree on the q scale layout: it is what
        # picks the kernel, and the metadata plan is computed for that kernel.
        query_scale, layout_q_descale = self._qfa_query_scale_for_layout(query_scale, max_seqlen_q)

        qfa_metadata = self._get_qfa_metadata(
            attn_metadata,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            value_scale_cache=kv_cache[3],
            max_seqlen_q=max_seqlen_q,
            mask_mode=mask_mode,
            layout_q_descale=layout_q_descale,
        )
        return self._run_qfa(
            quant_query,
            query_scale,
            kv_cache,
            attn_metadata,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            qfa_metadata=qfa_metadata,
            max_seqlen_q=max_seqlen_q,
            mask_mode=mask_mode,
            layout_q_descale=layout_q_descale,
            num_tokens=num_tokens,
            output=output,
        )

    # KV cache writes happen in reshape_and_cache(), invoked from forward()
    # when key/value are present. This hook is only reached when attention is
    # split from cache update (forward_includes_kv_cache_update=False);
    # AscendAttentionBackend keeps it True, so normal inference never calls
    # this.
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        raise NotImplementedError("C8_MXFP KV cache update is only supported via reshape_and_cache in forward().")

    def reshape_and_cache(  # type: ignore[override]
        self,
        quant_key: torch.Tensor,
        quant_value: torch.Tensor,
        key_scale: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
    ) -> None:
        num_actual_tokens = quant_key.shape[0]
        slot_mapping = attn_metadata.slot_mapping[:num_actual_tokens]
        key_cache, value_cache = kv_cache[0], kv_cache[1]
        block_size = key_cache.shape[1]
        # Write the K/V payloads in the PA_NZ layout QFA reads; allocation,
        # hybrid partitioning, PD and CoW keep seeing the natural storage.
        scatter_mxfp_pa_nz_kv_cache(
            quant_key,
            quant_value,
            key_cache,
            value_cache,
            slot_mapping,
            block_size,
        )

        # Only K's scale is per-token; V's static scale is filled once at KV
        # cache setup by NPUModelRunner._fill_c8_mxfp_v_scale_caches.
        scatter_mxfp_k_scale_cache(
            # Byte view: index_put_ on float8 falls back to AICPU.
            key_scale.view(torch.uint8) if key_scale.dtype != torch.uint8 else key_scale,
            kv_cache[2],
            self._qfa_k_scale_slot_index(attn_metadata, slot_mapping, block_size),
        )
        notify_kv_cache_written()

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for AscendC8MXFPAttentionBackendImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)
        if getattr(self, "enable_hamming_sparse", False):
            raise NotImplementedError("C8_MXFP attention does not support hamming sparse KV compression yet.")
        if self.vllm_config.kv_transfer_config is not None and not uses_mooncake_connector(
            self.vllm_config.kv_transfer_config
        ):
            # Other connectors move only the K/V pair and would silently drop
            # the scale caches.
            raise NotImplementedError(
                "C8_MXFP v1 PD disaggregation (kv_transfer) is only supported with "
                "MooncakeConnectorV1, whose block-level transfer registers and moves "
                "every per-layer cache tensor (FP8 K/V plus both E8M0 scale caches) "
                "as raw blocks."
            )
        if kv_cache is None or len(kv_cache) < 4:
            raise RuntimeError(
                "C8_MXFP attention requires a (k, v, k_scale, v_scale) KV cache "
                f"tuple, got: {type(kv_cache)} with length "
                f"{len(kv_cache) if kv_cache is not None else 0}."
            )

        record_attention_compute_start()

        query_mxfp8, query_scale = torch_npu.npu_dynamic_mx_quant(
            query[: attn_metadata.num_actual_tokens],
            dst_type=torch.float8_e4m3fn,
        )

        # KV-sharing consumer layers reuse another layer's cache; writing
        # their (dummy) K/V would corrupt the shared slots, so only the
        # owner layer quantizes and scatters K/V.
        if key is not None and value is not None and self.kv_sharing_target_layer_name is None:
            key_mxfp8, key_scale = torch_npu.npu_dynamic_mx_quant(
                key[: attn_metadata.num_actual_tokens],
                dst_type=torch.float8_e4m3fn,
            )

            original_value_shape = value.shape
            value = value.view(original_value_shape[0], -1)
            value_mxfp8 = torch_npu.npu_quantize(
                value[: attn_metadata.num_actual_tokens],
                layer.v_cache_scale_float_reciprocal,
                None,
                torch.float8_e4m3fn,
                -1,
                False,
            )
            value_mxfp8 = value_mxfp8.view((attn_metadata.num_actual_tokens, *original_value_shape[1:]))

            self.reshape_and_cache(key_mxfp8, value_mxfp8, key_scale, kv_cache, attn_metadata)

        # PA_NZ: QFA reads the paged cache through the NZ view taken in
        # _run_qfa, so the cache tuple is passed through as-is -- no
        # transpose, no storage copy.
        return self._forward_mxfp8_attention(query_mxfp8, query_scale, kv_cache, attn_metadata, output)
