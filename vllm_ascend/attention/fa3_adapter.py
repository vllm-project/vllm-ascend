"""
flash-attention-npu (FA3) adapter for vllm-ascend.

**Limitations** (these are validated at runtime and raise ``ValueError``):
  - ``head_dim > FA3_MAX_HEAD_DIM`` (default 256) is not supported.
  - FP8 / INT8 quantized KV caches are not handled here (they use separate
    C8-attention backends in vllm-ascend).

Maps vllm-ascend's attention parameter formats to the flash-attention-npu API,
allowing FA3 to replace CANN's npu_fused_infer_attention_score (V1 FIA)
in eager-mode forward paths.

For features FA3 does not support (learnable_sink, ACL graph capture),
the original CANN paths remain as fallback.

Data format differences bridged here:
  - vllm cumulative seq lengths WITHOUT a leading zero   → FA3 format WITH leading zero
  - vllm ``(num_blocks, block_size, -1)`` flattened cache → FA3 ``(num_blocks, block_size, H, D)``
  - vllm ``sparse_mode=3/4`` / ``pre_tokens``              → FA3 ``causal`` / ``window_size``
"""

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

try:
    from flash_attn_npu_3 import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache as fa3_kvcache,
        get_scheduler_metadata,
    )
    HAS_FLASH_ATTN_NPU = True
except ImportError:
    HAS_FLASH_ATTN_NPU = False

# FA3 kernel compilation limits (set by ``round_up_headdim`` in flash-attention-npu).
FA3_MAX_HEAD_DIM = 256

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_cu_seqlens(actual_seq_lengths: List[int], device: torch.device) -> torch.Tensor:
    """Convert vllm's cumulative seq lengths (no leading zero) to FA3's format.

    vllm (cumulative, no leading 0):  ``[l0, l0+l1, …]``
    FA3 cu_seqlens (with leading 0):  ``[0, l0, l0+l1, …]``

    Returns int32 tensor on ``device``.
    """
    return torch.tensor([0] + actual_seq_lengths, dtype=torch.int32, device=device)


def _max_seqlen(cumulative: List[int]) -> int:
    """Maximum *individual* sequence length from a cumulative list."""
    if not cumulative:
        return 0
    max_len = cumulative[0]
    for i in range(1, len(cumulative)):
        seq_len = cumulative[i] - cumulative[i - 1]
        if seq_len > max_len:
            max_len = seq_len
    return max_len


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def strip_padding_dummy(
    actual_seq_lengths_q: List[int],
    seq_lens_list: List[int],
):
    """Drop the padding dummy segment from a (possibly padded) batch.

    vllm-ascend pads cudagraph batches by appending ONE dummy request whose
    query spans all padding tokens and whose KV length is 0.  Passing that
    segment to FA3 as a real request both corrupts the metadata fingerprint
    (max_seqlen_q explodes) and makes the kernel allocate a query workspace
    sized by the padding (multi-GiB -> OOM).

    Returns ``(real_cu_q, real_seq_lens, max_seqlen_q)`` where ``real_cu_q``
    is the cumulative query layout (WITH leading zero) of the REAL requests.
    Real requests keep their original order and occupy the query tensor
    contiguously from row 0, so the rebuilt cu stays valid.
    """
    cu = [0] + list(actual_seq_lengths_q)
    real_cu = [0]
    real_kv = []
    for i, kv in enumerate(seq_lens_list):
        q_i = cu[i + 1] - cu[i]
        if kv == 0 and q_i > 1:
            continue  # padding dummy
        real_kv.append(kv)
        real_cu.append(real_cu[-1] + q_i)
    max_q = 0
    for i in range(1, len(real_cu)):
        max_q = max(max_q, real_cu[i] - real_cu[i - 1])
    return real_cu, real_kv, max_q


def fa3_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_metadata,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window: Optional[int] = None,
    causal: bool = True,
    cache_mode: bool = False,
    block_table: Optional[torch.Tensor] = None,
    seq_lens_list: Optional[List[int]] = None,
    scheduler_metadata=None,
) -> torch.Tensor:
    """Unified FA3 attention forward.

    ``cache_mode=False`` (PrefillNoCache)
        Key/value are the **dense** prefill tensors; calls
        :func:`flash_attn_varlen_func`.

    ``cache_mode=True`` (PrefillCacheHit / DecodeOnly / ChunkedPrefill)
        Key/value are **paged cache views** ``(num_blocks, block_size, -1)``;
        calls :func:`flash_attn_with_kvcache`.  *block_table* and
        *seq_lens_list* are required.  *scheduler_metadata* (pre-computed
        by :func:`get_scheduler_metadata`) may be passed to avoid internal
        re-computation in FA3.


    Returns
        Tensor ``(total_tokens, num_heads, head_size)``.
    """
    if not HAS_FLASH_ATTN_NPU:
        raise ImportError("flash-attention-npu is not installed")
    if head_size > FA3_MAX_HEAD_DIM:
        raise ValueError(
            f"flash-attention-npu supports head_dim <= {FA3_MAX_HEAD_DIM}, "
            f"got {head_size}"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"FA3 requires num_heads divisible by num_kv_heads, "
            f"got {num_heads} vs {num_kv_heads}"
        )

    device = query.device
    actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
    # For non-causal attention the original CANN V1 path uses ``sparse_mode=0``
    # (no mask at all), so we always pass a full window regardless of
    # *sliding_window*.  This matches existing behaviour and avoids applying
    # an unintended local bias on non-causal layers (e.g. cross-attention).
    if causal and sliding_window is not None:
        window_size = (sliding_window, 0)
    else:
        window_size = (-1, -1)

    if cache_mode:
        # ---- paged KV cache path ----
        assert block_table is not None, "block_table required for cache_mode=True"
        assert seq_lens_list is not None, "seq_lens_list required for cache_mode=True"

        # key   → (num_blocks, block_size, num_kv_heads, head_size)
        # value → (num_blocks, block_size, num_kv_heads, head_size)
        num_blocks, bs = key.shape[0], key.shape[1]
        k_fa = key.view(num_blocks, bs, num_kv_heads, head_size)
        v_fa = value.view(num_blocks, bs, num_kv_heads, head_size)

        # Strip the padding dummy segment (KV len 0, query spanning all
        # padding tokens); see strip_padding_dummy.
        from vllm_ascend.attention.fa3_adapter import strip_padding_dummy

        real_cu, real_kv, max_seqlen_q = strip_padding_dummy(
            actual_seq_lengths_q, seq_lens_list,
        )
        cache_seqlens = torch.tensor(real_kv, dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor(real_cu, dtype=torch.int32, device=device)

        out = fa3_kvcache(
            query,
            k_fa,
            v_fa,
            cache_seqlens=cache_seqlens,
            page_table=block_table.contiguous(),
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=scale,
            causal=causal,
            window_size=window_size,
            scheduler_metadata=scheduler_metadata,
        )
    else:
        # ---- dense prefill path (no KV cache) ----
        # For PrefillNoCache the KV lengths equal the query lengths.
        cu_seqlens_q = _to_cu_seqlens(actual_seq_lengths_q, device)
        cu_seqlens_k = _to_cu_seqlens(actual_seq_lengths_q, device)  # same as q
        max_seqlen_q = _max_seqlen(actual_seq_lengths_q)
        max_seqlen_k = max_seqlen_q

        out = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scale,
            causal=causal,
            window_size=window_size,
        )

    # FA3 returns (total_q, num_heads, head_size) — exactly what we need.
    return out
