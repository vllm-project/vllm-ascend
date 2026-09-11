# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adapter for the ``cann_ops_transformer.flash_attn`` operator.

``flash_attn`` (and its two-phase companion ``flash_attn_metadata``) is the
CANN ops-transformer attention operator that natively consumes paged KV
caches whose first axis (the block axis) is non-contiguous. This is the
layout produced by hybrid (attention + Mamba) models after PR #14340:
the page is padded to the hybrid common page size, so the cache view has
``stride(0) > block_size * num_kv_heads * head_size`` while the block
interior stays contiguous.

This module keeps the operator loading lazy (matching sparse_flash_mla.py)
and translates the vLLM-Ascend GQA metadata/kwargs into the operator's
parameter convention (docs: attention/flash_attn/docs/torchapi_flash_attn.md
in the cann/ops-transformer repository; the file formerly lived at
torch_extension/cann_ops_transformer/docs/zh/flash_attn.md).
"""

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module, util

import torch

# Paged layouts accepted by the operator:
# - PA_BBND: (num_blocks, block_size, KV_N, D), only dim0 may be non-contiguous
# - PA_BNBD: (num_blocks, KV_N, block_size, D), dim0 or dim1 may be non-contiguous
# The vLLM-Ascend cache view is (num_blocks, block_size, KV_N, D) on current
# branches; callers pass the layout explicitly so PrefillNoCache can use TND.
LAYOUT_KV_PAGED_BBND = "PA_BBND"
LAYOUT_KV_PAGED_BNBD = "PA_BNBD"
LAYOUT_KV_TND = "TND"

# head_dim sizes validated by the operator for the QK == V case of standard
# GQA: (64,64), (72,72), (128,128), (256,256).
SUPPORTED_HEAD_DIMS = frozenset({64, 72, 128, 256})

# Minimum persistent metadata buffer size (elements) for ACLGraph mode.
# flash_attn_metadata emits a fixed-size tiling/task-split descriptor for a
# given (num_heads, head_dim) configuration; 4096 leaves headroom for large
# batch/head configurations (cf. DSA's 1024-element metadata buffers). The
# buffer is grown to the operator output size on first use if that is larger.
CANN_FLASH_ATTN_METADATA_BUFFER_SIZE = 4096


@lru_cache
def _get_flash_attn_ops() -> tuple[Callable, Callable]:
    """Load flash_attn operators without importing vllm_ascend.ops."""
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return namespace.flash_attn, namespace.flash_attn_metadata
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("cann_ops_transformer.flash_attn requires a matching CANN toolkit on Ascend A5.") from exc


def is_cann_ops_flash_attn_available() -> bool:
    """Return True when the cann_ops_transformer extension is importable."""
    return util.find_spec("cann_ops_transformer") is not None


def _filter_none(kwargs: dict) -> dict:
    """Drop None values: optional int attrs default to -1 internally and
    passing None to an int parameter breaks the op binding."""
    return {k: v for k, v in kwargs.items() if v is not None}


_ATTN_CAUSAL_MASK: torch.Tensor | None = None


def get_causal_attn_mask(device: torch.device) -> torch.Tensor:
    """Fixed (2048, 2048) int8 lower-triangular causal mask required by
    mask_mode=3 (and window mode 4), per the operator's Mask param-group
    constraints. Lazily allocated once and kept alive so eager decode/prefill
    calls reuse the same buffer.

    NOTE(semantics): polarity confirmed against the operator README: §1.5
    (mask_mode=3 = causal, lower-triangular) and §1.7 (an all-1 mask row
    makes the row invalid, i.e. 1 = masked/excluded). triu(diagonal=1)
    therefore marks exactly the strictly-upper (future) positions as
    masked, which is the causal convention vLLM expects.
    """
    global _ATTN_CAUSAL_MASK
    if _ATTN_CAUSAL_MASK is None or _ATTN_CAUSAL_MASK.device != device:
        _ATTN_CAUSAL_MASK = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=device), diagonal=1)
    return _ATTN_CAUSAL_MASK


def flash_attn_metadata_adapter(**kwargs) -> torch.Tensor:
    """Build flash_attn metadata from GQA attention metadata kwargs.

    When ``out_buffer`` is given (ACLGraph mode), the operator output is
    copied into that persistent buffer and the buffer is returned instead,
    so the main flash_attn op can be captured on a stable address; the
    buffer contents are refreshed by the attention metadata builder every
    step (same discipline as DSA's sas/qli metadata buffers).
    """
    _, metadata_op = _get_flash_attn_ops()
    # flash_attn_metadata has no q/k/v tensor inputs; it only computes the
    # AICore/AIVCore task split from the length/mask description below.
    params = _filter_none(
        {
            "num_heads_q": kwargs["num_heads_q"],
            "num_heads_kv": kwargs["num_heads_kv"],
            "head_dim": kwargs["head_dim"],
            "cu_seqlens_q": kwargs.get("cu_seqlens_q"),
            "cu_seqlens_kv": kwargs.get("cu_seqlens_kv"),
            "seqused_kv": kwargs.get("seqused_kv"),
            "batch_size": kwargs.get("batch_size"),
            "max_seqlen_q": kwargs.get("max_seqlen_q"),
            "max_seqlen_kv": kwargs.get("max_seqlen_kv"),
            "mask_mode": kwargs.get("mask_mode", 0),
            "win_left": kwargs.get("win_left", -1),
            "win_right": kwargs.get("win_right", -1),
        }
    )
    # layout_q/layout_kv/layout_out must match the flash_attn call: TND q
    # requires layout_out=TND (default BSND is rejected by the layout
    # matching table).
    metadata = metadata_op(
        **params,
        layout_q=kwargs.get("layout_q", "TND"),
        layout_kv=kwargs.get("layout_kv", LAYOUT_KV_PAGED_BBND),
        layout_out=kwargs.get("layout_out", "TND"),
    )
    out_buffer = kwargs.get("out_buffer")
    if out_buffer is None:
        return metadata
    num_elements = metadata.numel()
    if num_elements > out_buffer.numel():
        raise ValueError(
            f"flash_attn_metadata output needs {num_elements} elements but the "
            f"persistent graph-mode buffer only holds {out_buffer.numel()}."
        )
    out_buffer[:num_elements].copy_(metadata)
    return out_buffer


def flash_attn_adapter(q: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt vLLM-Ascend GQA attention kwargs to flash_attn.

    Key translation from the npu_fused_infer_attention_score convention:
    - ``key``/``value`` must be the raw paged cache views
      ``(num_blocks, block_size, num_kv_heads, head_size)``; do NOT flatten
      with ``view(num_block, block_size, -1)`` (the view fails on
      non-contiguous caches) and do NOT call ``.contiguous()`` (a full
      cache copy defeats the first-axis non-contiguous support and loses
      in-place cache writes). For the no-cache prefill pass TND k/v and
      ``layout_kv="TND"`` (the operator requires contiguous k/v for TND,
      which a prefix slice of a contiguous tensor still satisfies).
    - ``seq_lens`` (int32, (B,)) maps to ``seqused_kv``.
    - ``cu_seqlens_q`` (int32, (B+1,)) is the cumulative query length.
    - ``block_table`` must be int32 and indexed in kernel-block granularity,
      matching the cache view's block axis (cf. PR #14340).
    """
    attention_op, _ = _get_flash_attn_ops()
    params = _filter_none(
        {
            "block_table": kwargs.get("block_table"),
            "cu_seqlens_q": kwargs.get("cu_seqlens_q"),
            "cu_seqlens_kv": kwargs.get("cu_seqlens_kv"),
            "seqused_kv": kwargs.get("seqused_kv"),
            "sinks": kwargs.get("sinks"),
            "metadata": kwargs.get("metadata"),
            "softmax_scale": kwargs["softmax_scale"],
            "mask_mode": kwargs.get("mask_mode", 0),
            "attn_mask": kwargs.get("attn_mask"),
            "win_left": kwargs.get("win_left", -1),
            "win_right": kwargs.get("win_right", -1),
            "max_seqlen_q": kwargs.get("max_seqlen_q", -1),
            "max_seqlen_kv": kwargs.get("max_seqlen_kv", -1),
        }
    )
    return attention_op(
        q,
        kwargs["key"],
        kwargs["value"],
        **params,
        layout_q=kwargs.get("layout_q", "TND"),
        layout_kv=kwargs.get("layout_kv", LAYOUT_KV_PAGED_BBND),
        layout_out=kwargs.get("layout_out", "TND"),
    )
