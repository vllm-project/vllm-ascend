# SPDX-License-Identifier: Apache-2.0
"""Cross-model KV sharing for Ascend attention backends.

Implements the upstream ``kv_sharing_target_layer_name`` contract: a draft
attention layer that reads K/V from a target model layer's paged KV cache
instead of owning its own.  Upstream vLLM serves this via the TRITON_ATTN
backend; Ascend has no such backend, so the consumption is implemented here
as a first-class attention capability.

This module is the shared contract between two layers:

* **spec_decode layer** (proposer init) registers the binding of each draft
  layer to its target via :func:`bind`.
* **attention layer** (forward) reads it back via :func:`binding_of` and runs
  the shared-KV attention via the forward hooks.

The helpers self-gate on ``kv_sharing_target_layer_name is not None``, so they
are generic to any cross-model KV-sharing draft model (Gemma4 MTP is the first
user) and stay no-op for normal layers.  No config-based gate is used: during
draft forward the active config is the draft model's config, which has no
``speculative_config``, so a config gate would misfire as False (see the
draft-forward-misfire lesson documented across this PR).
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vllm_ascend.device.utils import _gather_paged_kv_to_dense


# ---------------------------------------------------------------------------
# Binding registry (typed view over impl attributes)
#
# A draft attention backend impl is bound to its target by three pieces of
# state, written at proposer-init time and read at forward time:
#   * target_impl  : the target layer's attention backend (owning the real KV)
#   * gid          : this layer's KV-cache-group id (for per-group block tables)
#   * bt_ref       : the proposer's {gid: block_table} dict
#
# These are stored as ``_kv_share_*`` attributes on the impl and surfaced
# through ``binding_of`` so callers never touch the attribute names directly.
# ---------------------------------------------------------------------------


@dataclass
class KVShareBinding:
    """Resolved cross-model KV-sharing binding for one attention backend."""

    target_impl: Any
    gid: int | None
    bt_ref: dict | None


def bind(
    impl,
    *,
    target_impl: Any = None,
    gid: int | None = None,
    bt_ref: dict | None = None,
) -> None:
    """Register (part of) a KV-sharing binding on a draft attention backend.

    Each keyword is optional so the two-phase init can set the target/bt_ref
    (during ``_fix_draft_kv_head_counts``) and the gid (later, during
    ``initialize_attn_backend``) independently.  Only provided values are
    written, so repeated calls merge rather than overwrite.
    """
    if target_impl is not None:
        impl._kv_share_target_impl = target_impl
    if bt_ref is not None:
        impl._kv_share_bt_ref = bt_ref
    if gid is not None:
        impl._kv_share_gid = gid


def binding_of(impl) -> KVShareBinding | None:
    """Return the KV-sharing binding of ``impl``, or None for normal layers.

    Returns None when ``impl`` is not a KV-sharing layer
    (``kv_sharing_target_layer_name is None``); otherwise a dataclass view of
    the three ``_kv_share_*`` attributes (each may still be unset).
    """
    if getattr(impl, "kv_sharing_target_layer_name", None) is None:
        return None
    return KVShareBinding(
        target_impl=getattr(impl, "_kv_share_target_impl", None),
        gid=getattr(impl, "_kv_share_gid", None),
        bt_ref=getattr(impl, "_kv_share_bt_ref", None),
    )


# ---------------------------------------------------------------------------
# Shared-KV attention
# ---------------------------------------------------------------------------


def _forward_shared_kv_prefill_attention(
    impl,
    query: torch.Tensor,
    shared_key: torch.Tensor,
    shared_value: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Manual PyTorch attention with already-dense shared KV from block_table.

    Ascend FIA (npu_fusion_attention) cannot handle cross-attention where
    actual_seq_qlen differs from actual_seq_kvlen — it either crashes with
    mask shape errors or produces zero output.  Use PyTorch's
    scaled_dot_product_attention instead, which correctly supports
    cross-attention with arbitrary Q/KV lengths and GQA (grouped-query
    attention).
    """
    num_tokens = attn_metadata.actual_seq_lengths_q[-1]
    q = query[:num_tokens]  # [T, H, D]
    k = shared_key  # [S, Hkv, D]
    v = shared_value  # [S, Hkv, D]

    # Build a block-diagonal causal mask that respects per-request
    # boundaries.  The flattened [num_tokens, S] batch concatenates
    # requests; a single global causal mask would let row i of request r
    # attend to KV columns belonging to request r' < r (cross-request
    # leak).
    #
    # Per-request Q lengths come from actual_seq_lengths_q (cumulative,
    # so diff to get per-request).  Per-request KV lengths come from
    # seq_lens_list.  Q and KV lengths differ for cross-attention (MTP
    # draft decode: Q=1 new token, KV=full sequence), so we must track
    # them independently.
    S = k.shape[0]
    cum_q = attn_metadata.actual_seq_lengths_q
    if cum_q and len(cum_q) > 1:
        q_lens = [cum_q[i] - cum_q[i - 1] for i in range(1, len(cum_q))]
    else:
        q_lens = [num_tokens]
    kv_lens = attn_metadata.seq_lens_list or [S]
    # Pair Q and KV lengths per request; if counts mismatch (padding),
    # zip stops at the shorter — both lists should have the same number
    # of real requests.
    mask = torch.full((num_tokens, S), float("-inf"), dtype=q.dtype, device=q.device)
    q_off = 0
    kv_off = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        if q_len <= 0 or kv_len <= 0:
            q_off += q_len
            kv_off += kv_len
            continue
        # Query row j (0-indexed in this request's Q block) is the
        # (kv_len - q_len + j)-th token of the full sequence.  It attends
        # to KV columns [0, kv_len - q_len + j] (causal), restricted to
        # the sliding window if configured.
        offset = kv_len - q_len
        for j in range(q_len):
            causal_pos = offset + j  # position in the full sequence
            window_start = (
                max(0, causal_pos - impl.sliding_window + 1)
                if impl.sliding_window is not None and impl.sliding_window < kv_len
                else 0
            )
            mask[q_off + j, kv_off + window_start : kv_off + causal_pos + 1] = 0
        q_off += q_len
        kv_off += kv_len
    attn_mask = mask

    # Handle GQA: expand KV heads to match Q heads.
    # Ascend NPU's scaled_dot_product_attention does not broadcast
    # head dimension, so we must explicitly repeat KV heads.
    if q.shape[1] != k.shape[1]:
        n_rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(n_rep, dim=1)  # [S, Hkv, D] -> [S, Hq, D]
        v = v.repeat_interleave(n_rep, dim=1)  # [S, Hkv, D] -> [S, Hq, D]

    # Always use 4D format [B, H, L, D] for Ascend NPU.
    q_4d = q.unsqueeze(0).transpose(1, 2)  # [T, H, D] -> [1, H, T, D]
    k_4d = k.unsqueeze(0).transpose(1, 2)  # [S, H, D] -> [1, H, S, D]
    v_4d = v.unsqueeze(0).transpose(1, 2)  # [S, H, D] -> [1, H, S, D]
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [T, S] -> [1, 1, T, S]
    attn_output = F.scaled_dot_product_attention(
        q_4d,
        k_4d,
        v_4d,
        attn_mask=attn_mask,
        scale=impl.scale,
    )  # [1, H, T, D]
    attn_output = attn_output.squeeze(0).transpose(0, 1)  # [T, H, D]

    output[:num_tokens] = attn_output
    return output


def _get_current_token_shared_kv(
    impl,
    attn_metadata,
) -> tuple:
    """Gather current-token KV from the producer layer's shared cache."""
    if impl.key_cache is None or impl.value_cache is None:
        return None, None
    num_tokens = attn_metadata.actual_seq_lengths_q[-1]
    if attn_metadata.slot_mapping is None or attn_metadata.slot_mapping.numel() < num_tokens:
        return None, None
    slots = attn_metadata.slot_mapping[:num_tokens].long()
    key = impl.key_cache.reshape(-1, impl.num_kv_heads, impl.head_size).index_select(0, slots)
    value = impl.value_cache.reshape(-1, impl.num_kv_heads, impl.head_size).index_select(0, slots)
    return key, value


def _get_shared_kv_from_block_table(
    impl,
    attn_metadata,
) -> tuple:
    """Gather K/V from the shared target cache using block tables.

    Used when slot_mapping is not available (e.g., during speculative
    decoding where the draft model inherits attn_metadata from the
    target but slot_mapping may not be populated for draft layers).

    For KV-sharing draft layers, impl.key_cache points to the draft
    model's own (empty) cache.  We must swap to the target layer's
    cache via the binding's target_impl, mirroring the PA path fix.
    """
    binding = binding_of(impl)
    tgt_impl = binding.target_impl if binding is not None else None
    if tgt_impl is not None and tgt_impl.key_cache is not None:
        read_kc = tgt_impl.key_cache
        read_vc = tgt_impl.value_cache
    else:
        read_kc = impl.key_cache
        read_vc = impl.value_cache

    if read_kc is None or read_vc is None:
        return None, None

    # Per-group block-table routing: draft layers share KV with target
    # layers that may be in DIFFERENT KV cache groups.  attn_metadata.block_tables
    # is the common (gid=0) table; using it for layers whose target is in gid!=0
    # reads from the wrong pool.  Route each layer to its per-group block_table
    # via the binding's gid + bt_ref ({gid: block_table} set by
    # set_per_group_block_table).
    #
    # We deliberately do NOT probe other groups' block tables by reading KV
    # tensor means: that would force an NPU->CPU sync (.item()) on the
    # attention hot path, use a data-dependent zero/non-zero heuristic,
    # and a broad `except Exception` that hides real routing/shape bugs.
    # If the routed block table is wrong, fail loudly so the routing is
    # fixed at the source rather than masked here.
    my_gid = binding.gid if binding is not None else None
    bt_ref = binding.bt_ref if binding is not None else None
    routed_bt = None
    if my_gid is not None and bt_ref is not None and my_gid in bt_ref:
        routed_bt = bt_ref[my_gid]
    block_table = routed_bt if routed_bt is not None else attn_metadata.block_tables
    seq_lens = attn_metadata.seq_lens_list
    if block_table is None or not seq_lens:
        return None, None

    dense_key, dense_value = _gather_paged_kv_to_dense(
        read_kc,
        read_vc,
        block_table,
        seq_lens,
        impl.num_kv_heads,
        impl.head_size,
    )
    return dense_key, dense_value


def maybe_kv_share_prefill(
    impl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache,
    attn_metadata,
    output: torch.Tensor,
):
    """Intercept entry point for KV-sharing draft layers.

    Called at the top of AscendAttentionBackendImpl.forward_impl.  Returns
    the attention output tensor if this layer is a KV-sharing draft layer
    whose prefill should be computed from the shared target cache via
    PyTorch SDPA; returns None to let the caller fall through to the
    normal FIA / PA path.

    Also initialises impl.key_cache / impl.value_cache from the kv_cache
    tuple for KV-sharing layers (they do not own a private cache).
    """
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    # Ensure self.key_cache / self.value_cache are initialised from the
    # kv_cache tuple BEFORE the shared-KV lookup, otherwise they will be
    # None (draft layers do not own a private cache).
    if (
        impl.kv_sharing_target_layer_name is not None
        and impl.key_cache is None
        and kv_cache is not None
        and len(kv_cache) >= 2
    ):
        impl.key_cache, impl.value_cache = kv_cache[0], kv_cache[1]

    # Draft KV-sharing layers: always route to PA (head_dim=512) / FIA
    # (head_dim=256), which consume tensorised block_table/seq_lens. Never
    # take the SDPA path which gathers paged KV into a dense tensor —
    # for long sequences that allocation is fatal (30+ GiB OOM).
    _is_draft = getattr(_EXTRA_CTX, "is_draft_model", False)
    if _is_draft and impl.kv_sharing_target_layer_name is not None:
        return None

    _kv_prefill_eligible = (
        impl.kv_sharing_target_layer_name is not None
        and key is not None
        and value is not None
        and query.shape[0] == key.shape[0]
        and attn_metadata.attn_state
        in (
            AscendAttentionState.PrefillNoCache,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        )
    )
    if not _kv_prefill_eligible:
        return None

    # For SpecDecoding / draft-model layers, slot_mapping points to
    # empty/wrong positions (draft layers do not write KV).  Skip the
    # slot-based lookup and go straight to the block-table gather.
    if attn_metadata.attn_state != AscendAttentionState.SpecDecoding and not getattr(
        _EXTRA_CTX, "is_draft_model", False
    ):
        shared_key, shared_value = _get_current_token_shared_kv(impl, attn_metadata)
    else:
        shared_key, shared_value = None, None

    if shared_key is None or shared_value is None:
        shared_key, shared_value = _get_shared_kv_from_block_table(impl, attn_metadata)

    if shared_key is None or shared_value is None:
        return None

    return _forward_shared_kv_prefill_attention(
        impl,
        query,
        shared_key,
        shared_value,
        attn_metadata,
        output,
    )


def should_skip_draft_kv_write(impl) -> bool:
    """True for Q-only draft KV-sharing layers.

    Gemma4MTPAttention.forward() creates a dummy K/V via torch.empty() and
    passes it as key/value to self.attn().  Writing this uninitialized
    memory back via reshape_and_cache would corrupt the shared target KV
    cache, causing progressive degradation across sequential loop steps.
    Skip the write for draft KV-shared layers — they only READ.
    """
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    return getattr(impl, "kv_sharing_target_layer_name", None) is not None and getattr(
        _EXTRA_CTX, "is_draft_model", False
    )


def maybe_skip_reshape_for_kv_share(impl, attn_metadata) -> bool:
    """True if reshape_and_cache should be skipped for a KV-sharing target layer.

    KV-sharing target layers (e.g. Gemma4 MTP draft) consume the producer
    layer's cache.  Re-caching here would overwrite the shared KV slots
    before attention reads it.  When True the caller must still record the
    producer's reshape_cache_event (if this layer is a producer) and return
    early.
    """
    return getattr(impl, "kv_sharing_target_layer_name", None) is not None
