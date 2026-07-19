# SPDX-License-Identifier: Apache-2.0
"""Cross-model KV sharing for Ascend attention backends.

Implements the upstream ``kv_sharing_target_layer_name`` contract: a draft
attention layer that reads K/V from a target model layer's paged KV cache
instead of owning its own.  Upstream vLLM serves this via the TRITON_ATTN
backend; Ascend has no such backend, so the consumption is implemented here.

A5 (is_950) -- the path that survives the September PA retirement:
    The draft layer and its target layer share a KV-cache group, hence the
    same physical cache tensors and the same per-request block table.  The
    FIA path (forward_fused_infer_attention) therefore reads the shared KV
    *implicitly* -- no per-forward swap is needed.  This module only
    contributes the read-only guards so the draft never writes dummy K/V
    back into the shared slots:
      * should_skip_draft_kv_write       -- Q-only draft layers skip reshape
      * maybe_skip_reshape_for_kv_share  -- graph-replay/capture second gate

A2/A3 (PagedAttention, retires September):
    PA graph-capture bakes a block-table tensor reference into the workspace,
    so the draft layer must swap to the target layer's cache + per-group
    block table at capture time.  That wiring lives in resolve_capture_kv /
    maybe_expand_paged_kv_for_verify and is reached only when the PA gate
    (``not is_950()``) is active.

The historical PyTorch-SDPA cross-attention path (gathering paged KV into a
dense tensor) has been removed: it was dead code -- draft layers always
routed to PA/FIA, and non-draft layers never carry
``kv_sharing_target_layer_name``.

The helpers self-gate on ``kv_sharing_target_layer_name is not None``, so they
are generic to any cross-model KV-sharing draft model (Gemma4 MTP is the
first user) and stay no-op for normal layers.  No config-based gate is used:
during draft forward the active config is the draft model's config, which has
no ``speculative_config``, so a config gate would misfire as False (see the
draft-forward-misfire lesson documented across this PR).
"""

from dataclasses import dataclass
from typing import Any

from vllm_ascend.attention.utils import expand_paged_kv_to_per_query


# ---------------------------------------------------------------------------
# Binding registry (typed view over impl attributes)
#
# A draft attention backend impl is bound to its target by three pieces of
# state, written at proposer-init time and read at PA-capture time (A2/A3):
#   * target_impl  : the target layer's attention backend (owning the real KV)
#   * gid          : this layer's KV-cache-group id (for per-group block tables)
#   * bt_ref       : the proposer's {gid: block_table} dict
#
# These are stored as ``_kv_share_*`` attributes on the impl and surfaced
# through ``binding_of`` so callers never touch the attribute names directly.
# On A5 (FIA) these attributes are registered but never read -- the shared KV
# is consumed implicitly via the cache-group alias.
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
# Read-only guards (A5 FIA path + A2/A3 PA path)
#
# Both paths share the same invariant: a KV-sharing draft layer must never
# write its dummy K/V into the target's cache slots.  These two helpers gate
# reshape_and_cache at the two call sites in attention_v1 (forward and
# reshape_and_cache) so the invariant holds regardless of the dispatch path.
# ---------------------------------------------------------------------------


def should_skip_draft_kv_write(impl) -> bool:
    """True for Q-only draft KV-sharing layers.

    Gemma4MTPAttention.forward() creates a dummy K/V via torch.empty() and
    passes it as key/value to self.attn().  Writing this uninitialized
    memory back via reshape_and_cache would corrupt the shared target KV
    cache, causing progressive degradation across sequential loop steps.
    Skip the write for draft KV-shared layers -- they only READ.
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


# ---------------------------------------------------------------------------
# PagedAttention capture / verify entry points (A2/A3 only; retires September)
#
# These are reached only when the PA gate (``not is_950()``) is active, i.e.
# on A2/A3.  A5 (is_950) runs FIA end-to-end and never calls them.
# AscendAttentionBackendImpl delegates the KV-source resolution and the
# MTP-verify per-query expansion here so its capture / decode paths stay free
# of cross-model KV-sharing and spec-decode specifics.
# ---------------------------------------------------------------------------


def resolve_capture_kv(impl, attn_metadata, num_tokens):
    """Resolve the KV sources for a PagedAttention graph-capture step.

    Returns ``(key_cache, value_cache, block_table, context_lens)`` for the
    capture path to feed into the PA workspace / op.  For KV-sharing draft
    layers it swaps to the target layer's cache and routes the per-group
    block table; for MTP verify (SpecDecoding) it additionally expands the
    per-seq block_table / context_lens to per-query so each query token
    attends only up to its own position (no future leak).
    """
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    block_table = attn_metadata.block_tables
    context_lens = attn_metadata.seq_lens
    key_cache = impl.key_cache
    value_cache = impl.value_cache
    if getattr(_EXTRA_CTX, "is_draft_model", False) and getattr(impl, "kv_sharing_target_layer_name", None) is not None:
        binding = binding_of(impl)
        if binding is not None and binding.target_impl is not None and binding.target_impl.key_cache is not None:
            key_cache = binding.target_impl.key_cache
            value_cache = binding.target_impl.value_cache
        if binding is not None and binding.gid is not None and binding.bt_ref is not None and binding.gid in binding.bt_ref:
            block_table = binding.bt_ref[binding.gid]
    if attn_metadata.attn_state == AscendAttentionState.SpecDecoding:
        spec_cfg = impl.vllm_config.speculative_config
        if (
            spec_cfg is not None
            and num_tokens == context_lens.shape[0] * (spec_cfg.num_speculative_tokens + 1)
        ):
            block_table, context_lens = expand_paged_kv_to_per_query(
                block_table, context_lens, spec_cfg.num_speculative_tokens
            )
    return key_cache, value_cache, block_table, context_lens


def maybe_expand_paged_kv_for_verify(impl, attn_metadata, num_tokens):
    """Return ``(block_table, context_lens)`` for a PagedAttention step.

    Expands per-seq block_table / context_lens to per-query for MTP verify
    (SpecDecoding) so token0 does not attend draft1's KV (future leak);
    otherwise returns the common tensors unchanged.
    """
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    block_table = attn_metadata.block_tables
    context_lens = attn_metadata.seq_lens
    if attn_metadata.attn_state == AscendAttentionState.SpecDecoding:
        spec_cfg = impl.vllm_config.speculative_config
        if (
            spec_cfg is not None
            and num_tokens == context_lens.shape[0] * (spec_cfg.num_speculative_tokens + 1)
        ):
            block_table, context_lens = expand_paged_kv_to_per_query(
                block_table, context_lens, spec_cfg.num_speculative_tokens
            )
    return block_table, context_lens
