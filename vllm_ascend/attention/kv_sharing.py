# SPDX-License-Identifier: Apache-2.0
"""Cross-model KV sharing for Ascend attention backends.

Implements the upstream ``kv_sharing_target_layer_name`` contract: a draft
attention layer that reads K/V from a target model layer's paged KV cache
instead of owning its own.  Upstream vLLM serves this via the TRITON_ATTN
backend; Ascend has no such backend, so the consumption is implemented here.

A5 (is_950) -- the supported path:
    The draft layer and its target layer share a KV-cache group, hence the
    same physical cache tensors and the same per-request block table.  The
    FIA path (forward_fused_infer_attention) therefore reads the shared KV
    *implicitly* -- no per-forward swap is needed.  This module only
    contributes the read-only guards so the draft never writes dummy K/V
    back into the shared slots:
      * should_skip_draft_kv_write       -- Q-only draft layers skip reshape
      * maybe_skip_reshape_for_kv_share  -- graph-replay/capture second gate

The A2/A3 PagedAttention capture/verify KV-routing (bind / binding_of /
resolve_capture_kv / maybe_expand_paged_kv_for_verify) has been removed from
this branch: the PR focuses on A5 Gemma4 MTP, and A5 does not capture (the
PA gate ``not is_950()`` never fires).  That machinery is preserved as a
patch reference for re-introducing A2/A3 Gemma4 MTP support; see commit
``c03b2713`` for the original implementation.

The helpers self-gate on ``kv_sharing_target_layer_name is not None``, so they
are generic to any cross-model KV-sharing draft model (Gemma4 MTP is the
first user) and stay no-op for normal layers.  No config-based gate is used:
during draft forward the active config is the draft model's config, which has
no ``speculative_config``, so a config gate would misfire as False (see the
draft-forward-misfire lesson documented across this PR).
"""


# ---------------------------------------------------------------------------
# Read-only guards (A5 FIA path)
#
# Invariant: a KV-sharing draft layer must never write its dummy K/V into the
# target's cache slots.  These two helpers gate reshape_and_cache at the two
# call sites in attention_v1 (forward and reshape_and_cache) so the invariant
# holds regardless of the dispatch path.
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
