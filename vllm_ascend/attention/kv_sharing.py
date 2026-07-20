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
    *implicitly* -- no per-forward swap is needed.

    reshape_and_cache in attention_v1.py inlines ``self.kv_sharing_target_layer_name
    is not None`` to skip the write for all KV-sharing layers (matching upstream).
    This module contributes the additional draft-specific guard:

      * should_skip_draft_kv_write  -- Q-only draft layers skip reshape
        (checks ``kv_sharing_target_layer_name`` AND ``is_draft_model``)

The A2/A3 PagedAttention capture/verify KV-routing (bind / binding_of /
resolve_capture_kv / maybe_expand_paged_kv_for_verify) has been removed from
this branch: the PR focuses on A5 Gemma4 MTP, and A5 never enters the PA
path (``using_paged_attention`` returns False on A5).  That machinery is
preserved as a patch reference for re-introducing A2/A3 Gemma4 MTP support.

The helper self-gates on ``kv_sharing_target_layer_name is not None``, so it
is generic to any cross-model KV-sharing draft model (Gemma4 MTP is the
first user) and stays no-op for normal layers.  No config-based gate is used:
during draft forward the active config is the draft model's config, which has
no ``speculative_config``, so a config gate would misfire as False (see the
draft-forward-misfire lesson documented across this PR).
"""


# ---------------------------------------------------------------------------
# Read-only guard (A5 FIA path, draft-specific)
#
# Invariant: a KV-sharing draft layer must never write its dummy K/V into the
# target's cache slots.  reshape_and_cache already inlines the general
# ``kv_sharing_target_layer_name is not None`` skip (matching upstream);
# this helper adds the ``is_draft_model`` gate for Q-only draft layers in
# the forward() dispatch path.
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
