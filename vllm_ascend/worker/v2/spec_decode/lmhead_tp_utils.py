# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch

from vllm_ascend.utils import lmhead_tp_enable, lmhead_tp_max_num_logits


class LmheadTPDraftSamplingMixin:
    """Construction-time validation and head-capacity plumbing for draft
    sampling under lmhead TP.

    The row alignment itself lives in the LM-head forward
    (``AscendLogitsProcessor._get_logits_lmheadtp``): every vocab-sharded
    draft head pads its hidden states up to the head-level capacity
    (per-step dynamic value published by the speculator, falling back to
    the static constructor capacity) and trims the logits back. What this
    mixin guards are the sampling shapes that padding cannot align:
    probabilistic sampling writes into fixed-size draft buffers, local
    argmax reduces over the local vocab shard only, and paths that bypass
    the LM head entirely (top-k on the sharded head) never see the pad.
    """

    # Attributes injected by the hosting speculator; annotated for standalone
    # mypy analysis.
    max_num_reqs: int
    num_speculative_steps: int
    speculative_config: Any
    use_local_argmax_reduction: bool
    enable_adaptive_verification: bool

    # Speculators whose draft collectives cannot be fed group-uniform rows
    # opt out here and are rejected at construction. No current family opts
    # out: DFlash2's head call is collective-free and its candidate top-k
    # merge gathers over the lmhead-TP group on uniformly padded draft-batch
    # rows (qwen3_dflash2.compute_candidates); the flag stays as the guard
    # for a future family whose draft collectives cannot be aligned.
    _lmhead_tp_sample_draft_supported = True

    # The DP-agreed request count of the current propose round, stashed by
    # the hosting speculator's propose (from the target's DPSyncState —
    # every field is identical on every rank, and FULL descriptors pad it
    # to the bucket request count). The autoregressive draft syncs are
    # request-scaled, not head-row-scaled: the bootstrap reuses the
    # target's verify sync (num_reqs * (num_speculative_steps + 1) query
    # rows per request) and the continuations rebuild on a one-row basis,
    # while the draft head samples exactly one row per request. The synced
    # token max therefore overshoots those head rows on the bootstrap by
    # the verify basis, so the stashed request count is the tightest
    # group-uniform capacity for one-row-per-request heads in every phase.
    # None = not stashed (DFlash/DSpark sample off the raw synced query
    # rows; replicated drafts sync global tokens): publish the raw synced
    # max exactly as before.
    _lmhead_tp_uniform_num_reqs: int | None = None

    def _lmhead_tp_validate_draft_sampling(self) -> None:
        """Fail unsupported draft sampling at construction, not first use."""
        if not lmhead_tp_enable():
            return
        if not self._lmhead_tp_sample_draft_supported:
            raise NotImplementedError(
                f"lmhead TP does not support {type(self).__name__}: its draft "
                "sampling does not go through sample_draft, so the "
                "group-aligned row padding cannot be applied."
            )
        if self.speculative_config.draft_sample_method == "probabilistic":
            raise NotImplementedError(
                "lmhead TP does not support draft_sample_method='probabilistic': "
                "the gumbel path writes into fixed-size draft buffers that cannot "
                "hold the group-aligned padding rows."
            )
        if self.use_local_argmax_reduction:
            raise NotImplementedError(
                "lmhead TP does not support use_local_argmax_reduction: "
                "get_top_tokens reduces over the local vocab shard only, which "
                "silently produces wrong tokens under pure-DP lmhead TP."
            )
        if self.enable_adaptive_verification:
            raise NotImplementedError(
                "lmhead TP does not support enable_adaptive_verification: the "
                "online acceptance estimator sizes its kernels by the draft "
                "logit row count and indexes its per-token features and "
                "confidence buffers by it, so the group-aligned padding rows "
                "read the request mapping and write those buffers out of bounds."
            )

    def _lmhead_tp_inject_static_capacity(self, model: torch.nn.Module) -> None:
        """Attach the config-derived static row capacity to a draft LM head.

        Generic draft model families (EAGLE, DFlash) construct their LM head
        in shared model files that know nothing about lmhead TP; the
        speculator injects the static bound post-load so the head-level pad
        has a fallback for steps without a DP sync. The coefficient mirrors
        the old sample_draft pad: ``num_speculative_steps + 1`` rows per
        request. The bound is deliberately loose — it only backs steps
        without a DP sync, while the per-step publish feeds the tight
        synced value.
        """
        if not lmhead_tp_enable():
            return
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            raise ValueError(
                f"{type(model).__name__} exposes no lm_head; lmhead TP cannot "
                "attach the draft row capacity. Report this draft model to "
                "the vllm-ascend maintainers."
            )
        lm_head.lmhead_tp_capacity = lmhead_tp_max_num_logits(
            self.max_num_reqs, self.num_speculative_steps + 1
        )

    def _lmhead_tp_publish_draft_capacity(
        self, num_tokens_across_dp: torch.Tensor | None, tokens_per_req: int = 1
    ) -> None:
        """Publish the per-step draft LM-head capacity from the DP sync.

        ``num_tokens_across_dp`` is the group-agreed per-rank query token
        count (upstream ``_build_uniform_batch_dp_sync``): identical on
        every rank, so its max is a group-uniform row bound. When the
        round's DP-agreed request count is stashed
        (``_lmhead_tp_uniform_num_reqs``), heads that sample one row per
        request (EAGLE) convert the synced rows to that basis: the
        autoregressive draft syncs are request-scaled (the bootstrap
        reuses the target's (k+1)-rows-per-request verify sync, the
        continuations rebuild on a one-row basis), so the request count —
        not the raw synced token max — is the tightest group-uniform
        capacity for those heads in every phase. Heads whose sample rows
        the synced query rows already cover (DFlash/DSpark's
        ``num_query_per_req`` rows per request) never stash and keep the
        raw synced max. On the raw path ``tokens_per_req`` sanity-checks
        that the synced rows are a whole number of requests. Called from
        ``_run_model`` before this step's head call — inside the captured
        region for FULL graphs, where the capture-time value matches the
        capture shapes. Without a sync value the previous dynamic value is
        dropped so the static ``lmhead_tp_capacity`` (the deliberately
        loose load-time bound) stays authoritative. Gated on lmhead TP so
        non-lmhead-TP runs do not pay the per-step DP-sync ``.item()``.
        """
        if not lmhead_tp_enable():
            return
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            raise ValueError(
                f"{type(self.model).__name__} exposes no lm_head; lmhead TP "
                "cannot publish the draft row capacity. Report this draft "
                "model to the vllm-ascend maintainers."
            )
        if num_tokens_across_dp is not None and num_tokens_across_dp.numel() > 0:
            max_query_rows = int(num_tokens_across_dp.max().item())
            # Stashed request count: convert the synced query rows to the
            # one-row-per-request head basis (the bootstrap sync overshoots
            # by the (k+1)-token verify basis). The guard ignores a stale
            # stash (a valid request count can never exceed this round's
            # synced rows) so it cannot publish a desynced capacity.
            uniform_reqs = self._lmhead_tp_uniform_num_reqs
            if uniform_reqs is not None and uniform_reqs <= max_query_rows:
                lm_head._lmhead_tp_dynamic_capacity = uniform_reqs * tokens_per_req
                return
            if max_query_rows % tokens_per_req != 0:
                raise ValueError(
                    f"lmhead TP: synced draft query rows ({max_query_rows}) are "
                    f"not divisible by tokens_per_req ({tokens_per_req}); the "
                    "DP-synced capacity would desync the draft-head collectives."
                )
            lm_head._lmhead_tp_dynamic_capacity = max_query_rows
        elif hasattr(lm_head, "_lmhead_tp_dynamic_capacity"):
            # No DP sync this round: drop the previous round's dynamic value
            # so the static lmhead_tp_capacity stays authoritative and a
            # stale capacity cannot be reused.
            del lm_head._lmhead_tp_dynamic_capacity
