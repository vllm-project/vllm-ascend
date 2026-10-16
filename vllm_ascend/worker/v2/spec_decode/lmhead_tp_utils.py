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

    # Speculators whose sampling bypasses sample_draft cannot be row-aligned;
    # they opt out and are rejected at construction (DFlash2:
    # get_top_k_tokens on the sharded head).
    _lmhead_tp_sample_draft_supported = True

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
        request.
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
        every rank, so its max is the group-agreed request bound. The head
        emits ``num_speculative_steps + 1`` rows per request (the mixin-era
        coefficient), so the per-step capacity every rank must feed is
        ``(max query rows // tokens_per_req) * (num_speculative_steps + 1)``.
        Called from ``_run_model`` before this step's head call — inside
        the captured region for FULL graphs, where the capture-time value
        matches the capture shapes. Without a sync value the previous
        dynamic value is dropped so the static ``lmhead_tp_capacity``
        stays authoritative. Gated on lmhead TP so non-lmhead-TP runs do
        not pay the per-step DP-sync ``.item()``.
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
            if max_query_rows % tokens_per_req != 0:
                raise ValueError(
                    f"lmhead TP: synced draft query rows ({max_query_rows}) are "
                    f"not divisible by tokens_per_req ({tokens_per_req}); the "
                    "DP-synced capacity would desync the draft-head collectives."
                )
            group_max_reqs = max_query_rows // tokens_per_req
            lm_head._lmhead_tp_dynamic_capacity = group_max_reqs * (self.num_speculative_steps + 1)
        elif hasattr(lm_head, "_lmhead_tp_dynamic_capacity"):
            # No DP sync this round: drop the previous round's dynamic value
            # so the static lmhead_tp_capacity stays authoritative and a
            # stale capacity cannot be reused.
            del lm_head._lmhead_tp_dynamic_capacity
