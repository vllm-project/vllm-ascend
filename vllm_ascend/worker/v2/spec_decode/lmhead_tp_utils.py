# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

from vllm_ascend.utils import lmhead_tp_enable


class LmheadTPDraftSamplingMixin:
    """Construction-time validation for draft sampling under lmhead TP.

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
