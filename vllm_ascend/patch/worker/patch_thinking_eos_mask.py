# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.
"""Model Runner V2: mask EOS while a request is inside its think block.

V2 companion to ``vllm_ascend/sample/thinking_eos_mask.py`` (see that module
and upstream vllm#55420 for the rationale). Patches the V2 sampler's
``ThinkingBudgetState`` construction so the masked subclass is used; the
stock kernelized budget path is left untouched, and the mask itself runs
as NPU Triton kernels
(``vllm_ascend/ops/triton/v2/sample/thinking_eos_mask.py``):

- a marker-cache kernel incrementally tracks the last reasoning-start /
  natural-or-implicit end position in the committed token history,
  mirroring the upstream budget path, so the Python-side per-request D2H
  scan is gone;
- a per-row kernel masks the configured EOS token ids to ``-inf`` on every
  logits row of in-think requests: plain decode, bonus and spec-decode
  verify rows alike, so with MTP enabled EOS can neither be sampled nor
  drafted-and-accepted mid-think.

Enabled by the same env vars as the V1 path:
``VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS`` (unset keeps stock behavior) and
``VLLM_ASCEND_THINKING_MASK_IMPLICIT_END_TOKEN_IDS``.

Removal plan: drop this patch once upstream lands an engine-side fix for
vllm-project/vllm#55420 (e.g. a merged successor of PR #55562).
"""

import torch
import vllm.v1.worker.gpu.sample.sampler as v2_sampler
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState

from vllm_ascend import envs
from vllm_ascend.ops.triton.v2.sample.thinking_eos_mask import (
    apply_thinking_eos_mask,
)


def _parse_token_ids(raw):
    try:
        return [int(token) for token in (raw or "").split(",") if token.strip()]
    except ValueError as e:
        raise ValueError(
            "VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS / "
            "VLLM_ASCEND_THINKING_MASK_IMPLICIT_END_TOKEN_IDS expect a "
            f"comma-separated list of integer token ids, got {raw!r} ({e}). "
            "Look up the ids with the model tokenizer, e.g. "
            "tokenizer.convert_tokens_to_ids('<|im_end|>')."
        ) from e


class ThinkingEosMaskV2State(ThinkingBudgetState):
    """V2 thinking-budget state with an opt-in EOS mask in think."""

    def __init__(self, req_states, reasoning_config):
        super().__init__(req_states, reasoning_config)
        self._eos_mask_token_ids = _parse_token_ids(envs.VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS)
        self._implicit_end_token_ids = _parse_token_ids(envs.VLLM_ASCEND_THINKING_MASK_IMPLICIT_END_TOKEN_IDS)
        self._mask_enabled = bool(self._eos_mask_token_ids) and self.enabled
        if not self._mask_enabled:
            return

        self._eos_mask_token_ids_gpu = torch.tensor(self._eos_mask_token_ids, dtype=torch.int32, device=self.device)
        # Implicit end markers are single tokens and can be absent; the
        # kernel handles the empty list (zero static-range iterations).
        self._implicit_end_token_ids_gpu = torch.tensor(
            self._implicit_end_token_ids, dtype=torch.int32, device=self.device
        )

        # Marker caches owned by the mask path; separate from the budget
        # path's because the mask tracks every request, not only the
        # requests that set a thinking budget.
        self._mask_cached_last_start = torch.full((self.max_num_reqs,), -1, dtype=torch.int32, device=self.device)
        self._mask_cached_last_end = torch.full((self.max_num_reqs,), -1, dtype=torch.int32, device=self.device)
        self._mask_cached_scan_pos = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
        self._mask_reset_reqs: list[int] = []

    def add_request(self, req_idx, sampling_params):
        budget_used = super().add_request(req_idx, sampling_params)
        if self._mask_enabled:
            # Every request needs the logits-processing path so that
            # apply() can run the mask.
            self._mask_reset_reqs.append(req_idx)
            return True
        return budget_used

    def apply_staged_writes(self):
        super().apply_staged_writes()
        if self._mask_enabled and self._mask_reset_reqs:
            idx = async_tensor_h2d(self._mask_reset_reqs, dtype=torch.int64, device=self.device)
            self._mask_cached_last_start.index_fill_(0, idx, -1)
            self._mask_cached_last_end.index_fill_(0, idx, -1)
            self._mask_cached_scan_pos.index_fill_(0, idx, 0)
            self._mask_reset_reqs.clear()

    def apply(self, logits, ctx):
        # Budget-exhaustion forcing path unchanged; it launches before the
        # mask kernels, so when both configure the same token id the mask
        # wins - matching the V1 holder's ordering (parent forcing first,
        # then the EOS mask).
        super().apply(logits, ctx)
        if not self._mask_enabled:
            return
        apply_thinking_eos_mask(
            logits,
            ctx.idx_mapping,
            ctx.expanded_idx_mapping,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
            self._mask_cached_last_start,
            self._mask_cached_last_end,
            self._mask_cached_scan_pos,
            self.reasoning_start_token_ids,
            self.natural_reasoning_end_token_ids,
            self._implicit_end_token_ids_gpu,
            self._eos_mask_token_ids_gpu,
        )


def _apply_thinking_eos_mask_patch():
    eos_ids = _parse_token_ids(envs.VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS)
    if not eos_ids:
        return
    v2_sampler.ThinkingBudgetState = ThinkingEosMaskV2State


_apply_thinking_eos_mask_patch()
