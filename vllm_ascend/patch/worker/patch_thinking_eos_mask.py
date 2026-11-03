# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.
"""Model Runner V2: mask EOS while a request is inside its think block.

V2 companion to ``vllm_ascend/sample/thinking_eos_mask.py`` (see that module
and upstream vllm#55420 for the rationale). Patches the V2 sampler's
``ThinkingBudgetState`` construction so the masked subclass is used; the
kernelized budget path (``apply_thinking_budget``) is left untouched, and a
Python-side scan mirrors the V1 holder to track ``in_think`` per request and
mask the configured EOS ids on every logits row of in-think requests.

Enabled by the same env vars as the V1 path:
``VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS`` (unset keeps stock behavior) and
``VLLM_ASCEND_THINKING_MASK_IMPLICIT_END_TOKEN_IDS``.

Removal plan: drop this patch once upstream lands an engine-side fix for
vllm-project/vllm#55420 (e.g. a merged successor of PR #55562).
"""

import numpy as np
import torch
import vllm.v1.worker.gpu.sample.sampler as v2_sampler
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState

from vllm_ascend import envs


def _parse_token_ids(raw):
    return [int(token) for token in (raw or "").split(",") if token.strip()]


class ThinkingEosMaskV2State(ThinkingBudgetState):
    """V2 thinking-budget state with an opt-in EOS mask in think."""

    def __init__(self, req_states, reasoning_config):
        super().__init__(req_states, reasoning_config)
        self._eos_mask_token_ids = _parse_token_ids(
            envs.VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS
        )
        self._implicit_end_token_ids = _parse_token_ids(
            envs.VLLM_ASCEND_THINKING_MASK_IMPLICIT_END_TOKEN_IDS
        )
        self._mask_enabled = bool(self._eos_mask_token_ids) and self.enabled
        # Python-side incremental scan state, mirroring the V1 holder:
        # scan_pos is the first committed token index not yet scanned.
        self._scan_pos = np.zeros(self.max_num_reqs, dtype=np.int64)
        self._in_think = np.zeros(self.max_num_reqs, dtype=bool)

    def add_request(self, req_idx, sampling_params):
        budget_used = super().add_request(req_idx, sampling_params)
        if self._mask_enabled:
            # Every request needs the logits-processing path so that
            # apply() can run the mask.
            self._scan_pos[req_idx] = 0
            self._in_think[req_idx] = False
            return True
        return budget_used

    def apply(self, logits, ctx):
        # Budget-exhaustion forcing path unchanged.
        super().apply(logits, ctx)
        if self._mask_enabled:
            self._mask_eos_in_think(logits, ctx)

    def _mask_eos_in_think(self, logits, ctx):
        idx_mapping_np = ctx.idx_mapping_np
        num_rows = logits.shape[0]
        # A row maps to a request via idx_mapping (bonus/decode rows) or via
        # expanded_idx_mapping (spec-decode verify rows); fall back to the
        # plain mapping when the expanded one is absent.
        row_req = getattr(ctx, "expanded_idx_mapping", None)
        if row_req is None or row_req.shape[0] != num_rows:
            row_req = idx_mapping_np
        row_req = row_req[:num_rows]

        rows = []
        for row in range(num_rows):
            req_idx = row_req[row]
            if req_idx < 0:
                continue
            self._refresh_in_think(req_idx)
            if self._in_think[req_idx]:
                rows.append(row)
        if not rows:
            return
        flat_rows = [row for row in rows for _ in self._eos_mask_token_ids]
        flat_tokens = [
            token for _ in rows for token in self._eos_mask_token_ids
        ]
        rows_tensor = async_tensor_h2d(flat_rows, dtype=torch.long, device=logits.device)
        tokens_tensor = async_tensor_h2d(
            flat_tokens, dtype=torch.long, device=logits.device
        )
        fill = logits.new_full((len(flat_rows),), float("-inf"))
        logits.index_put_((rows_tensor, tokens_tensor), fill)

    def _refresh_in_think(self, req_idx):
        """Recompute in_think from the committed token history.

        Mirrors the V1 holder's last-occurrence semantics: in think iff the
        last think-start marker occurs after every think-end / implicit-end
        marker. Rescans the full committed tail each step; correct and simple
        at the cost of a per-step O(len) scan.
        """
        total_len = int(self.req_states.total_len.np[req_idx])
        tokens = self.req_states.all_token_ids.np[req_idx][:total_len].tolist()
        last_start = self._rfind_marker(tokens, self.reasoning_start_token_ids.tolist())
        last_end = max(
            self._rfind_marker(tokens, self.natural_reasoning_end_token_ids.tolist()),
            self._rfind_marker(tokens, self._implicit_end_token_ids),
        )
        self._in_think[req_idx] = last_start > last_end

    @staticmethod
    def _rfind_marker(tokens, marker_ids):
        if not marker_ids:
            return -1
        return max(
            (pos for pos, token in enumerate(tokens) if token in marker_ids),
            default=-1,
        )


def _apply_thinking_eos_mask_patch():
    eos_ids = _parse_token_ids(envs.VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS)
    if not eos_ids:
        return
    v2_sampler.ThinkingBudgetState = ThinkingEosMaskV2State


_apply_thinking_eos_mask_patch()
