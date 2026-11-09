# SPDX-License-Identifier: Apache-2.0
"""Opt-in EOS mask while a request is inside its thinking block.

Qwen3.8-class models occasionally sample their end-of-turn token inside
the think block: generation stops with the whole answer parked in
``reasoning_content`` and ``content`` left empty, which agent harnesses
(e.g. tau2) read as an empty reply. Upstream tracks this as
vllm-project/vllm#55420 with no engine-side fix merged yet
(``reasoning_eos_policy`` PR #55562 was closed unmerged); this module
carries the mask sketched in the #55420 discussion as a
deployment-local opt-in.

The holder extends the stock ``ThinkingBudgetStateHolder``:

- requests are tracked even without a ``thinking_token_budget`` (a huge
  sentinel budget keeps the inherited force-end machinery dormant);
- while a tracked request is inside its think block, the configured EOS
  token ids are masked to ``-inf`` on every sampling row: plain decode
  rows, the bonus row and the spec-decode verify rows alike, so with
  speculative decoding enabled EOS can neither be sampled nor
  drafted-and-accepted mid-think;
- token ids that implicitly leave the think block (e.g. the Qwen3
  tool-call start token, for a tool call issued straight from
  thinking) can be declared so the masked phase ends normally.

Enabled by ``VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS`` (comma-separated
token ids; unset keeps stock behavior). Implicit end ids come from
``VLLM_ASCEND_THINKING_MASK_IMPLICIT_END_TOKEN_IDS``; without them a
tool-calling turn would keep EOS masked until ``max_tokens``.
"""

import torch
from vllm.logger import logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.thinking_budget_state import (
    ThinkingBudgetStateHolder,
    maybe_create_thinking_budget_state_holder,
)

from vllm_ascend import envs

# Sentinel budget for mask-only tracking: large enough that the inherited
# countdown can never reach zero, so only the EOS mask applies and the
# force-end machinery stays dormant.
_MASK_ONLY_THINKING_BUDGET = 1 << 30


def _parse_token_ids(raw: str) -> list[int]:
    try:
        return [int(token) for token in raw.split(",") if token.strip()]
    except ValueError as e:
        raise ValueError(
            "VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS / "
            "VLLM_ASCEND_THINKING_MASK_IMPLICIT_END_TOKEN_IDS expect a "
            f"comma-separated list of integer token ids, got {raw!r} ({e}). "
            "Look up the ids with the model tokenizer, e.g. "
            "tokenizer.convert_tokens_to_ids('<|im_end|>')."
        ) from e


class ThinkingEosMaskStateHolder(ThinkingBudgetStateHolder):
    """Masks EOS ids on every sampling row while a request is in think."""

    def __init__(
        self,
        reasoning_config,
        max_num_seqs: int,
        num_spec_tokens: int,
        device: torch.device,
        is_pin_memory: bool,
        eos_mask_token_ids: list[int],
        implicit_end_token_ids: list[int],
    ):
        super().__init__(
            reasoning_config,
            max_num_seqs,
            num_spec_tokens,
            device,
            is_pin_memory,
        )
        self.eos_mask_token_ids = eos_mask_token_ids
        self.implicit_end_token_ids = implicit_end_token_ids
        self._mask_enabled = bool(eos_mask_token_ids)

    def sync_batch(self, batch_update: BatchUpdate | None) -> None:
        """Track every request while the mask is enabled.

        Differs from the parent only in the ``added`` handling: requests
        without a ``thinking_token_budget`` are tracked too, with a
        sentinel budget so the inherited force-end machinery never fires.
        """
        if not self._mask_enabled:
            return super().sync_batch(batch_update)
        if not self.is_enabled or not batch_update:
            return
        for index in batch_update.removed:
            self._state.pop(index, None)

        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            thinking_token_budget = params.thinking_token_budget
            if thinking_token_budget is None:
                thinking_token_budget = _MASK_ONLY_THINKING_BUDGET
            self._state[index] = self._init_state_entry(prompt_tok_ids, thinking_token_budget)
            self._state[index]["output_tok_ids"] = output_tok_ids
            self._state[index]["spec_token_ids"] = []

        for i1, i2, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                state1 = self._state.pop(i1, None)
                state2 = self._state.pop(i2, None)
                if state1 is not None:
                    self._state[i2] = state1
                if state2 is not None:
                    self._state[i1] = state2
            else:
                state = self._state.pop(i1, None)
                if state is not None:
                    self._state[i2] = state

    def _update_think_state(self, state: dict) -> None:
        super()._update_think_state(state)
        if not self._mask_enabled or not self.implicit_end_token_ids:
            return
        if not state.get("in_think", False) or state.get("in_end", False):
            return
        outputs = state.get("output_tok_ids") or []
        scan_from = state.get("scan_offset", 0)
        window = outputs[scan_from:]
        if not any(token in window for token in self.implicit_end_token_ids):
            return
        # An implicit end marker (e.g. a tool call started straight from
        # thinking) leaves the think block: mirror the parent's natural
        # think-end reset so the EOS mask stops applying from now on.
        state["in_think"] = False
        state["think_count"] = 0
        state["continue_thinking"] = False
        state["start_thinking"] = -1
        state["end_thinking"] = -1
        state["scan_offset"] = len(outputs)
        state["check_count_down"] = state["thinking_token_budget"]

    def apply_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids: list[list[int]] | None,
    ) -> torch.Tensor:
        logits = super().apply_to_logits(logits, predict_bonus_token, spec_token_ids)
        if not self._mask_enabled or not self._state:
            return logits
        # Row layout identical to the parent's _apply_forcing_to_logits:
        # spec verify contributes one row per draft, everything else one
        # row per request.
        spec_lists = spec_token_ids or []
        n_layout = max(len(spec_lists), max(self._state.keys()) + 1)

        row_indices: list[int] = []
        cumulative_total = 0
        for index in range(n_layout):
            spec_tokens = spec_lists[index] if index < len(spec_lists) else []
            if self.in_spec_mode:
                count = len(spec_tokens) if not predict_bonus_token else 1
            else:
                count = 1
            state = self._state.get(index)
            if state is not None and state.get("in_think", False) and not state.get("in_end", False):
                row_indices.extend(
                    row for row in range(cumulative_total, cumulative_total + count) if row < logits.shape[0]
                )
            cumulative_total += count

        if not row_indices:
            return logits
        # Build the (row, token) pairs on CPU and upload once, mirroring
        # the parent's forcing path (no per-iteration H2D syncs).
        flat_rows: list[int] = []
        flat_tokens: list[int] = []
        for row in row_indices:
            for token in self.eos_mask_token_ids:
                flat_rows.append(row)
                flat_tokens.append(token)
        rows_tensor = async_tensor_h2d(flat_rows, dtype=torch.long, device=logits.device)
        tokens_tensor = async_tensor_h2d(flat_tokens, dtype=torch.long, device=logits.device)
        fill = logits.new_full((len(flat_rows),), float("-inf"))
        logits.index_put_((rows_tensor, tokens_tensor), fill)
        return logits


def maybe_create_thinking_eos_mask_holder(
    reasoning_config,
    max_num_seqs: int,
    num_spec_tokens: int,
    device: torch.device,
    is_pin_memory: bool,
) -> ThinkingBudgetStateHolder | None:
    """Create the masked holder when configured, the stock one otherwise."""
    eos_mask_token_ids = _parse_token_ids(envs.VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS or "")
    if not eos_mask_token_ids:
        return maybe_create_thinking_budget_state_holder(
            reasoning_config, max_num_seqs, num_spec_tokens, device, is_pin_memory
        )
    implicit_end_token_ids = _parse_token_ids(envs.VLLM_ASCEND_THINKING_MASK_IMPLICIT_END_TOKEN_IDS or "")
    logger.info_once(
        "Thinking EOS mask enabled: eos_token_ids=%s, implicit_end_token_ids=%s",
        eos_mask_token_ids,
        implicit_end_token_ids,
    )
    return ThinkingEosMaskStateHolder(
        reasoning_config,
        max_num_seqs,
        num_spec_tokens,
        device,
        is_pin_memory,
        eos_mask_token_ids,
        implicit_end_token_ids,
    )
