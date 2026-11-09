# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V1 compatibility for upstream thinking-budget fixes.

https://github.com/vllm-project/vllm/pull/61005
https://github.com/vllm-project/vllm/pull/62970

Remove this subclass and restore the upstream factory in NPUInputBatch once
both fixes are included in all supported vLLM versions. V2 uses its own state.
"""

from typing import Any

import torch
from vllm.config.reasoning import ReasoningConfig
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


class NPUThinkingBudgetStateHolder(ThinkingBudgetStateHolder):
    def _committed_end_prefix(self, output: list[int]) -> int:
        end_ids = self.think_end_token_ids
        for n in range(min(len(end_ids) - 1, len(output)), 0, -1):
            if output[-n:] == end_ids[:n]:
                return n
        return 0

    def _force_end_window(self, state: dict[str, Any], progress: int, start: int = 0) -> None:
        window = len(state.get("spec_token_ids", [])) + 1
        n = max(0, min(len(self.think_end_token_ids) - progress, window - start))
        state["forced_end_start"] = len(state.get("output_tok_ids", [])) + start - progress
        state["force_index"] = list(range(start, start + n))
        state["force_tokens"] = self.think_end_token_ids[progress : progress + n]

    def _update_think_state(self, state: dict[str, Any]) -> None:
        state["force_tokens"] = []
        output = state.get("output_tok_ids", [])
        prev_length = state.get("prev_output_length", 0)
        end_ids = self.think_end_token_ids
        # Continue a forced end only from committed tokens, never from drafts.
        # Otherwise retain upstream counting, natural EOS and rejection recovery.
        forced_start = state.get("forced_end_start", len(output))
        committed = output[forced_start : forced_start + len(end_ids)]
        if end_ids and state.get("in_end", False) and committed and committed == end_ids[: len(committed)]:
            recent = committed
            state["prev_output_length"] = len(output)
            if self._find_last_sequence_index(recent, end_ids) >= 0:
                state.update(
                    in_end=False,
                    end_count=0,
                    force_index=[],
                    check_count_down=state["thinking_token_budget"],
                    start_thinking=-1,
                    end_thinking=-1,
                    think_count=0,
                    continue_thinking=False,
                    scan_offset=len(output),
                )
            else:
                state["end_count"] = self._committed_end_prefix(output)
                self._force_end_window(state, state["end_count"])
            return

        if state.get("in_end", False) and len(output) > prev_length:
            # Rejection happened before the scheduled forced boundary. Reset
            # speculative end mode before upstream counts the committed tokens;
            # a matching token elsewhere in the suffix is ordinary output.
            state.update(
                in_think=True,
                in_end=False,
                end_count=0,
                bonus_token_forced=False,
                check_count_down=max(0, state["thinking_token_budget"] - state["think_count"]),
            )

        super()._update_think_state(state)
        if state.get("in_end", False) and state.get("force_index"):
            # A resumed request can already contain a committed forced prefix.
            # Only resume it if the preceding thinking tokens used the budget;
            # an incidental matching suffix within the budget is ordinary text.
            progress = self._committed_end_prefix(output)
            if progress and state["think_count"] - progress >= state["thinking_token_budget"]:
                state["end_count"] = progress
                self._force_end_window(state, progress)
            else:
                self._force_end_window(state, 0, state["force_index"][0])

    def _apply_forcing_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids_for_layout: list[list[int]],
    ) -> torch.Tensor:
        # A draftless step uses the ordinary sampler: one row per request.
        use_spec_layout = self.in_spec_mode and not predict_bonus_token and any(spec_token_ids_for_layout)
        self.cu_num_tokens.clear()
        n_layout = max(len(spec_token_ids_for_layout), max(self._state, default=-1) + 1)
        offset = 0
        for index in range(n_layout):
            self.cu_num_tokens[index] = offset
            drafts = spec_token_ids_for_layout[index] if index < len(spec_token_ids_for_layout) else []
            offset += len(drafts) if use_spec_layout else 1

        rows: list[int] = []
        tokens: list[int] = []
        for index, state in self._state.items():
            if not state.get("in_end", False):
                continue
            num_drafts = len(state["spec_token_ids"])
            for pos, token in zip(state["force_index"], state.get("force_tokens", [])):
                if use_spec_layout:
                    if pos >= num_drafts:
                        continue
                    row = self.cu_num_tokens[index] + pos
                elif pos == num_drafts:
                    row = self.cu_num_tokens[index]
                else:
                    continue
                assert row < min(self._mask_capacity, logits.shape[0]), (
                    f"thinking-budget forced end row {row} is outside the logits "
                    f"({logits.shape[0]} rows, capacity {self._mask_capacity})"
                )
                rows.append(row)
                tokens.append(token)

        # Same batched H2D/index_put strategy as upstream, with no device reads.
        if rows:
            indices = async_tensor_h2d(rows, dtype=torch.long, device=logits.device)
            token_ids = async_tensor_h2d(tokens, dtype=torch.long, device=logits.device)
            logits.index_put_((indices, token_ids), logits.new_full((len(rows),), 1e9))
        return logits


def maybe_create_thinking_budget_state_holder(
    reasoning_config: ReasoningConfig | None,
    max_num_seqs: int,
    num_spec_tokens: int,
    device: torch.device,
    is_pin_memory: bool,
) -> NPUThinkingBudgetStateHolder | None:
    if reasoning_config is None:
        return None
    return NPUThinkingBudgetStateHolder(reasoning_config, max_num_seqs, num_spec_tokens, device, is_pin_memory)
