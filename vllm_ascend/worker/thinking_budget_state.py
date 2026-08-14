# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend-local thinking budget state extensions."""

from typing import TYPE_CHECKING, Any

import torch
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig
    from vllm.sampling_params import SamplingParams


PREMATURE_EOS_POLICY_ALLOW = "allow"
PREMATURE_EOS_POLICY_MASK_IN_REASONING = "mask_in_reasoning"
PREMATURE_EOS_POLICIES = {
    PREMATURE_EOS_POLICY_ALLOW,
    PREMATURE_EOS_POLICY_MASK_IN_REASONING,
}

DSV4_EOS_TOKEN_ID = 1
DSV4_THINK_START_TOKEN_ID = 128821
DSV4_THINK_END_TOKEN_ID = 128822


def maybe_create_ascend_thinking_budget_state_holder(
    reasoning_config: "ReasoningConfig | None",
    max_num_seqs: int,
    num_spec_tokens: int,
    device: torch.device,
    is_pin_memory: bool,
    premature_eos_policy: str = PREMATURE_EOS_POLICY_ALLOW,
) -> "AscendThinkingBudgetStateHolder | None":
    if reasoning_config is None:
        return None
    return AscendThinkingBudgetStateHolder(
        reasoning_config,
        max_num_seqs,
        num_spec_tokens,
        device,
        is_pin_memory,
        premature_eos_policy=premature_eos_policy,
    )


class AscendThinkingBudgetStateHolder(ThinkingBudgetStateHolder):
    """Adds DeepSeek V4 premature-EOS masking on top of vLLM thinking budget."""

    def __init__(
        self,
        reasoning_config: "ReasoningConfig | None",
        max_num_seqs: int,
        num_spec_tokens: int,
        device: torch.device,
        is_pin_memory: bool,
        premature_eos_policy: str = PREMATURE_EOS_POLICY_ALLOW,
    ):
        if premature_eos_policy not in PREMATURE_EOS_POLICIES:
            raise ValueError(
                f"premature_eos_policy must be one of {sorted(PREMATURE_EOS_POLICIES)}, got {premature_eos_policy!r}."
            )

        super().__init__(reasoning_config, max_num_seqs, num_spec_tokens, device, is_pin_memory)
        self._mask_premature_eos = (
            premature_eos_policy == PREMATURE_EOS_POLICY_MASK_IN_REASONING
            and self.think_start_token_ids == [DSV4_THINK_START_TOKEN_ID]
            and self.think_end_token_ids == [DSV4_THINK_END_TOKEN_ID]
        )

    def sync_batch(self, batch_update: BatchUpdate | None) -> None:
        """Add EOS-only tracked rows after upstream thinking-budget sync."""
        super().sync_batch(batch_update)
        if not self.is_enabled or not batch_update:
            return

        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            eos_token_id = self._get_eos_token_id(params)
            if not self._mask_premature_eos or eos_token_id is None:
                if index in self._state:
                    self._state[index]["eos_token_id"] = None
                continue
            if index not in self._state:
                self._state[index] = self._init_state_entry(prompt_tok_ids, -1)
            self._state[index]["eos_token_id"] = eos_token_id
            self._state[index]["output_tok_ids"] = output_tok_ids
            self._state[index]["spec_token_ids"] = []
            self._init_eos_mask_state(self._state[index])

    def update_state(
        self,
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None,
        repeat_indices: torch.Tensor | None = None,
    ) -> None:
        super().update_state(output_token_ids, spec_token_ids, repeat_indices)
        if not self.is_enabled or not self._state:
            return
        for state in self._state.values():
            self._update_eos_mask_state(state)

    def apply_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids: list[list[int]] | None,
    ) -> torch.Tensor:
        logits = super().apply_to_logits(logits, predict_bonus_token, spec_token_ids)
        if not self.is_enabled or not self._state or not self._mask_premature_eos:
            return logits

        spec_lists = spec_token_ids or []
        eos_mask_indices_cpu: list[int] = []
        eos_tokens_cpu: list[int] = []
        for seq_idx in sorted(self._state.keys()):
            if seq_idx not in self.cu_num_tokens:
                continue
            state = self._state[seq_idx]
            self._update_eos_mask_state(state)
            spec_tokens = spec_lists[seq_idx] if seq_idx < len(spec_lists) else []
            self._append_eos_mask_entries(
                eos_mask_indices_cpu,
                eos_tokens_cpu,
                seq_idx,
                state,
                spec_tokens,
                predict_bonus_token,
                logits.shape[0],
            )

        self._set_logits_values(logits, eos_mask_indices_cpu, eos_tokens_cpu, -float("inf"))
        return logits

    @staticmethod
    def _get_eos_token_id(params: "SamplingParams") -> int | None:
        if params.ignore_eos or params.eos_token_id != DSV4_EOS_TOKEN_ID:
            return None
        return params.eos_token_id

    @staticmethod
    def _consume_reasoning_token(in_reasoning: bool, token_id: int) -> bool:
        if token_id == DSV4_THINK_START_TOKEN_ID:
            return True
        if token_id == DSV4_THINK_END_TOKEN_ID:
            return False
        return in_reasoning

    def _advance_reasoning_state(self, in_reasoning: bool, token_ids: list[int]) -> bool:
        for token_id in token_ids:
            in_reasoning = self._consume_reasoning_token(in_reasoning, token_id)
        return in_reasoning

    def _init_eos_mask_state(self, state: dict[str, Any]) -> None:
        prompt_tok_ids = state.get("prompt_tok_ids") or []
        output_tok_ids = state.get("output_tok_ids") or []
        in_reasoning = self._advance_reasoning_state(False, prompt_tok_ids)
        state["in_reasoning"] = self._advance_reasoning_state(in_reasoning, output_tok_ids)
        state["consumed_output_len"] = len(output_tok_ids)

    def _update_eos_mask_state(self, state: dict[str, Any]) -> None:
        if not self._mask_premature_eos or state.get("eos_token_id") is None:
            state["in_reasoning"] = False
            state["consumed_output_len"] = len(state.get("output_tok_ids") or [])
            return
        output_tok_ids = state.get("output_tok_ids") or []
        consumed_output_len = state.get("consumed_output_len", 0)
        if consumed_output_len > len(output_tok_ids):
            self._init_eos_mask_state(state)
            return
        state["in_reasoning"] = self._advance_reasoning_state(
            state.get("in_reasoning", False),
            output_tok_ids[consumed_output_len:],
        )
        state["consumed_output_len"] = len(output_tok_ids)

    def _append_eos_mask_entries(
        self,
        rows: list[int],
        tokens: list[int],
        seq_idx: int,
        state: dict[str, Any],
        spec_tokens: list[int],
        predict_bonus_token: bool,
        num_logits_rows: int,
    ) -> None:
        base_row = self.cu_num_tokens[seq_idx]
        eos_token_id = state.get("eos_token_id")
        if eos_token_id is None:
            return

        def append_row(row: int) -> None:
            if row >= num_logits_rows:
                return
            rows.append(row)
            tokens.append(eos_token_id)

        if not self.in_spec_mode:
            if state.get("in_reasoning", False):
                append_row(base_row)
            return

        in_reasoning = state.get("in_reasoning", False)
        if predict_bonus_token:
            if self._advance_reasoning_state(in_reasoning, spec_tokens):
                append_row(base_row)
            return

        for draft_idx, token_id in enumerate(spec_tokens):
            if in_reasoning:
                append_row(base_row + draft_idx)
            in_reasoning = self._consume_reasoning_token(in_reasoning, token_id)

    def _set_logits_values(
        self,
        logits: torch.Tensor,
        rows_cpu: list[int],
        tokens_cpu: list[int],
        value: float,
    ) -> None:
        if not rows_cpu:
            return
        device = logits.device
        tensor_h2d = torch.tensor if device.type == "cpu" else async_tensor_h2d
        rows = tensor_h2d(rows_cpu, dtype=torch.long, device=device)
        tokens = tensor_h2d(tokens_cpu, dtype=torch.long, device=device)
        fill = logits.new_full((len(rows_cpu),), value)
        logits.index_put_((rows, tokens), fill)
