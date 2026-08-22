# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    LogitsProcessors,
    MoveDirectionality,
    build_logitsprocs,
)

from vllm_ascend.sample.reasoning_phase import (
    ReasoningPhaseStateHolder,
    ReasoningProtocolSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class ReasoningEosLogitsProcessor(LogitsProcessor):
    """Mask model EOS tokens while requests are inside reasoning."""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool) -> None:
        _ = is_pin_memory
        config = vllm_config.reasoning_config
        if config is None:
            raise ValueError("Reasoning EOS policy requires reasoning config")
        start_ids = config.reasoning_start_token_ids
        exit_ids = config.reasoning_exit_token_ids
        if not start_ids or not exit_ids:
            raise ValueError("Reasoning EOS policy requires tokenized phase markers")
        protocol = ReasoningProtocolSpec(
            start_token_ids=tuple(start_ids),
            exit_token_ids=tuple(tuple(ids) for ids in exit_ids),
        )
        self.phase_state = ReasoningPhaseStateHolder(protocol)

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.phase_state.remove_request(index)
        for index, params, prompt_ids, output_ids in batch_update.added:
            self.phase_state.add_request(
                index,
                prompt_ids,
                output_ids,
                params.model_eos_token_ids,
            )
        for source, destination, direction in batch_update.moved:
            self.phase_state.move_request(
                source,
                destination,
                swap=direction == MoveDirectionality.SWAP,
            )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return self._apply_entries(logits, self.phase_state.normal_mask_entries())

    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        draft_token_ids: Sequence[Sequence[int]],
        num_draft_tokens: Sequence[int],
    ) -> torch.Tensor:
        entries = self.phase_state.speculative_mask_entries(draft_token_ids, num_draft_tokens)
        return self._apply_entries(logits, entries)

    def apply_for_bonus(
        self,
        logits: torch.Tensor,
        draft_token_ids: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        entries = self.phase_state.bonus_mask_entries(draft_token_ids)
        return self._apply_entries(logits, entries)

    @staticmethod
    def _apply_entries(
        logits: torch.Tensor,
        entries: Sequence[tuple[int, tuple[int, ...]]],
    ) -> torch.Tensor:
        if not entries:
            return logits

        entry_rows = [row for row, _ in entries]
        rows = async_tensor_h2d(
            [row for row, eos_ids in entries for _ in eos_ids],
            device=logits.device,
        )
        token_ids = async_tensor_h2d(
            [token_id for _, eos_ids in entries for token_id in eos_ids],
            device=logits.device,
        )
        entry_indices = async_tensor_h2d(
            [index for index, (_, eos_ids) in enumerate(entries) for _ in eos_ids],
            device=logits.device,
        )
        previous = logits[rows, token_ids].clone()
        logits[rows, token_ids] = -torch.inf
        affected_rows = async_tensor_h2d(entry_rows, device=logits.device)
        restore_entries = torch.isneginf(logits[affected_rows]).all(dim=1)
        logits[rows, token_ids] = torch.where(restore_entries[entry_indices], previous, -torch.inf)
        return logits


def reasoning_eos_policy_enabled(reasoning_config: object | None) -> bool:
    return (
        reasoning_config is not None
        and getattr(reasoning_config, "premature_eos_policy", "allow") == "mask_in_reasoning"
    )


def build_ascend_logitsprocs(
    vllm_config: "VllmConfig",
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (),
) -> LogitsProcessors:
    processors = list(
        build_logitsprocs(
            vllm_config,
            device,
            is_pin_memory,
            is_pooling_model,
            custom_logitsprocs,
        ).all
    )
    if is_pooling_model or not reasoning_eos_policy_enabled(vllm_config.reasoning_config):
        return LogitsProcessors(processors)

    processors = [
        processor
        for processor in processors
        if not (
            type(processor).__name__ == "ReasoningEosLogitsProcessor"
            and type(processor).__module__.startswith("vllm.v1.sample.logits_processor")
        )
    ]
    processors.append(ReasoningEosLogitsProcessor(vllm_config, device, is_pin_memory))
    return LogitsProcessors(processors)
