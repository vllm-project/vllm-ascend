# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningProtocolSpec:
    """Token sequences that enter and leave a reasoning phase."""

    start_token_ids: tuple[int, ...]
    exit_token_ids: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.start_token_ids or not self.exit_token_ids:
            raise ValueError("Reasoning phase markers must not be empty")
        if any(not token_ids for token_ids in self.exit_token_ids):
            raise ValueError("Reasoning exit markers must not be empty")

    @property
    def max_marker_len(self) -> int:
        return max(
            len(self.start_token_ids),
            *(len(token_ids) for token_ids in self.exit_token_ids),
        )


class ReasoningPhaseTracker:
    """Incrementally track whether a token stream is inside reasoning."""

    def __init__(
        self,
        protocol: ReasoningProtocolSpec,
        token_ids: Sequence[int] = (),
    ) -> None:
        self.protocol = protocol
        self.in_reasoning = False
        self._suffix: list[int] = []
        self.extend(token_ids)

    def clone(self) -> "ReasoningPhaseTracker":
        tracker = ReasoningPhaseTracker(self.protocol)
        tracker.in_reasoning = self.in_reasoning
        tracker._suffix = self._suffix.copy()
        return tracker

    def extend(self, token_ids: Iterable[int]) -> None:
        for token_id in token_ids:
            self.consume(token_id)

    def consume(self, token_id: int) -> None:
        self._suffix.append(token_id)
        max_marker_len = self.protocol.max_marker_len
        if len(self._suffix) > max_marker_len:
            del self._suffix[:-max_marker_len]

        if self.in_reasoning:
            if any(self._endswith(ids) for ids in self.protocol.exit_token_ids):
                self.in_reasoning = False
        elif self._endswith(self.protocol.start_token_ids):
            self.in_reasoning = True

    def _endswith(self, token_ids: tuple[int, ...]) -> bool:
        marker_len = len(token_ids)
        return self._suffix[-marker_len:] == list(token_ids)


@dataclass
class _RequestPhaseState:
    tracker: ReasoningPhaseTracker
    prompt_token_ids: tuple[int, ...]
    output_token_ids: Sequence[int]
    num_consumed_output_tokens: int
    model_eos_token_ids: tuple[int, ...]


class ReasoningPhaseStateHolder:
    """Maintain committed reasoning phases for a persistent request batch."""

    def __init__(self, protocol: ReasoningProtocolSpec) -> None:
        self.protocol = protocol
        self._states: dict[int, _RequestPhaseState] = {}

    def add_request(
        self,
        index: int,
        prompt_token_ids: Sequence[int] | None,
        output_token_ids: Sequence[int],
        model_eos_token_ids: Iterable[int],
    ) -> None:
        eos_ids = tuple(sorted(model_eos_token_ids))
        if not eos_ids:
            self.remove_request(index)
            return
        prompt = tuple(prompt_token_ids or ())
        tracker = self._tracker_from_prompt(prompt)
        confirmed_output_len = self._confirmed_output_len(output_token_ids)
        tracker.extend(output_token_ids[:confirmed_output_len])
        self._states[index] = _RequestPhaseState(
            tracker=tracker,
            prompt_token_ids=prompt,
            output_token_ids=output_token_ids,
            num_consumed_output_tokens=confirmed_output_len,
            model_eos_token_ids=eos_ids,
        )

    def remove_request(self, index: int) -> None:
        self._states.pop(index, None)

    def move_request(self, source: int, destination: int, swap: bool) -> None:
        source_state = self._states.pop(source, None)
        destination_state = self._states.pop(destination, None)
        if source_state is not None:
            self._states[destination] = source_state
        if swap and destination_state is not None:
            self._states[source] = destination_state

    def sync_committed_outputs(self) -> None:
        for state in self._states.values():
            output_len = self._confirmed_output_len(state.output_token_ids)
            if output_len < state.num_consumed_output_tokens:
                state.tracker = self._tracker_from_prompt(state.prompt_token_ids)
                state.tracker.extend(state.output_token_ids[:output_len])
            else:
                state.tracker.extend(state.output_token_ids[state.num_consumed_output_tokens : output_len])
            state.num_consumed_output_tokens = output_len

    def normal_mask_entries(self) -> list[tuple[int, tuple[int, ...]]]:
        self.sync_committed_outputs()
        return [
            (index, state.model_eos_token_ids) for index, state in self._states.items() if state.tracker.in_reasoning
        ]

    def speculative_mask_entries(
        self,
        draft_token_ids: Sequence[Sequence[int]],
        num_draft_tokens: Sequence[int],
    ) -> list[tuple[int, tuple[int, ...]]]:
        self.sync_committed_outputs()
        entries: list[tuple[int, tuple[int, ...]]] = []
        row = 0
        for index, num_tokens in enumerate(num_draft_tokens):
            state = self._states.get(index)
            if state is None:
                row += num_tokens
                continue
            tracker = state.tracker.clone()
            drafts = draft_token_ids[index]
            for position in range(num_tokens):
                if tracker.in_reasoning:
                    entries.append((row + position, state.model_eos_token_ids))
                tracker.consume(drafts[position])
            row += num_tokens
        return entries

    def bonus_mask_entries(
        self,
        draft_token_ids: Sequence[Sequence[int]],
    ) -> list[tuple[int, tuple[int, ...]]]:
        self.sync_committed_outputs()
        entries: list[tuple[int, tuple[int, ...]]] = []
        for index, state in self._states.items():
            tracker = state.tracker.clone()
            if index < len(draft_token_ids):
                tracker.extend(draft_token_ids[index])
            if tracker.in_reasoning:
                entries.append((index, state.model_eos_token_ids))
        return entries

    def _tracker_from_prompt(self, prompt_token_ids: Sequence[int]) -> ReasoningPhaseTracker:
        prompt_suffix = prompt_token_ids[-self.protocol.max_marker_len :]
        return ReasoningPhaseTracker(self.protocol, prompt_suffix)

    @staticmethod
    def _confirmed_output_len(output_token_ids: Sequence[int]) -> int:
        try:
            return output_token_ids.index(-1)  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            for index, token_id in enumerate(output_token_ids):
                if token_id == -1:
                    return index
            return len(output_token_ids)
