# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backport vllm-project/vllm#63492 to the MiniMax-M3 reasoning parser.

Remove when all supported vLLM versions include the upstream fix.
"""

from collections.abc import Iterable, Sequence
from functools import wraps

from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser


def _last_message_ids(self: MiniMaxM3ReasoningParser, input_ids: Sequence[int]) -> Sequence[int]:
    # Every message, including the generation prompt, starts with ]~b].
    # Earlier messages may contain reasoning markers in instructions or history.
    message_begin_token_id = self.vocab.get("]~b]")
    if message_begin_token_id is not None:
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == message_begin_token_id:
                return input_ids[i + 1 :]
    return input_ids


def _patch_minimax_m3_reasoning() -> None:
    # Upstream #63492 introduces this helper alongside both end-check fixes.
    # Also makes repeated application of this backport a no-op.
    if hasattr(MiniMaxM3ReasoningParser, "_last_message_ids"):
        return

    original_is_reasoning_end = MiniMaxM3ReasoningParser.is_reasoning_end
    original_is_reasoning_end_streaming = MiniMaxM3ReasoningParser.is_reasoning_end_streaming

    @wraps(original_is_reasoning_end)
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return original_is_reasoning_end(self, self._last_message_ids(input_ids))

    @wraps(original_is_reasoning_end_streaming)
    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        # Preserve upstream's state fast paths before scanning accumulated IDs.
        if not (self._reasoning_ended_streaming or self._reasoning_active_streaming or self._pending_marker_streaming):
            input_ids = self._last_message_ids(input_ids)
        return original_is_reasoning_end_streaming(self, input_ids, delta_ids)

    MiniMaxM3ReasoningParser._last_message_ids = _last_message_ids
    MiniMaxM3ReasoningParser.is_reasoning_end = is_reasoning_end
    MiniMaxM3ReasoningParser.is_reasoning_end_streaming = is_reasoning_end_streaming


_patch_minimax_m3_reasoning()
