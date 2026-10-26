# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Keep vLLM v0.30 MiniMax parsing aligned with adaptive M3.1 templates.

Remove once supported vLLM releases include both template and prompt fixes.
"""

from functools import wraps

from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser


def _patch_minimax_m31_reasoning() -> None:
    original_init = MiniMaxM3ReasoningParser.__init__
    if getattr(original_init, "_vllm_ascend_minimax_m31", False):
        return
    original_is_reasoning_end = MiniMaxM3ReasoningParser.is_reasoning_end

    @wraps(original_init)
    def patched_init(self, tokenizer, *args, **kwargs) -> None:
        original_init(self, tokenizer, *args, **kwargs)
        self._assistant_prefix_ids = self._encode_marker("]~b]ai\n")
        template = getattr(tokenizer, "chat_template", None)
        if isinstance(template, str) and "<effort>" in template and "thinking_mode" not in template:
            # The adaptive-only template never prefills the thinking marker.
            self._initial_in_reasoning = False
            self._reasoning_active_streaming = False

    @wraps(original_is_reasoning_end)
    def patched_is_reasoning_end(self, input_ids) -> bool:
        prefix_ids = self._assistant_prefix_ids
        if prefix_ids and tuple(input_ids[-len(prefix_ids) :]) == prefix_ids:
            # Instructions and historical closers do not close this new turn.
            return False
        return original_is_reasoning_end(self, input_ids)

    patched_init._vllm_ascend_minimax_m31 = True  # type: ignore[attr-defined]
    MiniMaxM3ReasoningParser.__init__ = patched_init
    MiniMaxM3ReasoningParser.is_reasoning_end = patched_is_reasoning_end


_patch_minimax_m31_reasoning()
