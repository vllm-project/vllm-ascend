# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Temporary Kimi backport of vllm-project/vllm#68112.

Remove once every supported vLLM version includes that fix. Keep unrelated
parsers and implicit/content-token boundaries on their upstream implementation.
"""

from collections.abc import Sequence
from functools import wraps

import vllm.v1.structured_output as structured_output
from vllm.reasoning.kimi_k3_reasoning_parser import KimiK3ReasoningParser


class _TokenView(Sequence[int]):
    """A bounded history + draft view, including on vLLM without TokenBuffer."""

    def __init__(self, tokens: Sequence[int], suffix: Sequence[int]):
        self.tokens = tokens
        self.suffix = suffix
        self.length = len(tokens) + len(suffix)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        split = len(self.tokens)
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            if step == 1:
                result = list(self.tokens[start : min(stop, split)])
                if stop > split:
                    result.extend(self.suffix[max(0, start - split) : stop - split])
                return result
            return [self[i] for i in range(start, stop, step)]
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.tokens[index] if index < split else self.suffix[index - split]

    def pop(self):
        self.length -= 1


def _find_close_offset(reasoner: KimiK3ReasoningParser, tokens: Sequence[int]) -> int | None:
    # Match the context-free fast path in the upstream fix. An incomplete close
    # at the beginning requires the context-aware singleton fallback instead.
    if not reasoner._thinking_enabled:
        return 0
    close = reasoner._think_close_ids
    if not close or len(close) != len(reasoner._think_open_ids):
        return None
    for offset in range(1, len(close)):
        remaining = len(close) - offset
        if remaining <= len(tokens) and all(tokens[i] == close[offset + i] for i in range(remaining)):
            return None
    for i in range(len(tokens) - len(close) + 1):
        if tokens[i] == close[0] and all(tokens[i + k] == close[k] for k in range(len(close))):
            return i + len(close) - 1
    return len(tokens)


def _kimi_constraint_start(manager, request, spec_tokens, committed):
    if manager.enable_in_reasoning:
        return None
    structured_req = request.structured_output_request
    if structured_req is None or structured_req.reasoning_ended:
        return None
    reasoner = manager._get_reasoner(request)
    if not isinstance(reasoner, KimiK3ReasoningParser):
        return None
    if structured_req.reasoning_ended is None:
        if reasoner.is_reasoning_end(request.prompt_token_ids or []):
            return None
        structured_req.reasoning_ended = False
    if not spec_tokens:
        return None
    offset = _find_close_offset(reasoner, spec_tokens)
    if offset is not None:
        return offset + 1

    input_ids = _TokenView(request.all_token_ids, () if committed else spec_tokens)
    start = len(spec_tokens) + 1
    for i in range(len(spec_tokens), 0, -1):
        if reasoner.is_reasoning_end_streaming(input_ids, (spec_tokens[i - 1],)):
            start = i
        input_ids.pop()
    return start


def _apply_patch():
    manager_cls = structured_output.StructuredOutputManager
    # v0.30 uses an integer; newer releases also distinguish grammar/content
    # boundaries. Kimi's explicit close gives identical bounds in the latter.
    has_bounds = hasattr(manager_cls, "_get_constraint_bounds")
    method_name = "_get_constraint_bounds" if has_bounds else "_get_constraint_start"
    original = getattr(manager_cls, method_name)
    if getattr(original, "_ascend_kimi_first_boundary", False):
        return

    @wraps(original)
    def boundary(self, request, spec_tokens, spec_tokens_committed=False):
        start = _kimi_constraint_start(self, request, spec_tokens, spec_tokens_committed)
        if start is None:
            return original(self, request, spec_tokens, spec_tokens_committed)
        if has_bounds:
            return structured_output.ConstraintBounds(start, start)
        return start

    boundary._ascend_kimi_first_boundary = True
    setattr(manager_cls, method_name, boundary)


_apply_patch()
