# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Stream long DeepSeek DSML parameter values while they are generated.

``ParserEngine._compute_arg_delta()`` drops every streaming delta that shares
no character with ``arg_structural_chars``. DeepSeek V4 and V4.1 set that to
``'>'``, which only appears at DSML tag boundaries, so a long ``string="true"``
parameter value (a file body, a heredoc, ...) is held back until its closing
tag arrives. The client then receives no bytes for the whole generation — a
15k-character payload keeps the stream silent for minutes, past any client
read timeout, while the engine keeps decoding normally.

This patch lets value-body deltas through for those two parsers only, and
still holds back a trailing partial tag so that tag fragments are never
serialised into the arguments. Everything else is delegated to the upstream
implementation, which keeps the prefix/diff bookkeeping intact.

Remove once the fix ships in a supported vLLM release
(upstream vllm-project/vllm#64196).
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from vllm.logger import logger

try:
    from vllm.parser.engine.parser_engine import ParserEngine
except ImportError:  # pragma: no cover - the vLLM parser layout changed
    ParserEngine = None  # type: ignore[assignment,no-redef]

_PATCH_MARKER = "_vllm_ascend_dsml_arg_streaming"

# Parsers whose tool arguments reach the parser as DSML text.
_TARGET_PARSERS = frozenset({"deepseek_v4", "deepseek_v41"})

# V4 spells ``<｜DSML｜parameter``, V4.1 inserts a space after ``｜DSML｜``.
_OPEN_PARAM_MARKERS = ("<｜DSML｜parameter", "<｜DSML｜ parameter")
_CLOSE_PARAM_MARKERS = ("</｜DSML｜parameter>", "</｜DSML｜ parameter>")
_STOP_TAGS = (
    *_OPEN_PARAM_MARKERS,
    *_CLOSE_PARAM_MARKERS,
    "</｜DSML｜invoke>",
    "</｜DSML｜ invoke>",
)


def _partial_tag_overlap(text: str, tag: str) -> int:
    """Length of the longest prefix of *tag* that matches a suffix of *text*."""
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


def _ends_with_partial_tag(text: str) -> bool:
    """Whether *text* ends with an unterminated DSML tag."""
    return any(_partial_tag_overlap(text, tag) for tag in _STOP_TAGS)


def _in_open_parameter_value(text: str) -> bool:
    """Whether the last DSML parameter that opened has not been closed."""
    last_open = max(text.rfind(marker) for marker in _OPEN_PARAM_MARKERS)
    if last_open < 0:
        return False
    last_close = max(text.rfind(marker) for marker in _CLOSE_PARAM_MARKERS)
    return last_open > last_close


def _should_stream_value(args: str, raw_delta: str, structural: frozenset[str]) -> bool:
    """Whether a delta that the structural gate would drop is a value body."""
    if not raw_delta or not structural.isdisjoint(raw_delta):
        return False
    return _in_open_parameter_value(args) and not _ends_with_partial_tag(args)


def _patch_compute_arg_delta(engine_cls: Any) -> None:
    original = getattr(engine_cls, "_compute_arg_delta", None)
    if original is None or getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def _compute_arg_delta(self, idx: int, raw_delta: str) -> str | None:
        structural = getattr(self, "_arg_structural_chars", None)
        config = getattr(self, "parser_engine_config", None)
        if structural is None or getattr(config, "name", None) not in _TARGET_PARSERS:
            return original(self, idx, raw_delta)

        slots = getattr(self, "_tool_slots", None)
        if not slots or idx >= len(slots):
            return original(self, idx, raw_delta)
        if not _should_stream_value(slots[idx].args, raw_delta, structural):
            return original(self, idx, raw_delta)

        # Reuse the upstream conversion, prefix and diff bookkeeping with the
        # gate disabled for this single delta only.
        self._arg_structural_chars = None
        try:
            return original(self, idx, raw_delta)
        finally:
            self._arg_structural_chars = structural

    setattr(_compute_arg_delta, _PATCH_MARKER, True)
    engine_cls._compute_arg_delta = _compute_arg_delta
    logger.info(
        "vllm-ascend: streaming long DSML parameter values for %s",
        ", ".join(sorted(_TARGET_PARSERS)),
    )


if ParserEngine is not None:
    _patch_compute_arg_delta(ParserEngine)
