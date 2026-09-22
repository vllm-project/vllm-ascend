# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Serving adapters for the local DeepSeek V4.1 parser fallback."""

from vllm.parser.engine.adapters import make_adapters

from .parser import DeepSeekV41Parser

(
    DeepSeekV41ParserReasoningAdapter,
    DeepSeekV41EngineToolParser,
) = make_adapters(DeepSeekV41Parser)
DeepSeekV41EngineToolParser.structural_tag_model = "deepseek_v41"  # type: ignore[attr-defined]


__all__ = [
    "DeepSeekV41EngineToolParser",
    "DeepSeekV41ParserReasoningAdapter",
]
