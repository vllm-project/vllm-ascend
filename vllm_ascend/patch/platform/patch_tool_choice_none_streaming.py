#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Mirror of upstream fix vllm-project/vllm#42752 (issue #42747).
#
# Streaming Chat Completions with `tool_choice="none"` (or omitted on a
# no-tools request) could still produce `delta.tool_calls` and
# `finish_reason="tool_calls"` because `DelegatingParser.parse_delta` invokes
# the streaming tool extractor once the stream enters the tool-call phase,
# ignoring `request.tool_choice`.
#
# Non-streaming Chat Completions already short-circuit this: on the
# "none"/omitted path the tool parser is never run, so the model's
# tool-call-looking output stays in the assistant `content`.
#
# This patch replicates that semantics on the streaming path. When
# `tool_choice` is `None` or the string `"none"`, the tool parser is skipped
# inside the tool-call phase and the (post-reasoning) delta text is surfaced as
# plain `content`. As a result the model's tool-call-looking output -- including
# DSML markup for DeepSeek-V4 -- remains in `DeltaMessage.content` and
# `finish_reason` falls back to `"stop"`, consistent with non-streaming.
#
# Implementation note (vLLM v0.20.2 compatibility)
# ------------------------------------------------
# On v0.20.2 `DelegatingParser.parse_delta` drives streaming tool parsing
# through the public, single-return `extract_tool_calls_streaming(...)
# -> DeltaMessage | None`. We therefore wrap *that* method instead of
# reimplementing `parse_delta`: this is minimal, avoids coupling to
# `parse_delta`'s internal state machine / private helpers (e.g. a
# `_extract_tool_calls_streaming` tuple-returning variant that does not exist
# in v0.20.2), and leaves reasoning extraction and the auto / forced
# tool-choice paths completely untouched.

from __future__ import annotations

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser

logger = init_logger(__name__)

_TOOL_CHOICE_NONE = (None, "none")


def _patch_streaming_tool_choice_none() -> None:
    if getattr(
        DelegatingParser,
        "_vllm_ascend_tool_choice_none_streaming_patched",
        False,
    ):
        return

    _original_extract_tool_calls_streaming = (
        DelegatingParser.extract_tool_calls_streaming
    )

    def _patched_extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        # tool_choice == "none"/omitted: do NOT parse tool calls; surface the
        # delta text (the tool-call markup, e.g. DSML) as plain content so the
        # streamed output matches the non-streaming none path.
        if getattr(request, "tool_choice", None) in _TOOL_CHOICE_NONE:
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None
        return _original_extract_tool_calls_streaming(
            self,
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )

    DelegatingParser.extract_tool_calls_streaming = (
        _patched_extract_tool_calls_streaming
    )
    DelegatingParser._vllm_ascend_tool_choice_none_streaming_patched = True


_patch_streaming_tool_choice_none()
