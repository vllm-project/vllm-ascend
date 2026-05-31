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
# DeepSeek-V4 DSML: make the STREAMING tool_choice == "none" path consistent
# with the NON-streaming path.
#
# Background
# ----------
# For ChatCompletions, the non-streaming path (OpenAIServing._parse_tool_calls_
# from_content) does NOT run the tool parser when tool_choice is "none"/omitted,
# so any DSML tool-call markup the model emits stays verbatim in `content`.
# The streaming path, however, drives tool parsing through
# DelegatingParser.parse_delta -> extract_tool_calls_streaming, which runs the
# tool parser regardless of tool_choice and turns the same DSML into
# `delta.tool_calls` (finish_reason="tool_calls"). That is the upstream bug
# vllm-project/vllm#42747 ("streaming invokes tool parser despite
# tool_choice=none"); the two paths disagree.
#
# Fix
# ---
# Wrap DelegatingParser.extract_tool_calls_streaming: when tool_choice is
# "none"/omitted, do NOT invoke the tool parser; surface the delta text (the DSML
# markup) as plain `content` instead. tool_calls stay unset so finish_reason
# falls back to "stop". Result: streaming and non-streaming both return the raw
# DSML in content on the none path -- consistent behavior. The auto / forced
# tool-choice paths are untouched (original parser runs).
#
# (Mirrors upstream PR vllm-project/vllm#42752 done as a vllm-ascend monkey-patch.)

from __future__ import annotations

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser

logger = init_logger(__name__)


def _patch_streaming_tool_choice_none() -> None:
    if getattr(DelegatingParser, "_vllm_ascend_stream_none_content_patched", False):
        return

    _original_extract_tool_calls_streaming = DelegatingParser.extract_tool_calls_streaming

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
        # tool_choice == "none": keep DSML markup as plain content
        # (consistent with the non-streaming none path), do not parse tool calls.
        # We only treat the explicit "none" case here. tool_choice is None with
        # tools present defaults to "auto" upstream; tool_choice is None with
        # no tools is normalized to "none" by vLLM, so the default below covers
        # the omitted-no-tools case without us conflating it with an explicit
        # null (which non-streaming routes through the auto parser).
        if getattr(request, "tool_choice", "none") == "none":
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

    DelegatingParser.extract_tool_calls_streaming = _patched_extract_tool_calls_streaming
    DelegatingParser._vllm_ascend_stream_none_content_patched = True


_patch_streaming_tool_choice_none()
