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
# `finish_reason="tool_calls"` because `DelegatingParser.parse_delta`
# unconditionally invoked `_extract_tool_calls_streaming` once the stream
# entered the tool-call phase, ignoring `request.tool_choice`.
#
# Non-streaming Chat Completions already short-circuit this in
# `chat_completion/serving.py:1250`:
#
#     elif not request.tool_choice or request.tool_choice == "none":
#         message = ChatMessage(role=role, reasoning=reasoning, content=content)
#
# This patch replicates that semantics on the streaming path. When
# `tool_choice` is `None` or the string `"none"`, the tool parser is skipped
# inside the tool-call phase and the remaining (post-reasoning) text is
# surfaced as plain `content`. As a result the model's tool-call-looking
# output, including DSML markup for DeepSeek-V4, remains in
# `DeltaMessage.content` and `finish_reason` falls back to `"stop"`.

from __future__ import annotations

from vllm.entrypoints.openai.chat_completion.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser

logger = init_logger(__name__)


def _patch_delegating_parser_parse_delta() -> None:
    if getattr(
        DelegatingParser,
        "_vllm_ascend_tool_choice_none_streaming_patched",
        False,
    ):
        return

    def _patched_parse_delta(
        self,
        delta_text,
        delta_token_ids,
        request,
        prompt_token_ids=None,
    ):
        state = self._stream_state

        if not state.prompt_reasoning_checked and prompt_token_ids is not None:
            state.prompt_reasoning_checked = True
            if self._reasoning_parser is None or self.is_reasoning_end(
                prompt_token_ids
            ):
                state.reasoning_ended = True

        current_text = state.previous_text + delta_text
        current_token_ids = state.previous_token_ids + delta_token_ids
        delta_message: DeltaMessage | None = None

        # Reasoning extraction (unchanged from upstream).
        if self._in_reasoning_phase(state):
            delta_message = self.extract_reasoning_streaming(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=state.previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
            )
            if self._tool_parser and self.is_reasoning_end(delta_token_ids):
                state.reasoning_ended = True
                current_token_ids = self.extract_content_ids(delta_token_ids)
                if delta_message and delta_message.content:
                    current_text = delta_message.content
                    delta_message.content = None
                else:
                    current_text = ""

        # Tool-call extraction with `tool_choice="none"` / omitted guard.
        if self._in_tool_call_phase(state):
            if not state.tool_call_text_started:
                state.tool_call_text_started = True
                state.previous_text = ""
                state.previous_token_ids = []
                delta_text = current_text
                delta_token_ids = current_token_ids

            suppress_tool_parser = (
                not getattr(request, "tool_choice", None)
                or request.tool_choice == "none"
            )
            if suppress_tool_parser:
                # Mirror non-streaming: leave tool-call-looking text in content.
                if delta_text:
                    if delta_message is None:
                        delta_message = DeltaMessage(content=delta_text)
                    else:
                        delta_message.content = delta_text
            else:
                delta_message, state.function_name_returned = (
                    self._extract_tool_calls_streaming(
                        previous_text=state.previous_text,
                        current_text=current_text,
                        delta_text=delta_text,
                        previous_token_ids=state.previous_token_ids,
                        current_token_ids=current_token_ids,
                        delta_token_ids=delta_token_ids,
                        request=request,  # type: ignore[arg-type]
                        tool_call_idx=state.history_tool_call_cnt,
                        tool_call_id_type=state.tool_call_id_type,
                        function_name_returned=state.function_name_returned,
                    )
                )
                if (
                    delta_message
                    and delta_message.tool_calls
                    and delta_message.tool_calls[0].id is not None
                ):
                    state.history_tool_call_cnt += 1

        # No phase active: pass through as content (unchanged from upstream).
        if (
            delta_message is None
            and not self._in_reasoning_phase(state)
            and not self._in_tool_call_phase(state)
        ):
            delta_message = DeltaMessage(content=delta_text)

        state.previous_text = current_text
        state.previous_token_ids = current_token_ids
        return delta_message

    DelegatingParser.parse_delta = _patched_parse_delta
    DelegatingParser._vllm_ascend_tool_choice_none_streaming_patched = True


_patch_delegating_parser_parse_delta()
