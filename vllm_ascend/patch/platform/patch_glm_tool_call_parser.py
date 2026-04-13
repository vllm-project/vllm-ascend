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
# OpenAI chat streaming: backport GLM tool-call parser and finish-suffix fixes.
#

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Any, Final

from vllm.entrypoints.openai.chat_completion import serving as chat_serving
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ConversationMessage,
    GenerationError,
    MistralToolCall,
    OpenAIServingChat,
    ReasoningParser,
    RequestOutput,
    RequestResponseMetadata,
    TokenizerLike,
    TokenState,
    ToolParser,
    as_list,
    extract_harmony_streaming_delta,
    get_history_tool_calls_cnt,
    get_streamable_parser_for_assistant,
    is_mistral_tokenizer,
    make_tool_call_id,
    maybe_filter_parallel_tool_calls,
    should_include_usage,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers import glm4_moe_tool_parser as glm4_parser
from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser

logger = chat_serving.logger


def _create_remaining_args_delta(
    delta_message: DeltaMessage,
    remaining_call: str,
    index: int,
    fallback_tool_call_id: str | None = None,
    fallback_tool_call_type: str | None = None,
    fallback_tool_call_name: str | None = None,
) -> DeltaMessage:
    """
    Create a delta message for remaining tool arguments.

    Per OpenAI streaming semantics, id/type/function.name must only appear
    in the *first* chunk for a given tool call index.  Callers must pass
    non-None fallback_* values only when this is genuinely the first chunk
    (i.e. nothing has been streamed yet for this tool call).  When all
    fallback_* are None the header fields are omitted entirely, which is the
    correct behaviour for continuing/finishing chunks.
    """
    include_header = any(
        v is not None for v in (fallback_tool_call_id, fallback_tool_call_type, fallback_tool_call_name)
    )
    if not include_header:
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    function=DeltaFunctionCall(
                        arguments=remaining_call,
                    ),
                )
            ]
        )
    return DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=index,
                id=fallback_tool_call_id,
                type=fallback_tool_call_type,
                function=DeltaFunctionCall(
                    name=fallback_tool_call_name,
                    arguments=remaining_call,
                ),
            )
        ]
    )


def _record_streamed_tool_args(
    delta_message: DeltaMessage,
    streamed_tool_args: dict[int, str],
) -> None:
    if not delta_message.tool_calls:
        return

    for tool_call in delta_message.tool_calls:
        function = tool_call.function
        arguments = None
        if isinstance(function, DeltaFunctionCall):
            arguments = function.arguments
        elif isinstance(function, dict):
            arguments = function.get("arguments")

        if isinstance(arguments, str):
            streamed_tool_args[tool_call.index] = streamed_tool_args.get(tool_call.index, "") + arguments


def _compact_json_fragment(fragment: str) -> str:
    compact_chars: list[str] = []
    in_string = False
    escaped = False

    for ch in fragment:
        if in_string:
            compact_chars.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch in " \t\r\n":
            continue

        compact_chars.append(ch)
        if ch == '"':
            in_string = True

    return "".join(compact_chars)


def _compute_remaining_tool_args(
    cls,
    expected_args: Any,
    streamed_args: str,
) -> str:
    actual_call = streamed_args

    expected_call_candidates: list[str] = []
    parsed_expected_args: Any = expected_args
    expected_compact: str | None = None

    if isinstance(expected_args, str):
        expected_call_candidates.append(expected_args)
        try:
            parsed_expected_args = json.loads(expected_args)
        except json.JSONDecodeError:
            parsed_expected_args = expected_args
    else:
        expected_call_candidates.append(json.dumps(expected_args, ensure_ascii=False))

    if not isinstance(parsed_expected_args, str):
        expected_compact = json.dumps(
            parsed_expected_args,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if expected_compact not in expected_call_candidates:
            expected_call_candidates.append(expected_compact)

    for expected_call in expected_call_candidates:
        if expected_call.startswith(actual_call):
            return expected_call[len(actual_call) :]

    if expected_compact is not None:
        actual_compact = cls._compact_json_fragment(actual_call)
        if expected_compact.startswith(actual_compact):
            return expected_compact[len(actual_compact) :]

    if actual_call:
        logger.debug(
            "Unable to align streamed tool args with expected suffix; skip finish backfill.",
        )
    return ""


async def _patched_chat_completion_stream_generator(
    self,
    request: ChatCompletionRequest,
    result_generator: AsyncIterator[RequestOutput],
    request_id: str,
    model_name: str,
    conversation: list[ConversationMessage],
    tokenizer: TokenizerLike,
    request_metadata: RequestResponseMetadata,
    reasoning_parser: ReasoningParser | None = None,
) -> AsyncGenerator[str, None]:
    created_time = int(time.time())
    chunk_object_type: Final = "chat.completion.chunk"
    first_iteration = True

    # Copied from the current upstream vLLM method so this backport stays
    # self-contained instead of depending on fragile source rewrites.
    num_choices = 1 if request.n is None else request.n
    previous_num_tokens = [0] * num_choices
    raw_output_token_ids: list[list[int]] = [[] for _ in range(num_choices)]
    finish_reason_sent = [False] * num_choices
    num_prompt_tokens = 0
    num_cached_tokens = None
    if self.use_harmony:
        harmony_parsers = [get_streamable_parser_for_assistant() for _ in range(num_choices)]
        harmony_tools_streamed = [False] * num_choices
    tools_streamed = [False] * num_choices
    streamed_tool_args: list[dict[int, str]] = [{} for _ in range(num_choices)]

    if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
        tool_choice_function_name = request.tool_choice.function.name
    else:
        tool_choice_function_name = None

    tool_choice_auto = not tool_choice_function_name and self._should_stream_with_auto_tool_parsing(request)

    all_previous_token_ids: list[list[int]] | None
    function_name_returned = [False] * num_choices
    if self.tool_call_id_type == "kimi_k2":
        history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
    else:
        history_tool_call_cnt = 0

    previous_texts = [""] * num_choices

    if tool_choice_auto or reasoning_parser:
        all_previous_token_ids = [[]] * num_choices
        added_content_delta_arr = [False] * num_choices
        reasoning_end_arr = [False] * num_choices
        prompt_is_reasoning_end_arr: list[bool | None] = [None] * num_choices
    else:
        all_previous_token_ids = None

    try:
        if tool_choice_auto and self.tool_parser:
            if tokenizer is None:
                raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

            tool_parsers: list[ToolParser | None] = [self.tool_parser(tokenizer)] * num_choices
        else:
            tool_parsers = [None] * num_choices
    except Exception as e:
        logger.exception("Error in tool parser creation.")
        data = self.create_streaming_error_response(e)
        yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
        return

    stream_options = request.stream_options
    include_usage, include_continuous_usage = should_include_usage(stream_options, self.enable_force_include_usage)

    try:
        async for res in result_generator:
            if res.prompt_token_ids is not None:
                num_prompt_tokens = len(res.prompt_token_ids)
                if res.encoder_prompt_token_ids is not None:
                    num_prompt_tokens += len(res.encoder_prompt_token_ids)

            if first_iteration:
                num_cached_tokens = res.num_cached_tokens
                role = self.get_chat_request_role(request)

                for i in range(num_choices):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(
                            role=role,
                            content="",
                        ),
                        logprobs=None,
                        finish_reason=None,
                    )

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                        prompt_token_ids=(res.prompt_token_ids if request.return_token_ids else None),
                    )

                    if include_continuous_usage:
                        chunk.usage = self._make_usage_info(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=0,
                            reasoning_tokens=self._count_reasoning_tokens_for_usage(
                                raw_output_token_ids[i], reasoning_parser
                            ),
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

                if request.echo:
                    last_msg_content: str | list[dict[str, str]] = ""
                    if conversation and "content" in conversation[-1] and conversation[-1].get("role") == role:
                        last_msg_content = conversation[-1]["content"] or ""

                    if last_msg_content:
                        for i in range(num_choices):
                            choice_data = ChatCompletionResponseStreamChoice(
                                index=i,
                                delta=DeltaMessage(content=last_msg_content),
                                logprobs=None,
                                finish_reason=None,
                            )
                            chunk = ChatCompletionStreamResponse(
                                id=request_id,
                                object=chunk_object_type,
                                created=created_time,
                                choices=[choice_data],
                                model=model_name,
                            )
                            if include_continuous_usage:
                                chunk.usage = self._make_usage_info(
                                    prompt_tokens=num_prompt_tokens,
                                    completion_tokens=0,
                                    reasoning_tokens=(
                                        self._count_reasoning_tokens_for_usage(
                                            raw_output_token_ids[i],
                                            reasoning_parser,
                                        )
                                    ),
                                )

                            data = chunk.model_dump_json(exclude_unset=True)
                            yield f"data: {data}\n\n"
                first_iteration = False

            for output in res.outputs:
                i = output.index
                tool_parser = tool_parsers[i]

                if reasoning_parser and res.prompt_token_ids and prompt_is_reasoning_end_arr[i] is None:
                    prompt_is_reasoning_end_arr[i] = reasoning_parser.is_reasoning_end(res.prompt_token_ids)
                if finish_reason_sent[i]:
                    continue

                if request.logprobs and request.top_logprobs is not None:
                    assert output.logprobs is not None, "Did not output logprobs"
                    logprobs = self._create_chat_logprobs(
                        token_ids=output.token_ids,
                        top_logprobs=output.logprobs,
                        tokenizer=tokenizer,
                        num_output_top_logprobs=request.top_logprobs,
                        return_as_token_id=request.return_tokens_as_token_ids,
                    )
                else:
                    logprobs = None

                if self.use_harmony:
                    harmony_parser = harmony_parsers[i]
                    prev_recipient = harmony_parser.current_recipient

                    token_states: list[TokenState] = []
                    for token_id in output.token_ids:
                        harmony_parser.process(token_id)
                        token_delta = harmony_parser.last_content_delta or ""
                        token_states.append(
                            TokenState(
                                harmony_parser.current_channel,
                                harmony_parser.current_recipient,
                                token_delta,
                            )
                        )
                    delta_text = "".join(delta for _, _, delta in token_states)
                    cur_channel = harmony_parser.current_channel

                    if not cur_channel and delta_text:
                        cur_channel = "final"
                else:
                    delta_text = output.text

                if not delta_text and not output.token_ids and not previous_num_tokens[i]:
                    continue

                delta_message: DeltaMessage | None

                if tool_choice_auto or reasoning_parser:
                    assert previous_texts is not None
                    assert all_previous_token_ids is not None
                    previous_text = previous_texts[i]
                    previous_token_ids = all_previous_token_ids[i]
                    current_text = previous_text + delta_text
                    if previous_token_ids:
                        current_token_ids = previous_token_ids + as_list(output.token_ids)
                    else:
                        current_token_ids = as_list(output.token_ids)

                if self.use_harmony:
                    delta_message, tools_streamed_flag = extract_harmony_streaming_delta(
                        harmony_parser=harmony_parser,
                        token_states=token_states,
                        prev_recipient=prev_recipient,
                        include_reasoning=request.include_reasoning,
                    )
                    harmony_tools_streamed[i] |= tools_streamed_flag
                elif tool_choice_function_name:
                    if reasoning_parser and not reasoning_end_arr[i] and prompt_is_reasoning_end_arr[i]:
                        reasoning_end_arr[i] = True

                    if (
                        reasoning_parser
                        and not reasoning_end_arr[i]
                        and not reasoning_parser.is_reasoning_end(previous_token_ids)
                    ):
                        assert reasoning_parser is not None
                        delta_message = reasoning_parser.extract_reasoning_streaming(
                            previous_text,
                            current_text,
                            delta_text,
                            previous_token_ids,
                            current_token_ids,
                            output.token_ids,
                        )
                        if reasoning_parser.is_reasoning_end(as_list(output.token_ids)):
                            reasoning_end_arr[i] = True
                            if delta_message and delta_message.content:
                                current_text = delta_message.content
                                delta_message.content = None
                            else:
                                current_text = ""
                    else:
                        if reasoning_parser:
                            delta_text = previous_text + delta_text
                            current_text = ""

                        if function_name_returned[i]:
                            delta_tool_call = DeltaToolCall(
                                function=DeltaFunctionCall(arguments=delta_text),
                                index=i,
                            )
                        else:
                            if is_mistral_tokenizer(tokenizer):
                                tool_call_id = MistralToolCall.generate_random_id()
                            else:
                                tool_call_id = make_tool_call_id(
                                    id_type=self.tool_call_id_type,
                                    func_name=tool_choice_function_name,
                                    idx=history_tool_call_cnt,
                                )
                            delta_tool_call = DeltaToolCall(
                                id=tool_call_id,
                                type="function",
                                function=DeltaFunctionCall(
                                    name=tool_choice_function_name,
                                    arguments=delta_text,
                                ),
                                index=i,
                            )
                            function_name_returned[i] = True
                            history_tool_call_cnt += 1

                        delta_message = DeltaMessage(tool_calls=[delta_tool_call])
                        tools_streamed[i] = True

                elif request.tool_choice == "required":
                    assert previous_texts is not None
                    previous_text = previous_texts[i]
                    current_text = previous_text + delta_text
                    fn_name_returned = function_name_returned[i]
                    output_token_ids = as_list(output.token_ids)

                    if reasoning_parser is not None and not reasoning_end_arr[i] and prompt_is_reasoning_end_arr[i]:
                        reasoning_end_arr[i] = True

                    if reasoning_parser and not reasoning_end_arr[i]:
                        delta_message = reasoning_parser.extract_reasoning_streaming(
                            previous_text,
                            current_text,
                            delta_text,
                            previous_token_ids,
                            current_token_ids,
                            output_token_ids,
                        )
                        if reasoning_parser.is_reasoning_end(output_token_ids):
                            reasoning_end_arr[i] = True
                            if delta_message and delta_message.content:
                                current_text = delta_message.content
                                delta_message.content = None
                            else:
                                current_text = ""
                    else:
                        content = current_text
                        (
                            delta_message,
                            function_name_returned[i],
                        ) = self.extract_tool_call_required_streaming(
                            previous_text=previous_text,
                            current_text=content,
                            delta_text=delta_text,
                            function_name_returned=fn_name_returned,
                            tool_call_idx=history_tool_call_cnt,
                        )
                        if delta_message and delta_message.tool_calls and delta_message.tool_calls[0].id is not None:
                            history_tool_call_cnt += 1
                            tools_streamed[i] = True

                elif tool_choice_auto and reasoning_parser:
                    assert tool_parser is not None
                    assert added_content_delta_arr is not None
                    assert reasoning_end_arr is not None
                    output_token_ids = as_list(output.token_ids)
                    if not reasoning_end_arr[i]:
                        if prompt_is_reasoning_end_arr[i]:
                            reasoning_end_arr[i] = True
                            current_token_ids = output_token_ids
                        else:
                            delta_message = reasoning_parser.extract_reasoning_streaming(
                                previous_text,
                                current_text,
                                delta_text,
                                previous_token_ids,
                                current_token_ids,
                                output_token_ids,
                            )

                            if reasoning_parser.is_reasoning_end(output_token_ids):
                                reasoning_end_arr[i] = True
                                current_token_ids = reasoning_parser.extract_content_ids(output_token_ids)
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""

                    if reasoning_end_arr[i]:
                        delta_token_ids = output_token_ids
                        if not added_content_delta_arr[i]:
                            added_content_delta_arr[i] = True
                            previous_text = ""
                            previous_token_ids = []
                            delta_text = current_text
                            delta_token_ids = current_token_ids

                        delta_message = tool_parser.extract_tool_calls_streaming(
                            previous_text=previous_text,
                            current_text=current_text,
                            delta_text=delta_text,
                            previous_token_ids=previous_token_ids,
                            current_token_ids=current_token_ids,
                            delta_token_ids=delta_token_ids,
                            request=request,
                        )
                        if delta_message and delta_message.tool_calls:
                            tools_streamed[i] = True
                elif tool_choice_auto:
                    assert tool_parser is not None
                    delta_message = tool_parser.extract_tool_calls_streaming(
                        previous_text=previous_text,
                        current_text=current_text,
                        delta_text=delta_text,
                        previous_token_ids=previous_token_ids,
                        current_token_ids=current_token_ids,
                        delta_token_ids=output.token_ids,
                        request=request,
                    )
                    if delta_message and delta_message.tool_calls:
                        tools_streamed[i] = True
                elif reasoning_parser:
                    if prompt_is_reasoning_end_arr[i]:
                        delta_message = DeltaMessage(content=delta_text)
                    else:
                        delta_message = reasoning_parser.extract_reasoning_streaming(
                            previous_text,
                            current_text,
                            delta_text,
                            previous_token_ids,
                            current_token_ids,
                            output.token_ids,
                        )
                else:
                    delta_message = DeltaMessage(content=delta_text)

                if (tool_choice_auto or reasoning_parser) and not self.use_harmony:
                    assert previous_texts is not None
                    assert all_previous_token_ids is not None
                    previous_texts[i] = current_text
                    all_previous_token_ids[i] = current_token_ids
                else:
                    assert previous_texts is not None
                    previous_texts[i] += delta_text

                previous_num_tokens[i] += len(output.token_ids)
                raw_output_token_ids[i].extend(as_list(output.token_ids))

                if delta_message is None:
                    if output.finish_reason is None and not request.return_token_ids:
                        continue
                    delta_message = DeltaMessage()

                if self.enable_log_outputs and self.request_logger:
                    delta_content_parts = []
                    if delta_message.content:
                        delta_content_parts.append(delta_message.content)
                    if delta_message.reasoning:
                        reasoning = delta_message.reasoning
                        delta_content_parts.append(f"[reasoning: {reasoning}]")
                    if delta_message.tool_calls:
                        tool_args = "".join(
                            tc.function.arguments
                            for tc in delta_message.tool_calls
                            if tc.function and tc.function.arguments
                        )
                        if tool_args:
                            delta_content_parts.append(f"[tool_calls: {tool_args}]")

                    if delta_content_parts and self.enable_log_deltas:
                        delta_content = " ".join(delta_content_parts)
                        self.request_logger.log_outputs(
                            request_id=request_id,
                            outputs=delta_content,
                            output_token_ids=as_list(output.token_ids),
                            finish_reason=output.finish_reason,
                            is_streaming=True,
                            delta=True,
                        )

                if output.finish_reason is None:
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=delta_message,
                        logprobs=logprobs,
                        finish_reason=None,
                        token_ids=(as_list(output.token_ids) if request.return_token_ids else None),
                    )
                else:
                    self._raise_if_error(output.finish_reason, request_id)

                    auto_tools_called = False
                    if tool_parser:
                        auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
                        index = len(tool_parser.prev_tool_call_arr) - 1 if auto_tools_called else 0
                    else:
                        index = 0

                    if self._should_check_for_unstreamed_tool_arg_tokens(delta_message, output) and tool_parser:
                        already_streamed_args = streamed_tool_args[i].get(index, "")
                        remaining_call = self._compute_remaining_tool_args(
                            expected_args=tool_parser.prev_tool_call_arr[index].get("arguments", {}),
                            streamed_args=already_streamed_args,
                        )

                        # Per OpenAI streaming semantics, id/type/name must only
                        # appear in the *first* chunk for a tool call index.
                        # If already_streamed_args is non-empty the header was
                        # sent in a previous chunk, so pass None for all fallback
                        # fields so _create_remaining_args_delta omits them.
                        fallback_tool_call_id = None
                        fallback_tool_call_type = None
                        fallback_tool_call_name = None
                        if not already_streamed_args:
                            fallback_tool_call = (
                                tool_parser.prev_tool_call_arr[index]
                                if index < len(tool_parser.prev_tool_call_arr)
                                else {}
                            )
                            if isinstance(fallback_tool_call, dict):
                                fallback_tool_call_id = fallback_tool_call.get("id")
                                fallback_tool_call_type = fallback_tool_call.get("type")
                                fallback_tool_call_name = fallback_tool_call.get("name")

                            tool_call_ids = getattr(tool_parser, "_tool_call_ids", None)
                            if (
                                fallback_tool_call_id is None
                                and isinstance(tool_call_ids, list)
                                and index < len(tool_call_ids)
                            ):
                                fallback_tool_call_id = tool_call_ids[index]

                            if fallback_tool_call_type is None and (
                                fallback_tool_call_id is not None or fallback_tool_call_name is not None
                            ):
                                fallback_tool_call_type = "function"

                        delta_message = self._create_remaining_args_delta(
                            delta_message,
                            remaining_call,
                            index,
                            fallback_tool_call_id=fallback_tool_call_id,
                            fallback_tool_call_type=fallback_tool_call_type,
                            fallback_tool_call_name=fallback_tool_call_name,
                        )

                    if (
                        auto_tools_called
                        or (tools_streamed[i] and not tool_choice_function_name)
                        or (self.use_harmony and harmony_tools_streamed[i])
                    ):
                        finish_reason_ = "tool_calls"
                    else:
                        finish_reason_ = output.finish_reason if output.finish_reason else "stop"
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=delta_message,
                        logprobs=logprobs,
                        finish_reason=finish_reason_,
                        stop_reason=output.stop_reason,
                        token_ids=(as_list(output.token_ids) if request.return_token_ids else None),
                    )

                    finish_reason_sent[i] = True

                choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                self._record_streamed_tool_args(
                    choice_data.delta,
                    streamed_tool_args[i],
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[choice_data],
                    model=model_name,
                )

                if include_continuous_usage:
                    completion_tokens = previous_num_tokens[i]
                    chunk.usage = self._make_usage_info(
                        prompt_tokens=num_prompt_tokens,
                        completion_tokens=completion_tokens,
                        reasoning_tokens=self._count_reasoning_tokens_for_usage(
                            raw_output_token_ids[i], reasoning_parser
                        ),
                    )

                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"

        if include_usage:
            completion_tokens = sum(previous_num_tokens)
            reasoning_tokens = None
            if reasoning_parser is not None:
                reasoning_tokens = sum(
                    self._count_reasoning_tokens_for_usage(token_ids, reasoning_parser) or 0
                    for token_ids in raw_output_token_ids
                )
            final_usage = self._make_usage_info(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                num_cached_tokens=num_cached_tokens,
                reasoning_tokens=reasoning_tokens,
            )

            final_usage_chunk = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[],
                model=model_name,
                usage=final_usage,
            )
            final_usage_data = final_usage_chunk.model_dump_json(
                exclude_unset=True,
                exclude_none=True,
            )
            yield f"data: {final_usage_data}\n\n"

        num_completion_tokens = sum(previous_num_tokens)
        reasoning_tokens = None
        if reasoning_parser is not None:
            reasoning_tokens = sum(
                self._count_reasoning_tokens_for_usage(token_ids, reasoning_parser) or 0
                for token_ids in raw_output_token_ids
            )
        request_metadata.final_usage_info = self._make_usage_info(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_completion_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        if self.enable_log_outputs and self.request_logger:
            for i in range(num_choices):
                full_text = (
                    previous_texts[i]
                    if previous_texts and i < len(previous_texts)
                    else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                )
                self.request_logger.log_outputs(
                    request_id=request_id,
                    outputs=full_text,
                    output_token_ids=None,
                    finish_reason="streaming_complete",
                    is_streaming=True,
                    delta=False,
                )

    except GenerationError as e:
        yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
    except Exception as e:
        logger.exception("Error in chat completion stream generator.")
        data = self.create_streaming_error_response(e)
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


def _patched_extract_tool_calls_streaming(
    self: Glm4MoeModelToolParser,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Sequence[int],
    current_token_ids: Sequence[int],
    delta_token_ids: Sequence[int],
    request: ChatCompletionRequest,
) -> DeltaMessage | None:
    if not self._tools_enabled(request):
        return DeltaMessage(content=delta_text) if delta_text else None

    self._buffer += delta_text

    # Drain the current buffer before returning so a single terminal chunk can
    # emit both the final string value and the closing tool-call suffix.
    pending_delta: DeltaMessage | None = None

    def _append_delta(message: DeltaMessage | None) -> None:
        nonlocal pending_delta
        if message is None:
            return

        if pending_delta is None:
            pending_delta = message.model_copy(deep=True)
            return

        def _normalize_function_call(
            function: DeltaFunctionCall | dict[str, Any] | None,
        ) -> DeltaFunctionCall | None:
            if function is None:
                return None
            if isinstance(function, DeltaFunctionCall):
                return function.model_copy(deep=True)
            return DeltaFunctionCall.model_validate(function)

        if message.content:
            pending_delta.content = (pending_delta.content or "") + message.content
        if message.reasoning:
            pending_delta.reasoning = (pending_delta.reasoning or "") + message.reasoning

        for tool_call in message.tool_calls:
            for idx, existing_tool_call in enumerate(pending_delta.tool_calls):
                if existing_tool_call.index != tool_call.index:
                    continue

                existing_function = _normalize_function_call(existing_tool_call.function)
                incoming_function = _normalize_function_call(tool_call.function)
                merged_name = (
                    incoming_function.name
                    if incoming_function and incoming_function.name is not None
                    else existing_function.name
                    if existing_function
                    else None
                )
                merged_arguments = None
                if (existing_function and existing_function.arguments is not None) or (
                    incoming_function and incoming_function.arguments is not None
                ):
                    merged_arguments = ((existing_function.arguments or "") if existing_function else "") + (
                        (incoming_function.arguments or "") if incoming_function else ""
                    )

                merged_function = None
                if merged_name is not None or merged_arguments is not None:
                    merged_function = DeltaFunctionCall(
                        name=merged_name,
                        arguments=merged_arguments,
                    ).model_dump(exclude_none=True)

                pending_delta.tool_calls[idx] = DeltaToolCall(
                    index=existing_tool_call.index,
                    id=(tool_call.id if tool_call.id is not None else existing_tool_call.id),
                    type=(tool_call.type if tool_call.type is not None else existing_tool_call.type),
                    function=merged_function,
                )
                break
            else:
                pending_delta.tool_calls = [
                    *pending_delta.tool_calls,
                    tool_call.model_copy(deep=True),
                ]

    def _flush_pending() -> DeltaMessage | None:
        if pending_delta is None:
            return None
        if pending_delta.content is None and pending_delta.reasoning is None and not pending_delta.tool_calls:
            return None
        return pending_delta

    while True:
        if not self._in_tool_call:
            start_idx = self._buffer.find(self.tool_call_start_token)
            if start_idx == -1:
                for i in range(1, len(self.tool_call_start_token)):
                    if self._buffer.endswith(self.tool_call_start_token[:i]):
                        out = self._buffer[:-i]
                        self._buffer = self._buffer[-i:]
                        _append_delta(DeltaMessage(content=out) if out else None)
                        return _flush_pending()
                out = self._buffer
                self._buffer = ""
                _append_delta(DeltaMessage(content=out) if out else None)
                return _flush_pending()

            if start_idx > 0:
                out = self._buffer[:start_idx]
                self._buffer = self._buffer[start_idx:]
                _append_delta(DeltaMessage(content=out) if out else None)
                continue

            self._buffer = self._buffer[len(self.tool_call_start_token) :]
            self._begin_tool_call()
            continue

        if not self.current_tool_name_sent:
            nl = self._buffer.find("\n")
            ak = self._buffer.find(self.arg_key_start)
            end = self._buffer.find(self.tool_call_end_token)
            candidates = [i for i in [nl, ak, end] if i != -1]
            if not candidates:
                return _flush_pending()
            cut = min(candidates)
            tool_name = self._buffer[:cut].strip()
            if tool_name == "" and cut == end:
                self._buffer = self._buffer[end + len(self.tool_call_end_token) :]
                self._finish_tool_call()
                self._revert_last_tool_call_state()
                continue

            if cut == nl:
                self._buffer = self._buffer[nl + 1 :]
            else:
                self._buffer = self._buffer[cut:]

            self._current_tool_name = tool_name
            self.current_tool_name_sent = True
            _append_delta(self._emit_tool_name_delta(tool_name))
            continue

        assert self._current_tool_name is not None

        if self._streaming_string_value:
            val_end = self._buffer.find(self.arg_val_end)
            if val_end != -1:
                raw_content = self._buffer[:val_end]
                self._buffer = self._buffer[val_end + len(self.arg_val_end) :]
                self._streaming_string_value = False
                self._pending_key = None

                escaped = self._json_escape_string_content(raw_content)
                frag = escaped + '"'
                self.streamed_args_for_tool[self.current_tool_id] += frag
                _append_delta(self._emit_tool_args_delta(frag))
                continue

            safe_len = len(self._buffer)
            for i in range(1, len(self.arg_val_end)):
                if self._buffer.endswith(self.arg_val_end[:i]):
                    safe_len = len(self._buffer) - i
                    break

            if safe_len > 0:
                to_emit = self._buffer[:safe_len]
                self._buffer = self._buffer[safe_len:]
                escaped = self._json_escape_string_content(to_emit)
                if escaped:
                    self.streamed_args_for_tool[self.current_tool_id] += escaped
                    _append_delta(self._emit_tool_args_delta(escaped))
            return _flush_pending()

        if self._pending_key is not None:
            val_pos = self._buffer.find(self.arg_val_start)
            if val_pos == -1:
                return _flush_pending()
            if val_pos > 0:
                self._buffer = self._buffer[val_pos:]

            key = (self._pending_key or "").strip()
            is_string = self._is_string_type(self._current_tool_name, key, request.tools)

            if is_string:
                self._buffer = self._buffer[len(self.arg_val_start) :]

                if key in self._seen_keys[self.current_tool_id]:
                    self._pending_key = None
                    continue

                self._seen_keys[self.current_tool_id].add(key)
                key_json = json.dumps(key, ensure_ascii=False)

                if not self._args_started[self.current_tool_id]:
                    frag = "{" + key_json + ':"'
                    self._args_started[self.current_tool_id] = True
                else:
                    frag = "," + key_json + ':"'

                self.streamed_args_for_tool[self.current_tool_id] += frag
                self._streaming_string_value = True
                _append_delta(self._emit_tool_args_delta(frag))
                continue

            val_end = self._buffer.find(self.arg_val_end)
            if val_end == -1:
                return _flush_pending()

            raw_val = self._buffer[len(self.arg_val_start) : val_end].strip()
            self._buffer = self._buffer[val_end + len(self.arg_val_end) :]
            self._pending_key = None

            frag = self._append_arg_fragment(
                key=key,
                raw_val=raw_val,
            )
            if frag:
                _append_delta(self._emit_tool_args_delta(frag))
            continue

        end_pos = self._buffer.find(self.tool_call_end_token)
        key_pos = self._buffer.find(self.arg_key_start)
        if end_pos != -1 and (key_pos == -1 or end_pos < key_pos):
            self._buffer = self._buffer[end_pos + len(self.tool_call_end_token) :]
            frag = self._close_args_if_needed()
            if self._current_tool_name:
                try:
                    full_args_str = self.streamed_args_for_tool[self.current_tool_id]
                    args_dict = json.loads(full_args_str)
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": self._current_tool_name,
                        "arguments": args_dict,
                    }
                except (json.JSONDecodeError, IndexError) as e:
                    glm4_parser.logger.warning(
                        "Failed to finalize tool call state for tool %d: %s",
                        self.current_tool_id,
                        e,
                    )
            self._finish_tool_call()
            _append_delta(self._emit_tool_args_delta(frag) if frag else None)
            continue

        if key_pos == -1:
            return _flush_pending()
        if key_pos > 0:
            self._buffer = self._buffer[key_pos:]
        key_end = self._buffer.find(self.arg_key_end)
        if key_end == -1:
            return _flush_pending()
        key = self._buffer[len(self.arg_key_start) : key_end]
        self._buffer = self._buffer[key_end + len(self.arg_key_end) :]
        self._pending_key = key


OpenAIServingChat._create_remaining_args_delta = staticmethod(_create_remaining_args_delta)
OpenAIServingChat._record_streamed_tool_args = staticmethod(_record_streamed_tool_args)
OpenAIServingChat._compact_json_fragment = staticmethod(_compact_json_fragment)
OpenAIServingChat._compute_remaining_tool_args = classmethod(_compute_remaining_tool_args)
if not hasattr(OpenAIServingChat, "_make_usage_info"):
    raise RuntimeError("patch_glm_tool_call_parser requires the MiniMax usage-accounting patch to be applied first.")
_patched_chat_completion_stream_generator.__module__ = OpenAIServingChat.__module__
_patched_chat_completion_stream_generator.__qualname__ = (
    f"{OpenAIServingChat.__qualname__}.chat_completion_stream_generator"
)
OpenAIServingChat.chat_completion_stream_generator = _patched_chat_completion_stream_generator
Glm4MoeModelToolParser.extract_tool_calls_streaming = _patched_extract_tool_calls_streaming
