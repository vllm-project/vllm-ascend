# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
from vllm_ascend.patch.platform.patch_glm_tool_call_parser import (
    _patched_chat_completion_stream_generator,
)


class FakeTokenizer:
    def get_vocab(self):
        return {
            "<tool_call>": 1,
            "</tool_call>": 2,
            "<arg_key>": 3,
            "</arg_key>": 4,
            "<arg_value>": 5,
            "</arg_value>": 6,
        }


def _reset_streaming_state(parser):
    parser._buffer = ""
    parser._in_tool_call = False
    parser.current_tool_name_sent = False
    parser._current_tool_name = None
    parser._pending_key = None
    parser._streaming_string_value = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []
    parser._args_started = []
    parser._args_closed = []
    parser._seen_keys = []


def test_create_remaining_args_delta_uses_fallback_metadata_for_args_only_delta():
    original_delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='{"files":['),
            )
        ]
    )

    result = OpenAIServingChat._create_remaining_args_delta(
        original_delta,
        '{"files":[{"filepath":"HumanEval-X/README.md"}]}',
        0,
        fallback_tool_call_id="call_files",
        fallback_tool_call_type="function",
        fallback_tool_call_name="builtin_read_many_files",
    )

    tc = result.tool_calls[0]
    assert tc.index == 0
    assert tc.id == "call_files"
    assert tc.type == "function"
    assert tc.function.name == "builtin_read_many_files"
    assert tc.function.arguments == ('{"files":[{"filepath":"HumanEval-X/README.md"}]}')


def test_create_remaining_args_delta_uses_fallback_over_original_delta():
    # _create_remaining_args_delta ignores original_delta metadata and uses
    # the explicit fallback_* parameters instead.  The caller is responsible
    # for passing non-None fallback values only for the first chunk of a
    # tool call (when the header has not yet been streamed).
    original_delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                id="call_current",
                type="function",
                function=DeltaFunctionCall(
                    name="current_name",
                    arguments='{"files":[',
                ),
            )
        ]
    )

    result = OpenAIServingChat._create_remaining_args_delta(
        original_delta,
        "]}",
        0,
        fallback_tool_call_id="call_fallback",
        fallback_tool_call_type="function",
        fallback_tool_call_name="fallback_name",
    )

    tc = result.tool_calls[0]
    assert tc.id == "call_fallback"
    assert tc.type == "function"
    assert tc.function.name == "fallback_name"
    assert tc.function.arguments == "]}"


def test_record_streamed_tool_args_tracks_emitted_bytes():
    streamed_tool_args = {0: '{"files":['}
    delta_message = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments='{"filepath":"HumanEval-X/README.md"}]}'),
            )
        ]
    )

    OpenAIServingChat._record_streamed_tool_args(delta_message, streamed_tool_args)

    assert streamed_tool_args[0] == ('{"files":[{"filepath":"HumanEval-X/README.md"}]}')


def test_compute_remaining_tool_args_handles_compact_prefix():
    remaining = OpenAIServingChat._compute_remaining_tool_args(
        expected_args={"a": 1},
        streamed_args='{"a":1',
    )

    assert remaining == "}"


def test_compute_remaining_tool_args_handles_stringified_expected_args():
    remaining = OpenAIServingChat._compute_remaining_tool_args(
        expected_args='{"a":1}',
        streamed_args='{"a":1',
    )

    assert remaining == "}"


def test_compute_remaining_tool_args_handles_glm_mixed_whitespace_prefix():
    expected_args = {
        "todos": [
            {
                "content": "A",
                "activeForm": "B",
                "status": "in_progress",
            }
        ]
    }

    remaining = OpenAIServingChat._compute_remaining_tool_args(
        expected_args=expected_args,
        streamed_args=('{"todos":[{"content": "A", "activeForm": "B", "status": "in_progress"}]'),
    )

    assert remaining == "}"


def test_compute_remaining_tool_args_backfills_missing_suffix_for_glm_partial_prefix():
    expected_args = {
        "todos": [
            {
                "content": "A",
                "activeForm": "B",
                "status": "in_progress",
            }
        ]
    }

    remaining = OpenAIServingChat._compute_remaining_tool_args(
        expected_args=expected_args,
        streamed_args='{"todos":[{"content": "A"',
    )

    assert remaining == ',"activeForm":"B","status":"in_progress"}]}'


def test_compute_remaining_tool_args_returns_empty_for_non_matching_prefix():
    remaining = OpenAIServingChat._compute_remaining_tool_args(
        expected_args={"a": 1},
        streamed_args="not-json",
    )

    assert remaining == ""


def test_compute_remaining_tool_args_returns_full_call_when_no_args_were_sent():
    remaining = OpenAIServingChat._compute_remaining_tool_args(
        expected_args={
            "todos": "- [x] 分析项目结构和代码\n- [ ] 添加单元测试框架",
        },
        streamed_args="",
    )

    assert remaining == ('{"todos": "- [x] 分析项目结构和代码\\n- [ ] 添加单元测试框架"}')


def test_glm_streaming_final_chunk_emits_inline_string_value():
    parser = Glm4MoeModelToolParser(FakeTokenizer())
    _reset_streaming_state(parser)

    request = ChatCompletionRequest(
        model="zai-org/GLM-4.7",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "builtin_get_problems",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string"},
                        },
                    },
                },
            }
        ],
    )

    chunks = [
        "<tool_call>",
        "builtin_get_problems\n",
        "<arg_key>filepath</arg_key>",
        "<arg_value>pong.py</arg_value></tool_call>",
    ]

    last_tool_delta = None
    for chunk in chunks:
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if result is not None and result.tool_calls:
            last_tool_delta = result

    assert last_tool_delta is not None
    assert last_tool_delta.tool_calls[0].function.arguments == '{"filepath":"pong.py"}'
    assert parser.streamed_args_for_tool == ['{"filepath":"pong.py"}']
    assert parser.prev_tool_call_arr == [
        {
            "name": "builtin_get_problems",
            "arguments": {"filepath": "pong.py"},
        }
    ]


def test_glm47_streaming_delta_serializes_tool_call_fields():
    parser = Glm47MoeModelToolParser(FakeTokenizer())
    _reset_streaming_state(parser)

    request = ChatCompletionRequest(
        model="GLM-5",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "builtin_get_problems",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string"},
                        },
                    },
                },
            }
        ],
    )

    chunks = [
        "<tool_call>",
        "builtin_get_problems\n",
        "<arg_key>filepath</arg_key>",
        "<arg_value>pong.py</arg_value></tool_call>",
    ]

    serialized_deltas = []
    for chunk in chunks:
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if result is None:
            continue

        choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=result,
            logprobs=None,
            finish_reason=None,
        )
        response = ChatCompletionStreamResponse(
            id="chatcmpl-test",
            created=0,
            model="GLM-5",
            choices=[choice],
        )
        serialized_deltas.append(response.model_dump(exclude_unset=True)["choices"][0]["delta"])

    assert len(serialized_deltas) == 2
    assert serialized_deltas[0]["tool_calls"][0]["type"] == "function"
    assert serialized_deltas[0]["tool_calls"][0]["function"]["name"] == "builtin_get_problems"
    assert serialized_deltas[-1] != {}
    assert serialized_deltas[-1]["tool_calls"][0]["index"] == 0
    assert serialized_deltas[-1]["tool_calls"][0]["function"]["arguments"] == '{"filepath":"pong.py"}'


# ---------------------------------------------------------------------------
# Helpers shared by streaming-split tests
# ---------------------------------------------------------------------------

def _make_serving_mock(*, use_harmony: bool = False) -> MagicMock:
    """Return a minimal mock of OpenAIServingChat for streaming tests."""
    serving = MagicMock()
    serving.use_harmony = use_harmony
    serving.tool_parser = None
    serving.tool_call_id_type = "standard"
    serving.enable_log_outputs = False
    serving.enable_log_deltas = False
    serving.enable_force_include_usage = False
    serving._should_stream_with_auto_tool_parsing.return_value = False
    serving.get_chat_request_role.return_value = "assistant"
    serving._raise_if_error.return_value = None
    serving._should_check_for_unstreamed_tool_arg_tokens.return_value = False
    serving._make_usage_info.return_value = MagicMock()
    serving._count_reasoning_tokens_for_usage.return_value = None
    return serving


def _make_request_mock(*, tool_choice: str = "required") -> MagicMock:
    request = MagicMock()
    request.n = 1
    request.logprobs = False
    request.top_logprobs = None
    request.return_token_ids = False
    request.return_tokens_as_token_ids = False
    request.include_reasoning = False
    request.echo = False
    request._grammar_from_tool_parser = False
    request.tool_choice = tool_choice
    request.stream_options = None
    request.parallel_tool_calls = True
    request.tools = []
    return request


class _MockOutput:
    def __init__(self, *, token_ids=(1,), text="", finish_reason=None, stop_reason=None):
        self.index = 0
        self.token_ids = token_ids
        self.text = text
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason
        self.logprobs = None


class _MockResult:
    def __init__(self, outputs, prompt_token_ids=(1, 2, 3)):
        self.outputs = outputs
        self.prompt_token_ids = prompt_token_ids
        self.encoder_prompt_token_ids = None
        self.num_cached_tokens = None


async def _run_stream(serving, request, outputs):
    """Drive the patched generator and return parsed non-DONE data chunks."""
    async def _gen():
        for out in outputs:
            yield out

    chunks = []
    async for item in _patched_chat_completion_stream_generator(
        serving,
        request=request,
        result_generator=_gen(),
        request_id="test-req-id",
        model_name="glm-5",
        conversation=[],
        tokenizer=MagicMock(),
        request_metadata=MagicMock(),
        reasoning_parser=None,
    ):
        if item.startswith("data: ") and item.strip() != "data: [DONE]":
            data = json.loads(item[6:].strip())
            if data.get("choices"):
                chunks.append(data)
    return chunks


def _assert_split_invariant(chunks):
    """Assert that no chunk has both tool_calls delta data AND a non-null finish_reason."""
    for chunk in chunks:
        for choice in chunk["choices"]:
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")
            tool_calls = delta.get("tool_calls", [])
            assert not (tool_calls and finish_reason is not None), (
                f"Chunk carries both tool_calls data and finish_reason={finish_reason!r}: {chunk}"
            )


# ---------------------------------------------------------------------------
# Streaming-split tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_final_tool_calls_chunk_is_split_from_finish_reason():
    """
    The last tool_calls arguments delta and finish_reason='tool_calls' must
    be delivered in separate chunks (OpenAI streaming spec invariant).
    """
    serving = _make_serving_mock()
    request = _make_request_mock(tool_choice="required")

    final_delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                id="call-123",
                type="function",
                function=DeltaFunctionCall(name="detect_text", arguments="}"),
            )
        ]
    )
    serving.extract_tool_call_required_streaming.return_value = (final_delta, True)

    outputs = [_MockResult([_MockOutput(finish_reason="stop")])]
    chunks = await _run_stream(serving, request, outputs)

    _assert_split_invariant(chunks)

    tool_call_chunks = [c for c in chunks if any(ch["delta"].get("tool_calls") for ch in c["choices"])]
    finish_chunks = [c for c in chunks if any(ch.get("finish_reason") == "tool_calls" for ch in c["choices"])]

    assert len(tool_call_chunks) >= 1, "Must have at least one chunk with tool_calls data"
    assert len(finish_chunks) == 1, "Must have exactly one chunk with finish_reason='tool_calls'"

    data_idx = chunks.index(tool_call_chunks[-1])
    finish_idx = chunks.index(finish_chunks[0])
    assert finish_idx > data_idx, "finish_reason chunk must come after the last data chunk"

    # The finish chunk delta must be empty ({})
    finish_delta = finish_chunks[0]["choices"][0].get("delta", {})
    assert not finish_delta.get("tool_calls"), "finish chunk delta must have no tool_calls"
    assert not finish_delta.get("content"), "finish chunk delta must have no content"


@pytest.mark.asyncio
async def test_streaming_multi_chunk_tool_args_split_only_at_finish():
    """
    When arguments arrive across several intermediate chunks, only the
    final chunk (carrying the closing '}') triggers the split.  Intermediate
    chunks must NOT be split.
    """
    serving = _make_serving_mock()
    request = _make_request_mock(tool_choice="required")

    # Intermediate chunk: first part of arguments, no finish_reason
    intermediate_delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                id="call-456",
                type="function",
                function=DeltaFunctionCall(name="search", arguments='{"q":"foo"'),
            )
        ]
    )
    # Final chunk: closing brace + finish_reason
    final_delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments="}"),
            )
        ]
    )
    serving.extract_tool_call_required_streaming.side_effect = [
        (intermediate_delta, True),   # first call → no finish_reason
        (final_delta, True),          # second call → finish_reason="stop"
    ]

    outputs = [
        _MockResult([_MockOutput(finish_reason=None)]),   # intermediate
        _MockResult([_MockOutput(finish_reason="stop")]), # final
    ]
    chunks = await _run_stream(serving, request, outputs)

    _assert_split_invariant(chunks)

    finish_chunks = [c for c in chunks if any(ch.get("finish_reason") == "tool_calls" for ch in c["choices"])]
    assert len(finish_chunks) == 1


@pytest.mark.asyncio
async def test_streaming_multiple_tool_calls_split_invariant():
    """
    Multi-tool-call scenario: the invariant holds regardless of how many
    tool calls are present in the final delta.
    """
    serving = _make_serving_mock()
    request = _make_request_mock(tool_choice="required")

    final_delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                id="call-a",
                type="function",
                function=DeltaFunctionCall(name="tool_a", arguments='{"x":1}'),
            ),
            DeltaToolCall(
                index=1,
                id="call-b",
                type="function",
                function=DeltaFunctionCall(name="tool_b", arguments='{"y":2}'),
            ),
        ]
    )
    serving.extract_tool_call_required_streaming.return_value = (final_delta, True)

    outputs = [_MockResult([_MockOutput(finish_reason="stop")])]
    chunks = await _run_stream(serving, request, outputs)

    _assert_split_invariant(chunks)

    finish_chunks = [c for c in chunks if any(ch.get("finish_reason") == "tool_calls" for ch in c["choices"])]
    assert len(finish_chunks) == 1


@pytest.mark.asyncio
async def test_streaming_plain_text_not_split():
    """
    Plain text streaming (no tool calls) is NOT affected by the split.
    The finish chunk may still carry content with finish_reason='stop'.
    """
    serving = _make_serving_mock()
    # No tool_choice → pure text path
    request = _make_request_mock(tool_choice="none")
    request.tool_choice = "none"  # ensure it doesn't hit required/auto paths

    outputs = [_MockResult([_MockOutput(text="Hello!", finish_reason="stop")])]
    chunks = await _run_stream(serving, request, outputs)

    finish_chunks = [c for c in chunks if any(ch.get("finish_reason") == "stop" for ch in c["choices"])]
    assert len(finish_chunks) >= 1, "Should have a chunk with finish_reason='stop'"

    # No spurious extra empty-delta chunk should appear for plain text
    tool_call_chunks = [c for c in chunks if any(ch["delta"].get("tool_calls") for ch in c["choices"])]
    assert len(tool_call_chunks) == 0, "Plain text must not produce tool_calls chunks"
