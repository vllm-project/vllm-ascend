# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser import ParserManager
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

from vllm_ascend.patch.platform import patch_kimi_k3_parsers as parser_patch
from vllm_ascend.patch.platform.kimi_k3_xtml import (
    CLOSE_TOKEN,
    OPEN_TOKEN,
    KimiK3XTMLParseError,
)
from vllm_ascend.patch.platform.patch_kimi_k3_parsers import (
    ARGUMENT_END,
    CALL_END,
    END_OF_MSG_TOKEN,
    JSON_END,
    MESSAGE_END,
    RESPONSE_END,
    RESPONSE_START,
    SEP_TOKEN,
    THINK_END,
    THINK_START,
    TOOLS_END,
    TOOLS_START,
    KimiK3Parser,
    KimiK3ReasoningParser,
    KimiK3ToolParser,
)
from vllm_ascend.patch.platform.patch_kimi_k3_renderer import (
    prepare_kimi_k3_chat_template_kwargs,
)


class FakeTokenizer:
    CONTROL_IDS = {
        OPEN_TOKEN: 1000000,
        CLOSE_TOKEN: 1000001,
        SEP_TOKEN: 1000002,
        END_OF_MSG_TOKEN: 1000003,
    }

    def __init__(self):
        self.model = SimpleNamespace(
            decode_single_token_bytes=lambda token_id: chr(token_id).encode(),
        )

    def encode(self, text):
        token_ids = []
        index = 0
        while index < len(text):
            marker = next((marker for marker in self.CONTROL_IDS if text.startswith(marker, index)), None)
            if marker is None:
                token_ids.append(ord(text[index]))
                index += 1
            else:
                token_ids.append(self.CONTROL_IDS[marker])
                index += len(marker)
        return token_ids

    def decode(self, token_ids, **kwargs):
        del kwargs
        markers = {token_id: marker for marker, token_id in self.CONTROL_IDS.items()}
        return "".join(markers.get(token_id, chr(token_id)) for token_id in token_ids)

    def convert_tokens_to_ids(self, token):
        return self.CONTROL_IDS[token]

    def get_vocab(self):
        return {}


TOKENIZER = FakeTokenizer()


class ProvenanceTokenizer(FakeTokenizer):
    CONTROL_IDS = {
        OPEN_TOKEN: 1000,
        CLOSE_TOKEN: 1001,
        SEP_TOKEN: 1002,
        END_OF_MSG_TOKEN: 1003,
    }

    def __init__(self):
        self._next_token_id = 2000
        self._token_bytes: dict[int, bytes] = {}
        self.model = SimpleNamespace(
            decode_single_token_bytes=self._decode_single_token_bytes,
        )

    def convert_tokens_to_ids(self, token):
        return self.CONTROL_IDS[token]

    def text_token(self, text: str) -> int:
        token_id = self._next_token_id
        self._next_token_id += 1
        self._token_bytes[token_id] = text.encode()
        return token_id

    def _decode_single_token_bytes(self, token_id: int) -> bytes:
        return self._token_bytes[token_id]


def _structure(tokenizer: ProvenanceTokenizer, marker: str, tag: str = "") -> list[int]:
    token_ids = [tokenizer.convert_tokens_to_ids(marker)]
    if tag:
        token_ids.append(tokenizer.text_token(tag))
    return token_ids


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "plan_trip",
                "description": "Plan a trip.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "number"},
                        "flexible": {"type": "boolean"},
                        "metadata": {"type": "object"},
                        "stops": {"type": "array"},
                        "note": {"type": ["string", "null"]},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def _argument(key: str, value_type: str, value: str) -> str:
    return f'<|open|>argument key="{key}" type="{value_type}"{SEP_TOKEN}{value}{ARGUMENT_END}'


def _call(name: str, arguments: str = "", index: int = 1) -> str:
    return f'<|open|>call tool="{name}" index="{index}"{SEP_TOKEN}{arguments}{CALL_END}'


def _tool_output(*calls: str, response: str = "") -> str:
    tools = f"{TOOLS_START}{''.join(calls)}{TOOLS_END}" if calls else ""
    return f"{RESPONSE_START}{response}{RESPONSE_END}{tools}{MESSAGE_END}{END_OF_MSG_TOKEN}"


def _request(**kwargs):
    defaults = {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "help"}],
        "tools": _tools(),
        "tool_choice": "auto",
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _parser(*, thinking: bool):
    return KimiK3Parser(
        TOKENIZER,
        _tools(),
        chat_template_kwargs={"thinking": thinking},
    )


def _parse(parser, model_output: str, request: ChatCompletionRequest, **kwargs):
    return parser.parse(
        model_output,
        request,
        model_output_token_ids=parser.model_tokenizer.encode(model_output),
        **kwargs,
    )


def _token_chunks(text: str, chunk_size: int):
    token_ids = TOKENIZER.encode(text)
    for start in range(0, len(token_ids), chunk_size):
        delta_ids = token_ids[start : start + chunk_size]
        yield TOKENIZER.decode(delta_ids), delta_ids, start + chunk_size >= len(token_ids)


def _encoded_output(
    tokenizer: ProvenanceTokenizer,
    *,
    content: str = "",
    tool_name: str | None = None,
    arguments: str | None = None,
    complete: bool = True,
) -> tuple[str, list[int]]:
    text_parts: list[str] = []
    token_ids: list[int] = []

    def control(marker: str, tag: str = "") -> None:
        text_parts.append(marker + tag)
        token_ids.extend(_structure(tokenizer, marker, tag))

    def text(value: str) -> None:
        if value:
            text_parts.append(value)
            token_ids.append(tokenizer.text_token(value))

    control(OPEN_TOKEN, "response")
    control(SEP_TOKEN)
    text(content)
    control(CLOSE_TOKEN, "response")
    control(SEP_TOKEN)
    if tool_name is not None:
        control(OPEN_TOKEN, "tools")
        control(SEP_TOKEN)
        control(OPEN_TOKEN, f'call tool="{tool_name}" index="1"')
        control(SEP_TOKEN)
        if arguments is not None:
            control(OPEN_TOKEN, 'json type="object"')
            control(SEP_TOKEN)
            text(arguments)
            if complete:
                control(CLOSE_TOKEN, "json")
                control(SEP_TOKEN)
        if not complete:
            return "".join(text_parts), token_ids
        control(CLOSE_TOKEN, "call")
        control(SEP_TOKEN)
        control(CLOSE_TOKEN, "tools")
        control(SEP_TOKEN)
    control(CLOSE_TOKEN, "message")
    control(SEP_TOKEN)
    control(END_OF_MSG_TOKEN)
    return "".join(text_parts), token_ids


def _chat_serving() -> OpenAIServingChat:
    serving = object.__new__(OpenAIServingChat)
    serving.parser_cls = KimiK3Parser
    serving.tool_parser = KimiK3ToolParser
    serving.tool_call_id_type = "random"
    serving.response_role = "assistant"
    serving.use_harmony = False
    serving.enable_auto_tools = True
    serving.enable_force_include_usage = False
    serving.enable_prompt_tokens_details = False
    serving.system_fingerprint = None
    serving.enable_log_outputs = False
    serving.enable_log_deltas = False
    serving.request_logger = None
    return serving


def _serve(
    request: ChatCompletionRequest,
    tokenizer: FakeTokenizer,
    text: str,
    token_ids: list[int],
    finish_reason: str,
    *,
    serving: OpenAIServingChat | None = None,
    use_parser: bool = True,
    completion_outputs: list[CompletionOutput] | None = None,
    kv_transfer_params: dict | None = None,
):
    if completion_outputs is None:
        completion_outputs = [
            CompletionOutput(
                index=0,
                text=text,
                token_ids=token_ids,
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=finish_reason,
            )
        ]
    result = RequestOutput(
        request_id="test-request",
        prompt="prompt",
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=completion_outputs,
        finished=True,
        kv_transfer_params=kv_transfer_params,
    )
    serving = serving or _chat_serving()
    metadata = RequestResponseMetadata(request_id="test-request")

    async def results():
        yield result

    async def run():
        if request.stream:
            chunks = [
                chunk
                async for chunk in serving.chat_completion_stream_generator(
                    request,
                    results(),
                    "test-request",
                    request.model,
                    [],
                    tokenizer,
                    metadata,
                    chat_template_kwargs={"thinking": False},
                )
            ]
            return [
                json.loads(chunk.removeprefix("data: ").removesuffix("\n\n"))
                for chunk in chunks
                if chunk != "data: [DONE]\n\n"
            ]

        parser = (
            KimiK3Parser(
                tokenizer,
                request.tools,
                chat_template_kwargs={"thinking": False},
            )
            if use_parser
            else None
        )
        return await serving.chat_completion_full_generator(
            request,
            results(),
            "test-request",
            request.model,
            [],
            tokenizer,
            metadata,
            parser,
        )

    return asyncio.run(run())


def test_kimi_k3_parsers_are_registered_and_unified():
    assert ReasoningParserManager.get_reasoning_parser("kimi_k3") is KimiK3ReasoningParser
    assert ToolParserManager.get_tool_parser("kimi_k3") is KimiK3ToolParser
    assert (
        ParserManager.get_parser(
            tool_parser_name="kimi_k3",
            reasoning_parser_name="kimi_k3",
            enable_auto_tools=True,
            model_name="kimi-k3",
        )
        is KimiK3Parser
    )


def test_serving_full_forwards_token_provenance_for_literal_xtml():
    tokenizer = ProvenanceTokenizer()
    literal = RESPONSE_END + TOOLS_START + _call("get_time") + TOOLS_END
    text, token_ids = _encoded_output(tokenizer, content=literal)
    request = _request(
        tools=None,
        tool_choice="none",
        reasoning_effort="none",
    )

    response = _serve(
        request,
        tokenizer,
        text,
        token_ids,
        "stop",
    )

    assert not isinstance(response, ErrorResponse)
    assert response.choices[0].message.content == literal
    assert response.choices[0].message.tool_calls == []


def test_serving_full_skips_parser_for_remote_decode(monkeypatch):
    request = _request(
        kv_transfer_params={
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }
    )
    returned_transfer_params = {
        "remote_engine_id": "prefill-engine",
        "remote_block_ids": [[1, 2]],
    }

    def fail_parse(*args, **kwargs):
        del args, kwargs
        pytest.fail("Kimi K3 parser must not run for remote decode requests")

    monkeypatch.setattr(KimiK3Parser, "parse", fail_parse)
    response = _serve(
        request,
        TOKENIZER,
        "internal-prefill-token",
        TOKENIZER.encode("internal-prefill-token"),
        "length",
        kv_transfer_params=returned_transfer_params,
    )

    assert not isinstance(response, ErrorResponse)
    assert response.kv_transfer_params == returned_transfer_params


def test_serving_full_keeps_parser_for_remote_prefill():
    tokenizer = ProvenanceTokenizer()
    text, token_ids = _encoded_output(tokenizer, content="decode-output")
    request = _request(
        tools=None,
        tool_choice="none",
        reasoning_effort="none",
        kv_transfer_params={
            "do_remote_decode": False,
            "do_remote_prefill": True,
        },
    )

    response = _serve(request, tokenizer, text, token_ids, "stop")

    assert not isinstance(response, ErrorResponse)
    assert response.choices[0].message.content == "decode-output"


@pytest.mark.parametrize(
    ("tool_choice", "tool_name"),
    [
        ("required", "plan_trip"),
        ({"type": "function", "function": {"name": "get_time"}}, "get_time"),
    ],
)
def test_serving_full_preserves_required_and_named_content(
    tool_choice,
    tool_name,
):
    tokenizer = ProvenanceTokenizer()
    text, token_ids = _encoded_output(
        tokenizer,
        content="Calling the selected tool.",
        tool_name=tool_name,
    )
    request = _request(
        tool_choice=tool_choice,
        reasoning_effort="none",
    )

    response = _serve(
        request,
        tokenizer,
        text,
        token_ids,
        "stop",
    )

    assert not isinstance(response, ErrorResponse)
    message = response.choices[0].message
    assert message.content == "Calling the selected tool."
    assert [call.function.name for call in message.tool_calls] == [tool_name]


@pytest.mark.parametrize(
    ("engine_finish_reason", "expected"),
    [
        ("stop", "tool_calls"),
        ("length", "length"),
    ],
)
def test_serving_full_preserves_engine_finish_reason(
    engine_finish_reason,
    expected,
):
    tokenizer = ProvenanceTokenizer()
    text, token_ids = _encoded_output(tokenizer, tool_name="get_time")
    request = _request(reasoning_effort="none")

    response = _serve(
        request,
        tokenizer,
        text,
        token_ids,
        engine_finish_reason,
    )

    assert not isinstance(response, ErrorResponse)
    assert response.choices[0].finish_reason == expected


def test_serving_full_tracks_token_ids_and_finish_reasons_by_output_index():
    tokenizer = ProvenanceTokenizer()
    tool_text, tool_token_ids = _encoded_output(tokenizer, tool_name="get_time")
    literal = RESPONSE_END + TOOLS_START + _call("get_time") + TOOLS_END
    content_text, content_token_ids = _encoded_output(tokenizer, content=literal)
    request = _request(reasoning_effort="none", n=2)
    outputs = [
        CompletionOutput(
            index=4,
            text=tool_text,
            token_ids=tool_token_ids,
            cumulative_logprob=0.0,
            logprobs=None,
            finish_reason="length",
        ),
        CompletionOutput(
            index=1,
            text=content_text,
            token_ids=content_token_ids,
            cumulative_logprob=0.0,
            logprobs=None,
            finish_reason="stop",
        ),
    ]

    response = _serve(
        request,
        tokenizer,
        "",
        [],
        "stop",
        completion_outputs=outputs,
    )

    assert not isinstance(response, ErrorResponse)
    choices = {choice.index: choice for choice in response.choices}
    assert choices[4].finish_reason == "length"
    assert [call.function.name for call in choices[4].message.tool_calls] == ["get_time"]
    assert choices[1].finish_reason == "stop"
    assert choices[1].message.content == literal
    assert choices[1].message.tool_calls == []


@pytest.mark.parametrize(
    ("engine_finish_reason", "tool_name", "arguments", "complete", "expected"),
    [
        ("stop", "get_time", None, True, "tool_calls"),
        ("length", "get_time", None, False, "length"),
        ("length", "plan_trip", '{"city":"Par', False, "length"),
    ],
)
def test_serving_stream_preserves_engine_finish_reason(
    engine_finish_reason,
    tool_name,
    arguments,
    complete,
    expected,
):
    tokenizer = ProvenanceTokenizer()
    text, token_ids = _encoded_output(
        tokenizer,
        tool_name=tool_name,
        arguments=arguments,
        complete=complete,
    )
    request = _request(
        stream=True,
        reasoning_effort="none",
    )

    payloads = _serve(
        request,
        tokenizer,
        text,
        token_ids,
        engine_finish_reason,
    )

    choices = [choice for payload in payloads for choice in payload["choices"]]
    assert any(choice["delta"].get("tool_calls") for choice in choices)
    terminal = [choice for choice in choices if choice.get("finish_reason") is not None]
    assert [choice["finish_reason"] for choice in terminal] == [expected]


def test_serving_content_only_stream_remains_normal():
    tokenizer = ProvenanceTokenizer()
    text, token_ids = _encoded_output(tokenizer, content="No tool needed.")
    request = _request(
        stream=True,
        reasoning_effort="none",
    )

    payloads = _serve(
        request,
        tokenizer,
        text,
        token_ids,
        "stop",
    )

    choices = [choice for payload in payloads for choice in payload["choices"]]
    assert "".join(choice["delta"].get("content", "") for choice in choices) == "No tool needed."
    terminal = [choice for choice in choices if choice.get("finish_reason") is not None]
    assert [choice["finish_reason"] for choice in terminal] == ["stop"]


def test_finish_reason_restore_skips_ordinary_sse_json(monkeypatch):
    data = 'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}\n\n'

    def fail_json_loads(_payload):
        raise AssertionError("ordinary SSE chunks must not be decoded")

    monkeypatch.setattr(parser_patch.json, "loads", fail_json_loads)

    assert parser_patch._restore_engine_finish_reason(data, {0: "length"}) == data
    assert parser_patch._restore_engine_finish_reason(data, {}) == data
    assert parser_patch._restore_engine_finish_reason("data: [DONE]\n\n", {0: "length"}) == "data: [DONE]\n\n"


@pytest.mark.parametrize(
    ("engine_reason", "expected_reason"),
    [
        ("length", "length"),
        ("stop", "tool_calls"),
    ],
)
def test_finish_reason_restore_only_changes_non_stop_terminal_chunks(
    engine_reason,
    expected_reason,
):
    data = 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'

    restored = parser_patch._restore_engine_finish_reason(data, {0: engine_reason})

    payload = json.loads(restored.removeprefix("data: ").removesuffix("\n\n"))
    assert payload["choices"][0]["finish_reason"] == expected_reason


def test_finish_reason_restore_ignores_malformed_terminal_sse():
    data = 'data: {"finish_reason":"tool_calls"\n\n'

    assert parser_patch._restore_engine_finish_reason(data, {0: "length"}) == data


def test_kimi_k3_responses_are_rejected_before_generation():
    serving = object.__new__(OpenAIServingResponses)
    serving.parser = KimiK3Parser
    request = ResponsesRequest.model_validate(
        {
            "model": "kimi-k3",
            "input": "test",
        }
    )

    response = asyncio.run(serving._create_responses(request))

    assert isinstance(response, ErrorResponse)
    assert "Chat Completions" in response.error.message


def test_non_kimi_responses_delegate_to_original(monkeypatch):
    serving = object.__new__(OpenAIServingResponses)
    serving.parser = None
    expected = object()

    async def original(self, request, raw_request=None):
        return expected

    monkeypatch.setattr(
        OpenAIServingResponses,
        "_ascend_original_kimi_k3_create_responses",
        original,
    )
    request = ResponsesRequest.model_validate(
        {
            "model": "other-model",
            "input": "test",
        }
    )

    assert asyncio.run(serving._create_responses(request)) is expected


def test_non_kimi_chat_serving_behavior_is_unchanged():
    serving = _chat_serving()
    serving.parser_cls = None
    serving.tool_parser = None
    request = ChatCompletionRequest(
        model="other-model",
        messages=[{"role": "user", "content": "test"}],
    )
    text = "plain output"
    token_ids = TOKENIZER.encode(text)
    full_response = _serve(
        request,
        TOKENIZER,
        text,
        token_ids,
        "stop",
        serving=serving,
        use_parser=False,
    )
    request.stream = True
    payloads = _serve(
        request,
        TOKENIZER,
        text,
        token_ids,
        "stop",
        serving=serving,
        use_parser=False,
    )

    assert not isinstance(full_response, ErrorResponse)
    assert full_response.choices[0].message.content == text
    choices = [choice for payload in payloads for choice in payload["choices"]]
    assert "".join(choice["delta"].get("content", "") for choice in choices) == text
    assert [choice["finish_reason"] for choice in choices if choice.get("finish_reason") is not None] == ["stop"]


def test_nonempty_output_without_token_ids_is_rejected():
    with pytest.raises(KimiK3XTMLParseError, match="requires original model output token IDs"):
        _parser(thinking=False).parse(
            "plain output",
            _request(tools=None, tool_choice="none", reasoning_effort="none"),
        )


@pytest.mark.parametrize(
    ("tool_parser_name", "reasoning_parser_name", "enable_auto_tools"),
    [
        ("kimi_k3", None, True),
        (None, "kimi_k3", True),
        ("kimi_k3", "kimi_k3", False),
    ],
)
def test_kimi_k3_rejects_partial_parser_configuration(
    tool_parser_name,
    reasoning_parser_name,
    enable_auto_tools,
):
    with pytest.raises(ValueError, match="requires --enable-auto-tool-choice"):
        ParserManager.get_parser(
            tool_parser_name=tool_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            enable_auto_tools=enable_auto_tools,
            model_name="kimi-k3",
        )


def test_kimi_k3_adjusts_non_chat_requests_without_rejecting_them():
    parser = KimiK3ReasoningParser(
        TOKENIZER,
        chat_template_kwargs={"thinking": True},
    )
    request = SimpleNamespace(
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )

    assert parser.adjust_request(request) is request
    assert request.skip_special_tokens is False
    assert request.spaces_between_special_tokens is False


@pytest.mark.parametrize(
    ("request_kwargs", "expected_thinking"),
    [
        ({}, True),
        ({"reasoning_effort": "none"}, False),
        ({"reasoning_effort": "high"}, True),
    ],
)
def test_renderer_and_parser_share_canonical_thinking_state(
    request_kwargs,
    expected_thinking,
):
    request = _request(**request_kwargs)
    prepare_kimi_k3_chat_template_kwargs(request)
    parser = KimiK3Parser(
        TOKENIZER,
        request.tools,
        chat_template_kwargs=request.chat_template_kwargs,
    )

    assert request.chat_template_kwargs.get("thinking", True) is expected_thinking
    assert parser._thinking_enabled is expected_thinking


def test_non_streaming_reasoning_and_all_xtml_argument_types():
    arguments = "".join(
        [
            _argument("city", "string", "北京 & 海淀"),
            _argument("days", "number", "3"),
            _argument("flexible", "boolean", "false"),
            _argument("metadata", "object", '{"seat":"window"}'),
            _argument("stops", "array", '["上海","东京"]'),
            _argument("note", "null", "null"),
        ]
    )
    model_output = (
        "I should use the trip planner."
        + THINK_END
        + _tool_output(
            _call("plan_trip", arguments),
            response="I will check. ",
        )
    )

    reasoning, content, calls = _parse(
        _parser(thinking=True),
        model_output,
        _request(),
        enable_auto_tools=True,
    )

    assert reasoning == "I should use the trip planner."
    assert content == "I will check. "
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "plan_trip"
    assert json.loads(calls[0].arguments) == {
        "city": "北京 & 海淀",
        "days": 3,
        "flexible": False,
        "metadata": {"seat": "window"},
        "stops": ["上海", "东京"],
        "note": None,
    }


def test_multiple_calls_zero_arguments_and_json_block():
    raw_json = '{"city":"New York","days":2}'
    json_block = f'<|open|>json type="object"{SEP_TOKEN}{raw_json}{JSON_END}'
    output = _tool_output(
        _call("plan_trip", json_block),
        _call("get_time", index=2),
    )

    _, content, calls = _parse(
        _parser(thinking=False),
        output,
        _request(reasoning_effort="none"),
        enable_auto_tools=True,
    )

    assert content is None
    assert calls is not None
    assert [call.name for call in calls] == ["plan_trip", "get_time"]
    assert json.loads(calls[0].arguments) == {
        "city": "New York",
        "days": 2,
    }
    assert json.loads(calls[1].arguments) == {}


def test_tool_choice_none_preserves_request_and_only_cleans_envelope():
    request = _request(tool_choice="none", reasoning_effort="none")
    parser = _parser(thinking=False)

    reasoning_parser = KimiK3ReasoningParser(
        TOKENIZER,
        chat_template_kwargs={"thinking": False},
    )
    reasoning_parser.adjust_request(request)
    assert request.tool_choice == "none"
    assert request.skip_special_tokens is False
    assert request.spaces_between_special_tokens is False

    generated = _tool_output(
        _call("get_time"),
        response="No tool is needed.",
    )
    reasoning, content, calls = _parse(
        parser,
        generated,
        request,
        enable_auto_tools=True,
    )
    assert reasoning is None
    assert content == "No tool is needed."
    assert calls == []

    streamed_content: list[str] = []
    streamed_tool_calls = []
    parser = _parser(thinking=False)
    for chunk, delta_ids, finished in _token_chunks(generated, 3):
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=delta_ids,
            request=request,
            finished=finished,
        )
        if delta is not None:
            assert "tool_calls" not in delta.model_dump(exclude_unset=True)
            if delta.content:
                streamed_content.append(delta.content)
            streamed_tool_calls.extend(delta.tool_calls)

    assert "".join(streamed_content) == "No tool is needed."
    assert streamed_tool_calls == []


@pytest.mark.parametrize("tool_choice", ["none", None])
def test_tools_omitted_plain_chat_cleans_envelope_without_fallback(tool_choice):
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "help"}],
        tool_choice=tool_choice,
        reasoning_effort="none",
    )
    original_tool_choice = request.tool_choice
    prepare_kimi_k3_chat_template_kwargs(request)

    output = f"plain answer{RESPONSE_END}{MESSAGE_END}"
    reasoning, content, calls = _parse(
        _parser(thinking=False),
        output,
        request,
        enable_auto_tools=True,
    )

    assert request.tool_choice == original_tool_choice
    assert reasoning is None
    assert content == "plain answer"
    assert calls == []


def test_auto_content_only_is_a_normal_completion():
    output = f"no call needed{RESPONSE_END}{MESSAGE_END}"
    reasoning, content, calls = _parse(
        _parser(thinking=False),
        output,
        _request(reasoning_effort="none"),
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content == "no call needed"
    assert calls == []


def test_gpqa_and_mmmu_default_chat_replay_split_reasoning_and_content():
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "answer the question"}],
        reasoning_effort="high",
    )
    prepare_kimi_k3_chat_template_kwargs(request)

    for reasoning_text, answer in (
        ("GPQA reasoning trace", "Answer: D"),
        ("MMMU-Pro visual reasoning trace", "Answer: H"),
    ):
        generated = reasoning_text + THINK_END + RESPONSE_START + answer + RESPONSE_END + MESSAGE_END
        reasoning, content, calls = _parse(
            _parser(thinking=True),
            generated,
            request,
            enable_auto_tools=True,
        )

        assert reasoning == reasoning_text
        assert content == answer
        assert content
        assert calls == []
        assert "<|" not in reasoning + content


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "get_time"}},
    ],
)
def test_required_and_named_never_return_empty_success(tool_choice):
    request = _request(
        tool_choice=tool_choice,
        reasoning_effort="none",
    )
    with pytest.raises(KimiK3XTMLParseError, match="without a valid tool call"):
        output = _tool_output(response="I did not call anything.")
        _parse(
            _parser(thinking=False),
            output,
            request,
            enable_auto_tools=True,
        )


def test_named_choice_rejects_a_different_function():
    request = _request(
        tool_choice={"type": "function", "function": {"name": "get_time"}},
        reasoning_effort="none",
    )
    with pytest.raises(KimiK3XTMLParseError, match="requires 'get_time'"):
        output = _tool_output(_call("plan_trip"))
        _parse(
            _parser(thinking=False),
            output,
            request,
            enable_auto_tools=True,
        )


@pytest.mark.parametrize(
    ("tool_choice", "tool_name"),
    [
        ("required", "plan_trip"),
        ({"type": "function", "function": {"name": "get_time"}}, "get_time"),
    ],
)
def test_required_and_named_success_are_stream_full_equivalent(
    tool_choice,
    tool_name,
):
    request = _request(
        tool_choice=tool_choice,
        reasoning_effort="none",
    )
    generated = _tool_output(
        _call(tool_name),
        response="this response must be preserved",
    )

    _, full_content, full_calls = _parse(
        _parser(thinking=False),
        generated,
        request,
        enable_auto_tools=True,
    )

    parser = _parser(thinking=False)
    streamed_content: list[str] = []
    streamed_calls = []
    for chunk, delta_ids, finished in _token_chunks(generated, 2):
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=delta_ids,
            request=request,
            finished=finished,
        )
        if delta is not None:
            if delta.content:
                streamed_content.append(delta.content)
            streamed_calls.extend(delta.tool_calls)

    assert full_content == "this response must be preserved"
    assert "".join(streamed_content) == full_content
    assert full_calls is not None and [call.name for call in full_calls] == [tool_name]
    names = [call.function.name for call in streamed_calls if call.function.name]
    arguments = "".join(call.function.arguments or "" for call in streamed_calls if call.function is not None)
    assert names == [tool_name]
    assert json.loads(arguments) == {}
    assert streamed_calls[0].id is not None
    assert streamed_calls[0].type == "function"
    assert all(call.id is None for call in streamed_calls[1:])
    assert all(call.type is None for call in streamed_calls[1:])


def test_bfcl_case5_exact_native_output_is_extracted():
    tool = {
        "type": "function",
        "function": {
            "name": "solve_quadratic_equation",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "c": {"type": "number"},
                },
                "required": ["a", "b", "c"],
            },
        },
    }
    request = _request(
        tools=[tool],
        reasoning_effort="none",
    )
    arguments = "".join(
        [
            _argument("a", "number", "2"),
            _argument("b", "number", "6"),
            _argument("c", "number", "5"),
        ]
    )
    # K3's generation prompt already opens ``response``. The captured BFCL
    # output therefore begins with RESPONSE_END, not RESPONSE_START.
    generated = RESPONSE_END + TOOLS_START + _call("solve_quadratic_equation", arguments) + TOOLS_END + MESSAGE_END

    reasoning, content, calls = _parse(
        _parser(thinking=False),
        generated,
        request,
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content is None
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "solve_quadratic_equation"
    assert json.loads(calls[0].arguments) == {"a": 2, "b": 6, "c": 5}


@pytest.mark.parametrize(
    ("suffix", "message"),
    [
        (
            lambda: _call(
                "plan_trip",
                _argument("days", "number", "not-a-number"),
            ),
            "Invalid JSON",
        ),
        (lambda: _call("delete_everything"), "Unknown K3 tool"),
        (
            lambda: f'<|open|>call tool="plan_trip" index="1" index="1"{SEP_TOKEN}{CALL_END}',
            "Duplicate XTML attribute",
        ),
        (
            lambda: _call(
                "plan_trip",
                _argument("days", "number", "3") + _argument("days", "number", "4"),
            ),
            "Duplicate K3 argument key",
        ),
        (
            lambda: _call(
                "plan_trip",
                f'<|open|>json type="object"{SEP_TOKEN}{{"days":2}}{JSON_END}' + _argument("city", "string", "Paris"),
            ),
            "mixes json arguments",
        ),
        (
            lambda: _call(
                "plan_trip",
                f'<|open|>json type="object"{SEP_TOKEN}{{"days":2,"days":3}}{JSON_END}',
            ),
            "Duplicate JSON key",
        ),
        (
            lambda: _call(
                "plan_trip",
                f'<|open|>json type="object"{SEP_TOKEN}{{"days":NaN}}{JSON_END}',
            ),
            "Non-finite JSON number",
        ),
        (
            lambda: f'<|open|>call tool="plan_trip" index="01"{SEP_TOKEN}{CALL_END}',
            "canonical positive integer",
        ),
        (
            lambda: _call(
                "plan_trip",
                "stray text" + _argument("days", "number", "3"),
            ),
            "Unexpected text or tag",
        ),
    ],
)
def test_malformed_or_unknown_calls_raise_explicitly(suffix, message):
    with pytest.raises(KimiK3XTMLParseError, match=message):
        output = _tool_output(suffix(), response="unsafe")
        _parse(
            _parser(thinking=False),
            output,
            _request(reasoning_effort="none"),
            enable_auto_tools=True,
        )


def test_truncated_tools_block_preserves_completed_call():
    output = f"{RESPONSE_START}unsafe{RESPONSE_END}{TOOLS_START}{_call('get_time')}"
    reasoning, content, calls = _parse(
        _parser(thinking=False),
        output,
        _request(reasoning_effort="none"),
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content == "unsafe"
    assert calls is not None and [call.name for call in calls] == ["get_time"]
    assert json.loads(calls[0].arguments) == {}


def test_truncated_outer_envelope_preserves_completed_response():
    output = RESPONSE_START + "answer" + RESPONSE_END
    reasoning, content, calls = _parse(
        _parser(thinking=False),
        output,
        _request(reasoning_effort="none"),
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content == "answer"
    assert calls == []


def test_nonstream_uses_token_provenance_for_literal_xtml_text():
    tokenizer = ProvenanceTokenizer()
    literal = RESPONSE_END + TOOLS_START + _call("get_time") + TOOLS_END
    output = RESPONSE_START + literal + RESPONSE_END + MESSAGE_END + END_OF_MSG_TOKEN
    token_ids = (
        _structure(tokenizer, OPEN_TOKEN, "response")
        + _structure(tokenizer, SEP_TOKEN)
        + [tokenizer.text_token(literal)]
        + _structure(tokenizer, CLOSE_TOKEN, "response")
        + _structure(tokenizer, SEP_TOKEN)
        + _structure(tokenizer, CLOSE_TOKEN, "message")
        + _structure(tokenizer, SEP_TOKEN)
        + _structure(tokenizer, END_OF_MSG_TOKEN)
    )
    parser = KimiK3Parser(
        tokenizer,
        [],
        chat_template_kwargs={"thinking": False},
    )

    reasoning, content, calls = parser.parse(
        output,
        _request(tools=None, tool_choice="none", reasoning_effort="none"),
        model_output_token_ids=token_ids,
    )

    assert reasoning is None
    assert content == literal
    assert calls == []


def test_unexpected_real_control_token_is_not_emitted_as_content():
    tokenizer = ProvenanceTokenizer()
    output = RESPONSE_START + OPEN_TOKEN + "garbage"
    token_ids = (
        _structure(tokenizer, OPEN_TOKEN, "response")
        + _structure(tokenizer, SEP_TOKEN)
        + _structure(tokenizer, OPEN_TOKEN)
        + [tokenizer.text_token("garbage")]
    )
    parser = KimiK3Parser(
        tokenizer,
        [],
        chat_template_kwargs={"thinking": False},
    )

    with pytest.raises(KimiK3XTMLParseError, match="control token"):
        parser.parse(
            output,
            _request(tools=None, tool_choice="none", reasoning_effort="none"),
            model_output_token_ids=token_ids,
        )


def test_literal_argument_end_text_is_preserved_in_string_argument():
    tokenizer = ProvenanceTokenizer()
    literal = "before " + ARGUMENT_END + " after"
    arguments = _argument("city", "string", literal)
    output = _tool_output(_call("plan_trip", arguments))
    token_ids = (
        _structure(tokenizer, OPEN_TOKEN, "response")
        + _structure(tokenizer, SEP_TOKEN)
        + _structure(tokenizer, CLOSE_TOKEN, "response")
        + _structure(tokenizer, SEP_TOKEN)
        + _structure(tokenizer, OPEN_TOKEN, "tools")
        + _structure(tokenizer, SEP_TOKEN)
        + _structure(tokenizer, OPEN_TOKEN, 'call tool="plan_trip" index="1"')
        + _structure(tokenizer, SEP_TOKEN)
        + _structure(tokenizer, OPEN_TOKEN, 'argument key="city" type="string"')
        + _structure(tokenizer, SEP_TOKEN)
        + [tokenizer.text_token(literal)]
        + _structure(tokenizer, CLOSE_TOKEN, "argument")
        + _structure(tokenizer, SEP_TOKEN)
        + _structure(tokenizer, CLOSE_TOKEN, "call")
        + _structure(tokenizer, SEP_TOKEN)
        + _structure(tokenizer, CLOSE_TOKEN, "tools")
        + _structure(tokenizer, SEP_TOKEN)
        + _structure(tokenizer, CLOSE_TOKEN, "message")
        + _structure(tokenizer, SEP_TOKEN)
        + _structure(tokenizer, END_OF_MSG_TOKEN)
    )
    parser = KimiK3Parser(
        tokenizer,
        _tools(),
        chat_template_kwargs={"thinking": False},
    )

    _, _, calls = parser.parse(
        output,
        _request(reasoning_effort="none"),
        model_output_token_ids=token_ids,
    )

    assert calls is not None and len(calls) == 1
    assert json.loads(calls[0].arguments) == {"city": literal}


def test_every_protocol_prefix_is_serving_safe_when_truncated():
    tokenizer = ProvenanceTokenizer()
    pieces = [
        (OPEN_TOKEN, tokenizer.convert_tokens_to_ids(OPEN_TOKEN)),
        ("response", tokenizer.text_token("response")),
        (SEP_TOKEN, tokenizer.convert_tokens_to_ids(SEP_TOKEN)),
        ("checking", tokenizer.text_token("checking")),
        (CLOSE_TOKEN, tokenizer.convert_tokens_to_ids(CLOSE_TOKEN)),
        ("response", tokenizer.text_token("response")),
        (SEP_TOKEN, tokenizer.convert_tokens_to_ids(SEP_TOKEN)),
        (OPEN_TOKEN, tokenizer.convert_tokens_to_ids(OPEN_TOKEN)),
        ("tools", tokenizer.text_token("tools")),
        (SEP_TOKEN, tokenizer.convert_tokens_to_ids(SEP_TOKEN)),
        (OPEN_TOKEN, tokenizer.convert_tokens_to_ids(OPEN_TOKEN)),
        ('call tool="plan_trip" index="1"', tokenizer.text_token('call tool="plan_trip" index="1"')),
        (SEP_TOKEN, tokenizer.convert_tokens_to_ids(SEP_TOKEN)),
        (OPEN_TOKEN, tokenizer.convert_tokens_to_ids(OPEN_TOKEN)),
        ('argument key="city" type="string"', tokenizer.text_token('argument key="city" type="string"')),
        (SEP_TOKEN, tokenizer.convert_tokens_to_ids(SEP_TOKEN)),
        ("Paris", tokenizer.text_token("Paris")),
        (CLOSE_TOKEN, tokenizer.convert_tokens_to_ids(CLOSE_TOKEN)),
        ("argument", tokenizer.text_token("argument")),
        (SEP_TOKEN, tokenizer.convert_tokens_to_ids(SEP_TOKEN)),
        (CLOSE_TOKEN, tokenizer.convert_tokens_to_ids(CLOSE_TOKEN)),
        ("call", tokenizer.text_token("call")),
        (SEP_TOKEN, tokenizer.convert_tokens_to_ids(SEP_TOKEN)),
        (CLOSE_TOKEN, tokenizer.convert_tokens_to_ids(CLOSE_TOKEN)),
        ("tools", tokenizer.text_token("tools")),
        (SEP_TOKEN, tokenizer.convert_tokens_to_ids(SEP_TOKEN)),
        (CLOSE_TOKEN, tokenizer.convert_tokens_to_ids(CLOSE_TOKEN)),
        ("message", tokenizer.text_token("message")),
        (SEP_TOKEN, tokenizer.convert_tokens_to_ids(SEP_TOKEN)),
        (END_OF_MSG_TOKEN, tokenizer.convert_tokens_to_ids(END_OF_MSG_TOKEN)),
    ]
    request = _request(reasoning_effort="none")

    for end in range(len(pieces) + 1):
        parser = KimiK3Parser(
            tokenizer,
            _tools(),
            chat_template_kwargs={"thinking": False},
        )
        parser.parse(
            "".join(text for text, _ in pieces[:end]),
            request,
            enable_auto_tools=True,
            model_output_token_ids=[token_id for _, token_id in pieces[:end]],
        )


@pytest.mark.parametrize(
    ("thinking", "output", "expected_reasoning", "expected_content"),
    [
        (True, "truncated reasoning", "truncated reasoning", None),
        (False, "truncated response", None, "truncated response"),
    ],
)
def test_delimiter_free_prefix_remains_available_for_length_truncation(
    thinking,
    output,
    expected_reasoning,
    expected_content,
):
    reasoning, content, calls = _parse(
        _parser(thinking=thinking),
        output,
        _request(
            tools=None,
            tool_choice="none",
            reasoning_effort="max" if thinking else "none",
        ),
        enable_auto_tools=True,
    )

    assert reasoning == expected_reasoning
    assert content == expected_content
    assert calls == []


def test_duplicate_call_index_and_multiple_tools_blocks_are_rejected():
    duplicate_index = _tool_output(
        _call("plan_trip"),
        _call("get_time", index=1),
    )
    with pytest.raises(KimiK3XTMLParseError, match="unique and sequential"):
        _parse(
            _parser(thinking=False),
            duplicate_index,
            _request(reasoning_effort="none"),
            enable_auto_tools=True,
        )

    multiple_blocks = _tool_output(_call("get_time")) + TOOLS_START + _call("get_time") + TOOLS_END
    with pytest.raises(KimiK3XTMLParseError, match="Unexpected text"):
        _parse(
            _parser(thinking=False),
            multiple_blocks,
            _request(reasoning_effort="none"),
            enable_auto_tools=True,
        )


def test_named_choice_filters_prompt_tools_and_uses_required_instruction():
    request = _request(
        tool_choice={"type": "function", "function": {"name": "get_time"}},
        reasoning_effort="high",
    )
    prepare_kimi_k3_chat_template_kwargs(request)
    params = request.build_chat_params(None, "auto")

    assert params.chat_template_kwargs["thinking"] is True
    assert params.chat_template_kwargs["thinking_effort"] == "high"
    assert params.chat_template_kwargs["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in params.chat_template_kwargs["tools"]] == ["get_time"]
    assert request.tool_choice.function.name == "get_time"


def test_chat_params_accept_kimi_kwargs_and_standard_openai_fields():
    response_format = {"type": "json_object"}
    request = _request(
        chat_template_kwargs={
            "thinking": True,
            "thinking_effort": "max",
            "return_tensors": "pt",
        },
        response_format=response_format,
        structured_outputs={"json": {"type": "object"}},
        chat_template="{{ ignored by the K3 renderer }}",
        add_generation_prompt=False,
        continue_final_message=True,
    )

    prepare_kimi_k3_chat_template_kwargs(request)
    params = request.build_chat_params(None, "auto")

    assert params.chat_template_kwargs["thinking"] is True
    assert params.chat_template_kwargs["thinking_effort"] == "max"
    assert params.chat_template_kwargs["response_format"] == response_format
    assert params.chat_template_kwargs["return_tensors"] == "pt"


def test_chat_params_render_null_tool_choice_with_tools_as_auto_without_mutation():
    request = _request(tool_choice=None)

    prepare_kimi_k3_chat_template_kwargs(request)

    assert request.tool_choice is None
    assert request.chat_template_kwargs["tool_choice"] == "auto"


def test_chat_params_set_canonical_sampling_and_reasoning_controls():
    request = _request(reasoning_effort="minimal")
    prepare_kimi_k3_chat_template_kwargs(request)
    prepare_kimi_k3_chat_template_kwargs(request)  # idempotent

    params = request.build_chat_params(None, "auto")
    assert params.chat_template_kwargs["thinking"] is True
    assert params.chat_template_kwargs["thinking_effort"] == "low"
    assert params.chat_template_kwargs["tool_choice"] == "auto"
    assert request.skip_special_tokens is False
    assert request.spaces_between_special_tokens is False

    no_thinking = _request(reasoning_effort="none")
    prepare_kimi_k3_chat_template_kwargs(no_thinking)
    no_thinking_params = no_thinking.build_chat_params(None, "auto")
    assert no_thinking_params.chat_template_kwargs["thinking"] is False
    assert "thinking_effort" not in no_thinking_params.chat_template_kwargs


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 64])
def test_streaming_reconstructs_nonstream_for_multiple_calls(chunk_size: int):
    request = _request()
    arguments = "".join(
        [
            _argument("route<alternate", "string", 'New "York"'),
            _argument("days", "number", "3"),
            _argument("flexible", "boolean", "true"),
            _argument("metadata", "object", '{"seat":"window"}'),
            _argument("stops", "array", '["Paris","Tokyo"]'),
            _argument("note", "null", "null"),
        ]
    )
    generated = (
        "Need two calls."
        + THINK_END
        + _tool_output(
            _call("plan_trip", arguments),
            _call("get_time", index=2),
            response="Checking first. ",
        )
    )
    full_reasoning, full_content, full_calls = _parse(
        _parser(thinking=True),
        generated,
        request,
        enable_auto_tools=True,
    )

    parser = _parser(thinking=True)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    names: dict[int, str] = {}
    arguments_by_index: dict[int, str] = {}
    for chunk, delta_ids, finished in _token_chunks(generated, chunk_size):
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=delta_ids,
            request=request,
            prompt_token_ids=TOKENIZER.encode(THINK_START),
            finished=finished,
        )
        if delta is None:
            continue
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta.content:
            content_parts.append(delta.content)
        for tool_call in delta.tool_calls:
            assert tool_call.function is not None
            if tool_call.function.name:
                assert tool_call.id is not None
                assert tool_call.type == "function"
                names[tool_call.index] = tool_call.function.name
            else:
                assert tool_call.id is None
                assert tool_call.type is None
            if tool_call.function.arguments is not None:
                arguments_by_index[tool_call.index] = (
                    arguments_by_index.get(tool_call.index, "") + tool_call.function.arguments
                )

    assert "".join(reasoning_parts) == full_reasoning == "Need two calls."
    assert "".join(content_parts) == full_content == "Checking first. "
    assert full_calls is not None
    assert names == {index: call.name for index, call in enumerate(full_calls)}
    assert [json.loads(arguments_by_index[index]) for index in sorted(arguments_by_index)] == [
        json.loads(call.arguments) for call in full_calls
    ]
    assert "<|" not in "".join(reasoning_parts + content_parts)


def test_streaming_json_block_emits_name_and_arguments_incrementally():
    request = _request(reasoning_effort="none")
    raw_json = '{"city":"New York","metadata":{"days":[1,2,3]}}'
    parser = _parser(thinking=False)
    prefix = (
        RESPONSE_END
        + TOOLS_START
        + '<|open|>call tool="plan_trip" index="1"<|sep|>'
        + f'<|open|>json type="object"{SEP_TOKEN}'
        + raw_json
    )
    partial_delta = parser.parse_delta(
        delta_text=prefix,
        delta_token_ids=TOKENIZER.encode(prefix),
        request=request,
        finished=False,
    )

    assert partial_delta is not None
    assert partial_delta.tool_calls[0].function.name == "plan_trip"
    assert partial_delta.tool_calls[0].function.arguments == raw_json

    suffix = JSON_END + CALL_END
    delta = parser.parse_delta(
        delta_text=suffix,
        delta_token_ids=TOKENIZER.encode(suffix),
        request=request,
        finished=False,
    )

    assert delta is None


def test_streaming_rejects_invalid_tool_prefix_before_finish():
    generated = RESPONSE_END + TOOLS_START + _call("delete_everything")
    with pytest.raises(KimiK3XTMLParseError, match="Unknown K3 tool"):
        _parser(thinking=False).parse_delta(
            delta_text=generated,
            delta_token_ids=TOKENIZER.encode(generated),
            request=_request(reasoning_effort="none"),
            finished=False,
        )


def test_overlong_unclosed_tool_tag_is_rejected():
    generated = RESPONSE_END + TOOLS_START + OPEN_TOKEN + "call " + "x" * 9000
    with pytest.raises(KimiK3XTMLParseError, match="tag exceeds"):
        _parser(thinking=False).parse_delta(
            delta_text=generated,
            delta_token_ids=TOKENIZER.encode(generated),
            request=_request(reasoning_effort="none"),
            finished=False,
        )


def test_streaming_truncation_emits_partial_delta_but_no_completed_call():
    generated = (
        RESPONSE_END
        + TOOLS_START
        + '<|open|>call tool="plan_trip" index="1"<|sep|>'
        + '<|open|>argument key="city" type="string"<|sep|>Par'
    )
    parser = _parser(thinking=False)
    delta = parser.parse_delta(
        delta_text=generated,
        delta_token_ids=TOKENIZER.encode(generated),
        request=_request(reasoning_effort="none"),
        finished=True,
    )

    assert delta is not None
    assert delta.tool_calls[0].function.name == "plan_trip"
    assert delta.tool_calls[0].function.arguments == '{"city":"Par'
    assert parser._stream_parser is not None
    assert parser._stream_parser.snapshot.tool_calls == ()


def test_streaming_truncated_tools_preserves_completed_call_at_finish():
    generated = RESPONSE_END + TOOLS_START + _call("get_time")
    delta = _parser(thinking=False).parse_delta(
        delta_text=generated,
        delta_token_ids=TOKENIZER.encode(generated),
        request=_request(reasoning_effort="none"),
        finished=True,
    )

    assert delta is not None
    assert delta.tool_calls[0].function.name == "get_time"
    assert json.loads("".join(call.function.arguments or "" for call in delta.tool_calls)) == {}


def test_streaming_include_reasoning_false_and_choice_state_isolation():
    request = _request(include_reasoning=False)
    generated = "private reasoning" + THINK_END + _tool_output(response="public answer")
    parsers = [_parser(thinking=True), _parser(thinking=True)]
    outputs: list[list[str]] = [[], []]

    for chunk, delta_ids, finished in _token_chunks(generated, 3):
        for index, parser in enumerate(parsers):
            delta = parser.parse_delta(
                delta_text=chunk,
                delta_token_ids=delta_ids,
                request=request,
                prompt_token_ids=TOKENIZER.encode(THINK_START),
                finished=finished,
            )
            if delta is not None:
                assert delta.reasoning is None
                if delta.content:
                    outputs[index].append(delta.content)

    assert ["".join(parts) for parts in outputs] == [
        "public answer",
        "public answer",
    ]
