"""Exercise the Ascend signal through vLLM's native HTTP serializers."""

import asyncio
import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from openai.types.responses import ResponseErrorEvent
from pydantic import TypeAdapter
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.responses import serving as responses_serving
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.exception_handling.error_response import create_error_response
from vllm.entrypoints.serve.exception_handling.handlers.vllm_error import vllm_error_handler
from vllm.exceptions import VLLMError
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_ascend.core.pd_recompute import (
    PD_CONTINUE_MESSAGE,
    PD_RECOMPUTE_ARG,
    PD_RECOMPUTE_INTERRUPTED,
    PD_RECOMPUTE_RESUME,
    PD_RECOMPUTE_RETRY,
    PD_RECOMPUTE_VERSION,
)
from vllm_ascend.patch.platform.patch_pd_recompute import PDRecomputeError, _wrap_generate


def _output(text, tokens, *, signal=None, progress=None):
    return RequestOutput(
        request_id="decode-attempt",
        prompt="hello",
        prompt_token_ids=[17],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=tokens,
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="error" if signal else None,
                stop_reason="recomputed" if signal else None,
            )
        ],
        finished=signal is not None,
        kv_transfer_params={PD_RECOMPUTE_ARG: signal, **(progress or {})} if signal else None,
    )


@pytest.mark.parametrize("signal", [PD_RECOMPUTE_RETRY, PD_RECOMPUTE_INTERRUPTED])
def test_async_llm_adapter_keeps_buffered_delta_and_reports_error(signal):
    pending_text = "buffered text" if signal == PD_RECOMPUTE_INTERRUPTED else ""
    pending_ids = [31, 32] if pending_text else []
    was_closed = []

    async def generate(self, prompt, sampling_params, request_id) -> AsyncGenerator[RequestOutput, None]:
        try:
            yield _output(pending_text, pending_ids, signal=signal)
        finally:
            was_closed.append(request_id)

    async def exercise():
        wrapped = _wrap_generate(generate)
        params = SamplingParams(extra_args={PD_RECOMPUTE_ARG: PD_RECOMPUTE_VERSION})
        chunks = []
        with pytest.raises(PDRecomputeError, match=signal):
            async for response in wrapped(None, "hello", params, "opted-in"):
                chunks.append(response)
        assert [token for chunk in chunks for token in chunk.outputs[0].token_ids] == pending_ids
        assert all(not chunk.finished and chunk.outputs[0].finish_reason is None for chunk in chunks)
        assert all(chunk.kv_transfer_params is None for chunk in chunks)
        original = [response async for response in wrapped(None, "hello", SamplingParams(), "not-opted-in")]
        assert original[0].finished
        assert original[0].outputs[0].stop_reason == "recomputed"

    asyncio.run(exercise())
    assert was_closed == ["opted-in", "not-opted-in"]


def test_messages_serializer_keeps_scheduler_signal():
    async def errors():
        yield create_error_response(PDRecomputeError(PD_RECOMPUTE_RETRY))

    frontend = AnthropicServingMessages.__new__(AnthropicServingMessages)

    async def exercise():
        return [event async for event in frontend.message_stream_converter(errors())]

    events = asyncio.run(exercise())
    assert len(events) == 1
    assert events[0].startswith("event: error\n")
    payload = json.loads(events[0].split("data: ", 1)[1])
    assert payload["error"]["message"] == PD_RECOMPUTE_RETRY


@pytest.mark.parametrize("output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
@pytest.mark.parametrize("serialize_partial", [False, True])
def test_native_resume_preserves_partial_text_and_budget(output_kind, serialize_partial):
    progress = {"output_tokens": 3, "remaining_tokens": 7, "input_tokens": 19}

    async def generate(self, prompt, sampling_params, request_id):
        yield _output("Text with /tmp/example", [31, 32, 33], signal=PD_RECOMPUTE_INTERRUPTED, progress=progress)

    async def exercise():
        extra_args = {PD_RECOMPUTE_ARG: PD_RECOMPUTE_VERSION}
        if serialize_partial:
            extra_args[PD_RECOMPUTE_RESUME] = PD_RECOMPUTE_VERSION
        params = SamplingParams(output_kind=output_kind, extra_args=extra_args)
        iterator = _wrap_generate(generate)(None, "hello", params, "native")
        output = await anext(iterator)
        assert output.outputs[0].text == "Text with /tmp/example"
        assert output.outputs[0].stop_reason is None
        metadata = {PD_RECOMPUTE_ARG: PD_RECOMPUTE_INTERRUPTED, **progress}
        if output_kind == RequestOutputKind.FINAL_ONLY and serialize_partial:
            assert output.finished
            assert output.outputs[0].finish_reason == "length"
            assert output.kv_transfer_params == metadata
            with pytest.raises(StopAsyncIteration):
                await anext(iterator)
        else:
            assert not output.finished
            assert output.outputs[0].finish_reason is None
            assert output.kv_transfer_params is None
            with pytest.raises(PDRecomputeError) as error:
                await anext(iterator)
            frontend = responses_serving.OpenAIServingResponses.__new__(responses_serving.OpenAIServingResponses)
            event = json.loads(frontend._convert_generation_error_to_streaming_response(error.value))
            signal, detail = event["message"].split(":", 1)
            assert signal == event["code"] == PD_RECOMPUTE_INTERRUPTED
            assert json.loads(detail) == metadata

    asyncio.run(exercise())


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("text", [None, "visible prefix"])
def test_messages_partial_tool_arguments_keep_private_recompute_signal(interrupted, text):
    progress = {
        PD_RECOMPUTE_ARG: PD_RECOMPUTE_INTERRUPTED,
        "output_tokens": 3,
        "remaining_tokens": 7,
        "input_tokens": 19,
    }
    tool_call = SimpleNamespace(id="call_1", function=SimpleNamespace(name="lookup", arguments='{"key":'))
    choice = SimpleNamespace(
        finish_reason="length",
        stop_reason=None,
        message=SimpleNamespace(reasoning=None, content=text, tool_calls=[tool_call]),
    )
    partial = SimpleNamespace(
        id="chatcmpl-1",
        model="model",
        usage=None,
        kv_transfer_params=progress if interrupted else None,
        ec_transfer_params=None,
        choices=[choice],
    )
    frontend = AnthropicServingMessages.__new__(AnthropicServingMessages)
    if interrupted and text:
        result = frontend.messages_full_converter(partial)
        assert [block.text for block in result.content] == [text]
        assert result.kv_transfer_params == progress
        assert choice.message.tool_calls == [tool_call]
        return
    with pytest.raises(PDRecomputeError if interrupted else json.JSONDecodeError) as error:
        frontend.messages_full_converter(partial)
    if interrupted:
        signal, details = str(error.value).split(":", 1)
        assert signal == PD_RECOMPUTE_INTERRUPTED
        assert json.loads(details) == progress


@pytest.mark.parametrize("continuing", [False, True])
def test_messages_continuation_opens_only_the_last_assistant_message(continuing):
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "The answer"}]
    request = AnthropicMessagesRequest(
        model="model",
        max_tokens=7,
        messages=messages,
        vllm_xargs={PD_CONTINUE_MESSAGE: PD_RECOMPUTE_VERSION} if continuing else None,
    )
    converted = AnthropicServingMessages.to_chat_completion_request(request)
    assert converted.messages == messages
    assert converted.max_completion_tokens == 7
    assert converted.continue_final_message is continuing
    assert converted.add_generation_prompt is not continuing


def test_responses_continuation_selects_the_open_assistant_item():
    request = ResponsesRequest(
        model="model",
        max_output_tokens=7,
        input=[
            {"role": "user", "content": "hello"},
            {
                "type": "message",
                "role": "assistant",
                "id": "msg_partial",
                "status": "in_progress",
                "content": [{"type": "output_text", "text": "The answer", "annotations": []}],
            },
        ],
    )
    params = request.build_chat_params(None, "auto")
    assert params.chat_template_kwargs["continue_final_message"] is True
    assert params.chat_template_kwargs["add_generation_prompt"] is False


def test_responses_serializer_emits_valid_sdk_error_event():
    frontend = responses_serving.OpenAIServingResponses.__new__(responses_serving.OpenAIServingResponses)
    error_json = frontend._convert_generation_error_to_streaming_response(PDRecomputeError(PD_RECOMPUTE_INTERRUPTED))
    event = TypeAdapter(responses_serving.StreamingResponsesResponse).validate_json(error_json)
    assert isinstance(event, ResponseErrorEvent)
    assert event.code == PD_RECOMPUTE_INTERRUPTED
    assert event.sequence_number == -1


def test_sdk_event_is_the_only_terminal_event_from_responses(monkeypatch):
    frontend = responses_serving.OpenAIServingResponses.__new__(responses_serving.OpenAIServingResponses)
    frontend.use_harmony = False

    async def interrupted_generation(*args: Any) -> AsyncGenerator[Any, None]:
        if False:
            yield None
        raise PDRecomputeError(PD_RECOMPUTE_INTERRUPTED)

    monkeypatch.setattr(frontend, "_process_simple_streaming_events", interrupted_generation)
    response_kwargs = {"model_dump": lambda **options: {"id": "resp-test", "status": "in_progress"}}
    monkeypatch.setattr(
        responses_serving.ResponsesResponse,
        "from_request",
        lambda *args, **kwargs: SimpleNamespace(**response_kwargs),
    )
    for event_name in ("ResponseCreatedEvent", "ResponseInProgressEvent"):
        monkeypatch.setattr(responses_serving, event_name, lambda **values: SimpleNamespace(**values))

    async def exercise():
        events = [
            event
            async for event in frontend.responses_stream_generator(
                SimpleNamespace(request_id="decode-attempt"),
                None,
                None,
                None,
                "model",
                None,
                None,
            )
        ]
        assert [event.type for event in events] == ["response.created", "response.in_progress", "error"]
        assert [event.sequence_number for event in events] == [0, 1, 2]
        assert events[-1].code == PD_RECOMPUTE_INTERRUPTED

    asyncio.run(exercise())


@pytest.mark.parametrize("signal", [PD_RECOMPUTE_RETRY, PD_RECOMPUTE_INTERRUPTED])
def test_non_streaming_scheduler_signal_is_an_http_error(signal):
    app = FastAPI()
    app.add_exception_handler(VLLMError, vllm_error_handler)

    @app.post("/generate")
    async def generate():
        raise PDRecomputeError(signal)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://server") as client:
            response = await client.post("/generate")
        assert response.status_code == 500
        assert response.json()["error"]["message"] == signal

    asyncio.run(exercise())
