# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest, CompletionResponse, CompletionResponseChoice
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.entrypoints.serve.engine.serving import BaseServing

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_recompute_response.py"
spec = importlib.util.spec_from_file_location("ascend_recompute_test", PATCH_PATH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


def response(reason):
    return CompletionResponse(
        id="cmpl-test",
        created=1,
        model="test",
        choices=[CompletionResponseChoice(index=0, text="tail", finish_reason="stop", stop_reason=reason)],
        usage=UsageInfo(prompt_tokens=4, completion_tokens=1, total_tokens=5),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason", [None, "recomputed"])
@pytest.mark.parametrize("stop", [None, "recomputed", ["recomputed"]])
async def test_response_contract(stream, reason, stop):
    original_response = response(reason)
    frames = [f"data: {original_response.model_dump_json()}\n\n", "data: [DONE]\n\n"]
    closed = []

    async def source():
        try:
            for frame in frames:
                yield frame
        finally:
            closed.append(True)

    async def original(self, request):
        return source() if request.stream else original_response

    request = CompletionRequest(model="test", prompt="hello", stream=stream, stop=stop)
    result = await patch._wrap_response(original)(BaseServing, request)
    if stream:
        actual = [frame async for frame in result]
        assert closed == [True]
        assert actual[-1] == "data: [DONE]\n\n"
        payload = json.loads(actual[0][6:])
        if reason != "recomputed" or stop is not None:
            assert actual == frames
    else:
        payload = result.model_dump()
        if reason != "recomputed" or stop is not None:
            assert result is original_response
    if reason == "recomputed" and stop is None:
        assert payload["error"]["type"] == "RecomputeRequiredError"
        assert payload["partial_response"] == original_response.model_dump(mode="json")


@pytest.mark.asyncio
async def test_normal_frames_unchanged():
    frames = [": heartbeat\n\n", 'data: {"choices":[{"text":"recomputed","stop_reason":null}]}\n\n', "data: [DONE]\n\n"]

    async def source():
        for frame in frames:
            yield frame

    assert [frame async for frame in patch._stream(BaseServing, source())] == frames


@pytest.mark.asyncio
async def test_error_closes_source_before_delivery():
    closed = asyncio.Event()

    async def source():
        try:
            yield f"data: {response('recomputed').model_dump_json()}\n\n"
            raise AssertionError("must not request another event")
        finally:
            closed.set()

    stream = patch._stream(BaseServing, source())
    assert '"error"' in await anext(stream)
    assert closed.is_set()
    await stream.aclose()


@pytest.mark.asyncio
async def test_cancel_closes_nested_result():
    closed = asyncio.Event()

    async def results():
        try:
            yield "normal"
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def original(result_generator):
        async for item in result_generator:
            yield item

    stream = patch._wrap_generator(original)(results())
    assert await anext(stream) == "normal"
    await stream.aclose()
    assert closed.is_set()


def test_registration_idempotent():
    original = patch.OpenAIServingChat.create_chat_completion
    patch._apply_patch()
    assert patch.OpenAIServingChat.create_chat_completion is original


@pytest.mark.asyncio
async def test_chat_closes_raw_and_result_before_error():
    from vllm.entrypoints.generate.base.protocol import DeltaMessage
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionResponseStreamChoice,
        ChatCompletionStreamResponse,
    )

    closed = []

    async def results():
        try:
            yield "tail"
        finally:
            closed.append("result")

    class Serving:
        create_error_response = staticmethod(BaseServing.create_error_response)

        async def chat_completion_stream_generator_raw(self, request, result_generator):
            try:
                text = await anext(result_generator)
                yield ChatCompletionStreamResponse(
                    model="test",
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0, delta=DeltaMessage(content=text), finish_reason="stop", stop_reason="recomputed"
                        )
                    ],
                )
            finally:
                closed.append("raw")

    async def unused(self, request, result_generator):
        yield "unused"

    request = CompletionRequest(model="test", prompt="test", stream=True)
    stream = patch._wrap_chat_stream(unused)(Serving(), request, results())
    assert '"error"' in await anext(stream)
    assert closed == ["raw", "result"]
    assert await anext(stream) == "data: [DONE]\n\n"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
