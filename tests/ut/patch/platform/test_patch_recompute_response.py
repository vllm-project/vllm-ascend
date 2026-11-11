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
assert spec is not None and spec.loader is not None
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
@pytest.mark.parametrize("reason", [None, "recomputed", "-1", 42, -1])
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
        if reason != -1:
            assert actual == frames
    else:
        payload = result.model_dump()
        if reason != -1:
            assert result is original_response
    if reason == -1:
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
            yield f"data: {response(-1).model_dump_json()}\n\n"
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
                            index=0, delta=DeltaMessage(content=text), finish_reason="stop", stop_reason=-1
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


@pytest.mark.parametrize("interrupted", [False, True])
def test_marker_survives_native_output_processing_and_merge(interrupted):
    import msgspec
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
    from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer
    from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector

    request = EngineCoreRequest(
        request_id="r",
        external_req_id="r",
        prompt_token_ids=[0],
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(
            stop=["recomputed"], include_stop_str_in_output=True, output_kind=RequestOutputKind.DELTA
        ),
        pooling_params=None,
    )

    class Tokens(BaseIncrementalDetokenizer):
        def decode_next(self, token_id):
            return {1: "recom", 2: "puted"}[token_id]

    processor = OutputProcessor(tokenizer=None, log_stats=False)
    processor.add_request(request, prompt="test")
    processor.request_states["r"].detokenizer = Tokens(request)
    collector = RequestOutputCollector(RequestOutputKind.DELTA, "r")
    terminal = EngineCoreOutput(request_id="r", new_token_ids=[], finish_reason=FinishReason.STOP, stop_reason=-1)
    terminal = msgspec.msgpack.decode(msgspec.msgpack.encode(terminal), type=EngineCoreOutput)
    for output in [
        EngineCoreOutput(request_id="r", new_token_ids=[1]),
        terminal if interrupted else EngineCoreOutput(request_id="r", new_token_ids=[2]),
    ]:
        for response in processor.process_outputs([output]).request_outputs:
            collector.put(response)
    merged = collector.get_nowait()
    assert merged.finished
    assert merged.outputs[0].stop_reason == (-1 if interrupted else "recomputed")
    assert merged.outputs[0].text == ("recom" if interrupted else "recomputed")
    assert processor.get_num_unfinished_requests() == 0


def test_internal_marker_is_not_a_valid_user_stop_token():
    from unittest.mock import Mock

    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams

    params = SamplingParams(stop_token_ids=[patch.RECOMPUTE_STOP_REASON])
    with pytest.raises(VLLMValidationError):
        params._validate_stop_token_ids(Mock(get_vocab_size=lambda: 128))


def test_nonzero_choice_marker_is_detected():
    payload = response(None).model_dump(mode="json")
    payload["choices"].append({**payload["choices"][0], "index": 1, "stop_reason": patch.RECOMPUTE_STOP_REASON})
    error = patch._error(BaseServing, payload)
    assert error.error.type == "RecomputeRequiredError"
    assert error.partial_response == payload
