# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.parser import ParserManager
from vllm.reasoning import ReasoningParserManager

from vllm_ascend.patch.platform import patch_glm_reasoning_usage_accounting as patch


class FakeTokenizer:
    def get_vocab(self):
        return {"<think>": 1, "</think>": 2}


def _reasoning_parser(parser_name="glm45", chat_template_kwargs=None):
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    kwargs = {}
    if chat_template_kwargs is not None:
        kwargs["chat_template_kwargs"] = chat_template_kwargs
    return parser_cls(FakeTokenizer(), **kwargs)


async def _result_generator(result):
    yield result


@pytest.mark.parametrize("parser_name", ["glm45", "glm47"])
@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        pytest.param([1, 10, 11, 2, 20], 2, id="explicit-start"),
        pytest.param([99, 1, 10, 11, 2, 20], 2, id="prefix-before-start"),
        pytest.param([10, 11, 2, 20], 2, id="implicit-start"),
        pytest.param([10, 11], 2, id="truncated-reasoning"),
        pytest.param([2, 20], 0, id="end-token-first"),
        pytest.param([], 0, id="empty-output"),
    ],
)
def test_reasoning_token_count(parser_name, token_ids, expected):
    assert _reasoning_parser(parser_name).count_reasoning_tokens(token_ids) == expected


@pytest.mark.parametrize(
    "chat_template_kwargs",
    [
        {"thinking": False, "enable_thinking": False},
        {"thinking": False, "enable_thinking": None},
        {"thinking": None, "enable_thinking": False},
    ],
)
def test_reasoning_token_count_when_thinking_is_disabled(chat_template_kwargs):
    parser = _reasoning_parser(chat_template_kwargs=chat_template_kwargs)

    assert parser.count_reasoning_tokens([10, 11, 2, 20]) == 0


@pytest.mark.parametrize(
    "token_chunks",
    [
        [[10], [11, 2], [20]],
        [[99], [1, 10], [11, 2, 20]],
        [[10], [11]],
    ],
)
def test_incremental_counter_matches_batch_count(token_chunks):
    counter = patch._IncrementalReasoningCounter(1, 2, enabled=True)
    token_ids = []

    for chunk in token_chunks:
        token_ids.extend(chunk)
        counter.update(chunk)
        assert counter.count == _reasoning_parser().count_reasoning_tokens(token_ids)


def test_stream_usage_reports_reasoning_tokens():
    counter = patch._IncrementalReasoningCounter(1, 2, enabled=True)
    counter.update([10, 11, 2, 20])
    state = patch._StreamUsageState(counters=[counter])
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
    }

    data = patch._inject_stream_usage_details(
        f"data: {json.dumps(chunk)}\n\n",
        state,
    )
    payload = json.loads(data.removeprefix("data: ").removesuffix("\n\n"))

    assert payload["usage"]["completion_tokens_details"] == {
        "reasoning_tokens": 2,
    }


def test_full_response_usage_reports_reasoning_tokens():
    parser_cls = ParserManager.get_parser(reasoning_parser_name="glm45")
    parser = parser_cls(FakeTokenizer())
    final_res = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[10, 11, 2, 20])])
    usage = patch.UsageInfo(
        prompt_tokens=3,
        completion_tokens=4,
        total_tokens=7,
    )

    reasoning_tokens = patch._count_full_response_reasoning_tokens(parser, final_res)
    patch._set_usage_details(usage, reasoning_tokens)

    assert usage.completion_tokens_details.reasoning_tokens == 2


def test_stream_wrapper_injects_usage_and_updates_metadata():
    result = SimpleNamespace(outputs=[SimpleNamespace(index=0, token_ids=[10, 11, 2, 20])])
    request = SimpleNamespace(n=1)
    request_metadata = SimpleNamespace(final_usage_info=None)

    async def original(
        request,
        result_generator,
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        **kwargs,
    ):
        async for _ in result_generator:
            pass
        request_metadata.final_usage_info = patch.UsageInfo(
            prompt_tokens=3,
            completion_tokens=4,
            total_tokens=7,
        )
        yield ('data: {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}\n\n')
        yield "data: [DONE]\n\n"

    serving = SimpleNamespace(_ascend_original_chat_completion_stream_generator=original)

    async def collect_chunks():
        return [
            chunk
            async for chunk in patch._wrapped_chat_completion_stream_generator(
                serving,
                request,
                _result_generator(result),
                "chatcmpl-test",
                "test-model",
                [],
                FakeTokenizer(),
                request_metadata,
            )
        ]

    chunks = asyncio.run(collect_chunks())

    usage_chunk = json.loads(chunks[0].removeprefix("data: ").removesuffix("\n\n"))
    assert usage_chunk["usage"]["completion_tokens_details"] == {
        "reasoning_tokens": 2,
    }
    assert request_metadata.final_usage_info.completion_tokens_details.reasoning_tokens == 2


def test_full_wrapper_updates_response_and_metadata():
    final_res = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[10, 11, 2, 20])])
    request_metadata = SimpleNamespace(final_usage_info=None)
    parser_cls = ParserManager.get_parser(reasoning_parser_name="glm45")
    parser = parser_cls(FakeTokenizer())

    async def original(
        request,
        result_generator,
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        **kwargs,
    ):
        async for _ in result_generator:
            pass
        usage = patch.UsageInfo(
            prompt_tokens=3,
            completion_tokens=4,
            total_tokens=7,
        )
        request_metadata.final_usage_info = usage
        return patch.chat_protocol.ChatCompletionResponse(
            model=model_name,
            choices=[],
            usage=usage,
        )

    serving = SimpleNamespace(_ascend_original_chat_completion_full_generator=original)
    response = asyncio.run(
        patch._wrapped_chat_completion_full_generator(
            serving,
            SimpleNamespace(),
            _result_generator(final_res),
            "chatcmpl-test",
            "test-model",
            [],
            FakeTokenizer(),
            request_metadata,
            parser=parser,
        )
    )

    assert response.usage.completion_tokens_details.reasoning_tokens == 2
    assert request_metadata.final_usage_info.completion_tokens_details.reasoning_tokens == 2


def test_chat_usage_wrapper_is_bound_only_for_glm_instances():
    glm_parser_cls = ParserManager.get_parser(reasoning_parser_name="glm45")
    qwen_parser_cls = ParserManager.get_parser(reasoning_parser_name="qwen3")
    glm_serving = object.__new__(OpenAIServingChat)
    qwen_serving = object.__new__(OpenAIServingChat)

    glm_serving.parser_cls = glm_parser_cls
    qwen_serving.parser_cls = qwen_parser_cls

    assert glm_serving._ascend_glm_reasoning_usage_patched
    assert "chat_completion_stream_generator" in glm_serving.__dict__
    assert "chat_completion_full_generator" in glm_serving.__dict__
    assert not hasattr(qwen_serving, "_ascend_glm_reasoning_usage_patched")
    assert "chat_completion_stream_generator" not in qwen_serving.__dict__
    assert "chat_completion_full_generator" not in qwen_serving.__dict__
