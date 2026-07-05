# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest

from vllm_ascend.utils import vllm_version_is

pytestmark = pytest.mark.skipif(
    not vllm_version_is("0.24.0"),
    reason="parser reasoning usage patch only applies to vLLM 0.24.0",
)

from vllm.entrypoints.openai.chat_completion import protocol as chat_protocol  # noqa: E402
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat  # noqa: E402
from vllm.entrypoints.openai.engine import protocol as engine_protocol  # noqa: E402
from vllm.parser.abstract_parser import DelegatingParser  # noqa: E402

from vllm_ascend.patch.platform import patch_parser_reasoning_usage as usage_patch  # noqa: E402


class FakeCountingParser:
    def __init__(self, tokenizer, tools=None, chat_template_kwargs=None):
        self.reasoning_tokens = 0
        self._stream_state = SimpleNamespace(
            tool_call_id_type="random",
            history_tool_call_cnt=0,
        )

    def parse_delta(
        self,
        delta_text,
        delta_token_ids,
        request,
        prompt_token_ids=None,
        *,
        finished,
    ):
        self.reasoning_tokens += len(delta_token_ids)
        return None

    def parse(
        self,
        model_output,
        request,
        enable_auto_tools=False,
        model_output_token_ids=(),
    ):
        self.reasoning_tokens += len(model_output_token_ids)
        return None, model_output, []

    def count_reasoning_tokens(self, token_ids=None):
        return self.reasoning_tokens


class FakeSequenceCountingParser(FakeCountingParser):
    def parse_delta(
        self,
        delta_text,
        delta_token_ids,
        request,
        prompt_token_ids=None,
        *,
        finished,
    ):
        return None

    def count_reasoning_tokens(self, token_ids=None):
        return len(token_ids or [])


class FakeTokenizer:
    _TOKEN_TEXT = {
        1: "<think>",
        2: "</think>",
        10: "a",
        11: "b",
        20: "c",
    }

    def get_vocab(self):
        return {
            "<think>": 1,
            "</think>": 2,
            "<minimax:tool_call>": 3,
            "</minimax:tool_call>": 4,
        }

    def decode(self, token_ids):
        return "".join(self._TOKEN_TEXT.get(token_id, "") for token_id in token_ids)


def _fake_serving():
    return SimpleNamespace(
        parser_cls=FakeCountingParser,
        tool_call_id_type="random",
        enable_auto_tools=False,
    )


def _fake_request(**kwargs):
    values: dict[str, object] = {
        "n": None,
        "tools": [],
        "tool_choice": None,
        "_grammar_from_tool_parser": False,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_usage_schema_includes_completion_token_details():
    usage = engine_protocol.UsageInfo(
        prompt_tokens=3,
        completion_tokens=4,
        total_tokens=7,
        completion_tokens_details=usage_patch._make_completion_tokens_details(2),
    )

    payload = usage.model_dump(exclude_none=True)

    assert payload["completion_tokens_details"] == {"reasoning_tokens": 2}
    assert chat_protocol.UsageInfo is usage_patch.UsageInfo


def test_chat_generators_are_patched_at_class_level():
    assert OpenAIServingChat.chat_completion_stream_generator is usage_patch._wrapped_chat_completion_stream_generator
    assert OpenAIServingChat.chat_completion_full_generator is usage_patch._wrapped_chat_completion_full_generator


def test_delegating_parser_count_reasoning_tokens_delegates_to_reasoning_parser():
    class FakeReasoningParser:
        def count_reasoning_tokens(self, token_ids):
            return len(token_ids) + 1

    parser = object.__new__(DelegatingParser)
    parser._reasoning_parser = FakeReasoningParser()

    assert parser.count_reasoning_tokens([10, 11]) == 3


def test_stream_tracking_counts_reasoning_tokens_from_parser():
    state = usage_patch._create_usage_tracking_state(
        _fake_serving(),
        _fake_request(),
        FakeTokenizer(),
        conversation=[],
    )
    res = SimpleNamespace(
        prompt_token_ids=[1, 2],
        encoder_prompt_token_ids=None,
        num_cached_tokens=0,
        outputs=[
            SimpleNamespace(
                index=0,
                token_ids=[10, 11],
                text="ab",
                finish_reason=None,
            )
        ],
    )

    usage_patch._update_usage_tracking_state(state, _fake_request(), res)

    assert state.num_prompt_tokens == 2
    assert state.num_cached_tokens == 0
    assert state.completion_tokens == [2]
    assert state.reasoning_tokens == [2]


def test_stream_tracking_passes_accumulated_tokens_to_sequence_counter():
    serving = _fake_serving()
    serving.parser_cls = FakeSequenceCountingParser
    request = _fake_request()
    state = usage_patch._create_usage_tracking_state(
        serving,
        request,
        FakeTokenizer(),
        conversation=[],
    )

    for token_ids, text, finish_reason in [
        ([10, 11], "ab", None),
        ([20], "c", "stop"),
    ]:
        res = SimpleNamespace(
            prompt_token_ids=[1, 2],
            encoder_prompt_token_ids=None,
            num_cached_tokens=0,
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=token_ids,
                    text=text,
                    finish_reason=finish_reason,
                )
            ],
        )
        usage_patch._update_usage_tracking_state(state, request, res)

    assert state.output_token_ids == [[10, 11, 20]]
    assert state.reasoning_tokens == [3]


def test_stream_usage_details_are_injected():
    state = usage_patch._UsageTrackingState(
        completion_tokens=[2],
        reasoning_tokens=[2],
        counting_parsers=[],
        parser_failed=[],
    )
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
    }

    data = usage_patch._inject_stream_usage_details(
        f"data: {json.dumps(chunk)}\n\n",
        state,
    )
    payload = json.loads(data.removeprefix("data: ").removesuffix("\n\n"))

    assert payload["usage"]["completion_tokens_details"] == {
        "reasoning_tokens": 2,
    }


def test_full_response_reasoning_tokens_are_counted_per_output_choice():
    final_res = SimpleNamespace(
        outputs=[
            SimpleNamespace(index=0, token_ids=[10, 11], text="ab"),
            SimpleNamespace(index=1, token_ids=[20], text="c"),
        ],
    )

    reasoning_tokens = usage_patch._count_full_response_reasoning_tokens(
        _fake_serving(),
        _fake_request(),
        final_res,
        FakeTokenizer(),
        conversation=[],
    )

    assert reasoning_tokens == 3


def test_minimax_parser_engine_counts_initial_reasoning_tokens_when_available():
    pytest.importorskip("vllm.parser.engine.parser_engine")

    from vllm.parser.parser_manager import ParserManager

    parser_cls = ParserManager.get_parser(
        reasoning_parser_name="minimax_m2",
        enable_auto_tools=False,
        model_name="MiniMax-M2",
    )
    parser = parser_cls(FakeTokenizer(), tools=[])

    parser.parse(
        "ab</think>c",
        _fake_request(),
        enable_auto_tools=False,
        model_output_token_ids=[10, 11, 2, 20],
    )

    assert parser.count_reasoning_tokens([10, 11, 2, 20]) == 2


@pytest.mark.parametrize(
    ("token_ids", "expected_reasoning_tokens"),
    [
        ([10, 11, 2, 20], 2),
        ([10, 11, 20], 3),
        ([2, 20], 0),
    ],
)
def test_minimax_append_think_counts_reasoning_tokens(
    token_ids,
    expected_reasoning_tokens,
):
    from vllm.reasoning.minimax_m2_reasoning_parser import (
        MiniMaxM2AppendThinkReasoningParser,
    )

    parser = MiniMaxM2AppendThinkReasoningParser(FakeTokenizer())

    assert parser.count_reasoning_tokens(token_ids) == expected_reasoning_tokens


def test_stream_wrapper_does_not_forward_legacy_reasoning_parser_kwarg():
    async def run_wrapper():
        async def original_stream_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            chat_template_kwargs=None,
        ):
            assert chat_template_kwargs == {"thinking": True}
            yield "data: [DONE]\n\n"

        async def variadic_stream_wrapper(*args, **kwargs):
            async for data in original_stream_generator(*args, **kwargs):
                yield data

        async def empty_result_generator():
            if False:
                yield None

        serving = _fake_serving()
        serving._ascend_original_chat_completion_stream_generator_for_reasoning_usage = variadic_stream_wrapper
        request_metadata = SimpleNamespace(final_usage_info=None)

        return [
            data
            async for data in usage_patch._wrapped_chat_completion_stream_generator(
                serving,
                _fake_request(),
                empty_result_generator(),
                "request-id",
                "model-name",
                [],
                FakeTokenizer(),
                request_metadata,
                reasoning_parser=object(),
                chat_template_kwargs={"thinking": True},
            )
        ]

    assert asyncio.run(run_wrapper()) == ["data: [DONE]\n\n"]
