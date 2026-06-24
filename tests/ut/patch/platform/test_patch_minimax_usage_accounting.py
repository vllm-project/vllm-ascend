# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.minimax_m2_reasoning_parser import (
    MiniMaxM2AppendThinkReasoningParser,
    MiniMaxM2ReasoningParser,
)

from vllm_ascend.patch.platform import (
    patch_chat_usage_accounting as usage_patch,
)
from vllm_ascend.patch.platform import patch_minimax_usage_accounting  # noqa: F401


class FakeTokenizer:
    def get_vocab(self):
        return {
            "<think>": 1,
            "</think>": 2,
            "<minimax:tool_call>": 3,
            "</minimax:tool_call>": 4,
        }


@pytest.mark.parametrize(
    ("parser_cls", "token_ids", "expected_reasoning_tokens"),
    [
        pytest.param(
            MiniMaxM2ReasoningParser,
            [10, 11, 2, 20],
            2,
            id="minimax-reasoning-before-end-token",
        ),
        pytest.param(
            MiniMaxM2AppendThinkReasoningParser,
            [10, 11, 2, 20],
            2,
            id="append-think-reasoning-before-end-token",
        ),
        pytest.param(
            MiniMaxM2ReasoningParser,
            [10, 11, 20],
            3,
            id="minimax-no-end-token-means-all-output-is-reasoning",
        ),
        pytest.param(
            MiniMaxM2AppendThinkReasoningParser,
            [10, 11, 20],
            3,
            id="append-think-no-end-token-means-all-output-is-reasoning",
        ),
        pytest.param(
            MiniMaxM2ReasoningParser,
            [2, 20],
            0,
            id="minimax-end-token-first-means-no-reasoning-tokens",
        ),
        pytest.param(
            MiniMaxM2AppendThinkReasoningParser,
            [2, 20],
            0,
            id="append-think-end-token-first-means-no-reasoning-tokens",
        ),
    ],
)
def test_count_reasoning_tokens(
    parser_cls,
    token_ids,
    expected_reasoning_tokens,
):
    parser = parser_cls(FakeTokenizer())

    assert parser.count_reasoning_tokens(token_ids) == expected_reasoning_tokens


def test_update_usage_tracking_state_tracks_prompt_and_completion_tokens():
    state = usage_patch._create_usage_tracking_state(
        num_choices=2,
        reasoning_parser=None,
    )

    res = SimpleNamespace(
        prompt_token_ids=[1, 2],
        encoder_prompt_token_ids=[3],
        num_cached_tokens=4,
        outputs=[
            SimpleNamespace(index=0, token_ids=(10, 11)),
            SimpleNamespace(index=1, token_ids=[20]),
        ],
    )

    usage_patch._update_usage_tracking_state(state, res)

    assert state.num_prompt_tokens == 3
    assert state.num_cached_tokens == 4
    assert state.completion_tokens == [2, 1]
    assert state.raw_output_token_ids == [[10, 11], [20]]


def test_make_usage_info_injects_reasoning_token_details():
    fake_serving = SimpleNamespace(enable_prompt_tokens_details=True)
    usage = usage_patch._make_usage_info(
        fake_serving,
        prompt_tokens=3,
        completion_tokens=4,
        num_cached_tokens=1,
        reasoning_tokens=2,
    )

    payload = usage.model_dump(exclude_none=True)

    assert payload["completion_tokens_details"]["reasoning_tokens"] == 2
    assert payload["prompt_tokens_details"]["cached_tokens"] == 1


def test_make_usage_info_injects_zero_cached_tokens():
    fake_serving = SimpleNamespace(enable_prompt_tokens_details=True)
    usage = usage_patch._make_usage_info(
        fake_serving,
        prompt_tokens=3,
        completion_tokens=4,
        num_cached_tokens=0,
    )

    payload = usage.model_dump(exclude_none=True)

    assert payload["prompt_tokens_details"]["cached_tokens"] == 0


def test_make_full_response_usage_sums_reasoning_tokens():
    class FakeServing:
        enable_prompt_tokens_details = False

        def _make_usage_info(self, **kwargs):
            return usage_patch._make_usage_info(self, **kwargs)

    class FakeReasoningParser:
        def count_reasoning_tokens(self, token_ids):
            return 2 if 2 in token_ids else 0

    state = usage_patch._create_usage_tracking_state(
        num_choices=2,
        reasoning_parser=FakeReasoningParser(),
    )
    state.num_prompt_tokens = 3
    state.num_cached_tokens = 1
    state.final_res = SimpleNamespace(num_cached_tokens=1)
    state.completion_tokens = [4, 2]
    state.raw_output_token_ids = [[10, 11, 2, 20], [30, 31]]

    usage = usage_patch._make_full_response_usage(FakeServing(), state)

    assert usage.prompt_tokens == 3
    assert usage.completion_tokens == 6
    assert usage.total_tokens == 9
    assert usage.completion_tokens_details.reasoning_tokens == 2
    assert usage.prompt_tokens_details is None


def test_make_full_response_usage_accepts_wrapped_reasoning_parser():
    class FakeServing:
        enable_prompt_tokens_details = False

        def _make_usage_info(self, **kwargs):
            return usage_patch._make_usage_info(self, **kwargs)

    class FakeReasoningParser:
        def count_reasoning_tokens(self, token_ids):
            return token_ids.index(2) if 2 in token_ids else len(token_ids)

    state = usage_patch._create_usage_tracking_state(
        num_choices=1,
        reasoning_parser=SimpleNamespace(reasoning_parser=FakeReasoningParser()),
    )
    state.num_prompt_tokens = 3
    state.final_res = SimpleNamespace(num_cached_tokens=None)
    state.completion_tokens = [4]
    state.raw_output_token_ids = [[10, 11, 2, 20]]

    usage = usage_patch._make_full_response_usage(FakeServing(), state)

    assert usage.completion_tokens_details.reasoning_tokens == 2


def test_count_reasoning_tokens_accepts_minimax_unified_parser():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="minimax_m2",
        reasoning_parser_name="minimax_m2",
        enable_auto_tools=True,
        model_name="MiniMax-M2",
    )
    parser = parser_cls(FakeTokenizer(), tools=[])

    assert not hasattr(parser, "count_reasoning_tokens")
    assert usage_patch._count_reasoning_tokens_for_usage([10, 11, 2, 20], parser) == 2


def test_count_reasoning_tokens_accepts_deepseek_parser_manager_wrapper():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v4",
        reasoning_parser_name="deepseek_v4",
        enable_auto_tools=True,
        model_name="DeepSeek-V4",
    )
    parser = parser_cls(FakeTokenizer(), tools=[])

    assert not hasattr(parser, "count_reasoning_tokens")
    assert usage_patch._count_reasoning_tokens_for_usage([10, 11], parser) == 0


def test_stream_usage_details_are_injected_without_replacing_source():
    class FakeReasoningParser:
        def count_reasoning_tokens(self, token_ids):
            return token_ids.index(2) if 2 in token_ids else len(token_ids)

    state = usage_patch._create_usage_tracking_state(
        num_choices=1,
        reasoning_parser=FakeReasoningParser(),
        enable_prompt_tokens_details=True,
    )
    state.num_cached_tokens = 0
    state.raw_output_token_ids = [[10, 11, 2, 20]]

    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
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
    assert payload["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 0,
    }
    assert not hasattr(usage_patch, "_extract_class_method_source")
    assert not hasattr(usage_patch, "_patch_chat_completion_stream_generator")


def test_stream_usage_details_inject_prompt_details_without_reasoning():
    state = usage_patch._create_usage_tracking_state(
        num_choices=1,
        reasoning_parser=None,
        enable_prompt_tokens_details=True,
    )
    state.num_cached_tokens = 0

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

    data = usage_patch._inject_stream_usage_details(
        f"data: {json.dumps(chunk)}\n\n",
        state,
    )
    payload = json.loads(data.removeprefix("data: ").removesuffix("\n\n"))

    assert payload["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 0,
    }
    assert "completion_tokens_details" not in payload["usage"]
