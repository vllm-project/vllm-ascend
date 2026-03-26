# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.reasoning.minimax_m2_reasoning_parser import (
    MiniMaxM2AppendThinkReasoningParser,
    MiniMaxM2ReasoningParser,
)


class FakeTokenizer:
    def get_vocab(self):
        return {
            "<think>": 1,
            "</think>": 2,
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
            id="minimax-all-tokens-are-reasoning-before-end-token",
        ),
        pytest.param(
            MiniMaxM2AppendThinkReasoningParser,
            [10, 11, 20],
            3,
            id="append-think-all-tokens-are-reasoning-before-end-token",
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
