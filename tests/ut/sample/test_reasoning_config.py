# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.parser.engine.parser_engine_config import ParserState

import vllm_ascend.sample.reasoning_config as reasoning_config_module
from vllm_ascend.sample.reasoning_config import (
    AscendReasoningConfig,
    _reasoning_exit_strs,
)


class FakeTokenizer:
    token_ids = {"<think>": [90, 91], "</think>": [92, 93]}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return self.token_ids[text]


def test_initialize_token_ids_from_explicit_reasoning_markers(monkeypatch):
    monkeypatch.setattr(
        reasoning_config_module,
        "cached_tokenizer_from_config",
        lambda model_config: FakeTokenizer(),
    )
    config = AscendReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str="</think>",
        premature_eos_policy="mask_in_reasoning",
    )

    config.initialize_token_ids(SimpleNamespace())

    assert config.enabled
    assert config.reasoning_start_token_ids == [90, 91]
    assert config.reasoning_exit_token_ids == [[92, 93]]


def test_parser_engine_reasoning_exits_include_tool_transition():
    parser_config = SimpleNamespace(
        transitions={
            (ParserState.REASONING, "end"): SimpleNamespace(next_state=ParserState.CONTENT),
            (ParserState.REASONING, "tool"): SimpleNamespace(next_state=ParserState.TOOL_NAME),
            (ParserState.REASONING, "stay"): SimpleNamespace(next_state=ParserState.REASONING),
        },
        terminals={
            "end": "</think>",
            "tool": "<tool>",
            "stay": "continue",
        },
    )
    parser = SimpleNamespace(_parser_engine=SimpleNamespace(parser_engine_config=parser_config))

    assert _reasoning_exit_strs(parser) == ("</think>", "<tool>")
