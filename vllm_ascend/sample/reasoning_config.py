# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import field
from typing import Literal

from vllm.config.model import ModelConfig
from vllm.config.reasoning import ReasoningConfig
from vllm.config.utils import config
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config


@config
class AscendReasoningConfig(ReasoningConfig):
    """Ascend reasoning configuration with phase-aware EOS policy metadata."""

    premature_eos_policy: Literal["allow", "mask_in_reasoning"] = "allow"
    """Whether to mask model EOS tokens inside an open reasoning phase."""

    _natural_reasoning_end_token_ids: list[int] | None = field(default=None, init=False, repr=False)
    _reasoning_exit_token_ids: list[list[int]] | None = field(default=None, init=False, repr=False)

    @property
    def natural_reasoning_end_token_ids(self) -> list[int] | None:
        return self._natural_reasoning_end_token_ids

    @property
    def reasoning_exit_token_ids(self) -> list[list[int]] | None:
        return self._reasoning_exit_token_ids

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        if (
            self._reasoning_start_token_ids is not None
            and self._reasoning_end_token_ids is not None
            and self._natural_reasoning_end_token_ids is not None
            and self._reasoning_exit_token_ids is not None
        ):
            self._enabled = True
            return

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        reasoning_start_str = self.reasoning_start_str
        reasoning_end_str = self.reasoning_end_str
        natural_reasoning_end_str = ""
        reasoning_exit_strs: tuple[str, ...] = ()
        if self.reasoning_parser:
            parser_cls = ReasoningParserManager.get_reasoning_parser(self.reasoning_parser)
            reasoning_parser = parser_cls(tokenizer)
            start_token = reasoning_parser.reasoning_start_str
            if start_token and not reasoning_start_str:
                reasoning_start_str = start_token

            end_token = reasoning_parser.reasoning_end_str
            if end_token and not reasoning_end_str:
                reasoning_end_str = end_token
            natural_reasoning_end_str = end_token or ""
            reasoning_exit_strs = _reasoning_exit_strs(reasoning_parser)

        if not natural_reasoning_end_str:
            natural_reasoning_end_str = reasoning_end_str
        if not reasoning_start_str or not reasoning_end_str:
            return

        self._reasoning_start_token_ids = tokenizer.encode(reasoning_start_str, add_special_tokens=False)
        self._reasoning_end_token_ids = tokenizer.encode(reasoning_end_str, add_special_tokens=False)
        self._natural_reasoning_end_token_ids = tokenizer.encode(natural_reasoning_end_str, add_special_tokens=False)
        if not reasoning_exit_strs:
            reasoning_exit_strs = (natural_reasoning_end_str,)
        self._reasoning_exit_token_ids = []
        for exit_str in reasoning_exit_strs:
            token_ids = tokenizer.encode(exit_str, add_special_tokens=False)
            if token_ids and token_ids not in self._reasoning_exit_token_ids:
                self._reasoning_exit_token_ids.append(token_ids)

        if (
            not self._reasoning_start_token_ids
            or not self._reasoning_end_token_ids
            or not self._natural_reasoning_end_token_ids
            or not self._reasoning_exit_token_ids
        ):
            raise ValueError(
                "ReasoningConfig: failed to tokenize reasoning strings: "
                f"reasoning_start_str='{self.reasoning_start_str}', "
                f"reasoning_end_str='{self.reasoning_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
        self._enabled = True


def _reasoning_exit_strs(reasoning_parser: object) -> tuple[str, ...]:
    exit_strs = getattr(reasoning_parser, "reasoning_exit_strs", ())
    if exit_strs:
        return tuple(exit_strs)

    parser_engine = getattr(reasoning_parser, "_parser_engine", None)
    parser_config = getattr(parser_engine, "parser_engine_config", None)
    if parser_config is not None:
        exit_strs: list[str] = []
        for (state, terminal), transition in parser_config.transitions.items():
            if (
                state == ParserState.REASONING
                and transition.next_state != ParserState.REASONING
                and terminal in parser_config.terminals
            ):
                exit_strs.append(parser_config.terminals[terminal])
        return tuple(dict.fromkeys(exit_strs))

    end_str = getattr(reasoning_parser, "reasoning_end_str", None)
    return (end_str,) if end_str else ()
