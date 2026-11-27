# SPDX-License-Identifier: Apache-2.0
"""Qwen3 non-streaming reasoning boundary correction for vllm-ascend#23033."""

from dataclasses import replace

from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.qwen3 import PARAM_END, PARAM_START, THINK_END, TOOL_CALL_START, Qwen3Parser
from vllm.reasoning.qwen3_engine_reasoning_parser import Qwen3ParserReasoningAdapter


class AscendQwen3ReasoningParser(Qwen3ParserReasoningAdapter):
    def extract_reasoning(self, model_output, request):
        engine = self._parser_engine
        first_tool = model_output.find(TOOL_CALL_START)
        end = model_output.find(THINK_END)
        self._counting_parser_engine = None
        if (
            not engine.thinking_enabled
            or getattr(engine, "_prompt_closes_reasoning", False)
            or not 0 <= first_tool < end
            # An implicit tool call can legitimately include </think> as an argument.
            or model_output.rfind(PARAM_START, first_tool, end) > model_output.rfind(PARAM_END, first_tool, end)
        ):
            return super().extract_reasoning(model_output, request)

        # Complete output has an explicit boundary. A tool example quoted before
        # it belongs to reasoning, even if its name is a real tool on this request.
        config = engine.parser_engine_config
        transitions = config.transitions.copy()
        transitions.pop((ParserState.REASONING, "TOOL_START"), None)
        explicit = Qwen3Parser(self.model_tokenizer, parser_engine_config=replace(config, transitions=transitions))
        explicit.skip_tool_parsing = True
        self._streaming_count_valid = False
        # The serving layer recounts using the original token IDs after parse.
        self._counting_parser_engine = explicit
        return explicit.extract_reasoning(model_output, request)
