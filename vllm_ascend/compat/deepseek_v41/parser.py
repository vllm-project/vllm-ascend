# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 reasoning and spaced DSML tool calls."""

import contextlib
import functools
import json
from dataclasses import replace

import regex as re
from vllm.parser.deepseek_v4 import (
    DeepSeekV4Parser,
    _unwrap_wrapper_args,
    deepseek_v4_config,
)
from vllm.parser.engine.parser_engine_config import ParserEngineConfig

DSML_TOOL_START = "<｜DSML｜ calls>"
DSML_TOOL_END = "</｜DSML｜ calls>"
DSML_INVOKE_PREFIX = '<｜DSML｜ invoke name="'
DSML_INVOKE_END = "</｜DSML｜ invoke>"
DSML_PARAM_START = "<｜DSML｜ parameter"
DSML_PARAM_CLOSE = "</｜DSML｜ parameter>"

_PARAM_RE = re.compile(
    r'<｜DSML｜ parameter\s+name="([^"]+)"\s+string="(true|false)">'
    r"(.*?)"
    r"(?:</｜DSML｜ parameter>|(?=<｜DSML｜ parameter\s+name=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    r'<｜DSML｜ parameter\s+name="([^"]+)"\s+string="(true|false)">'
    r"(.*)$",
    re.DOTALL,
)


@functools.cache
def deepseek_v41_config(thinking: bool = False) -> ParserEngineConfig:
    config = deepseek_v4_config(thinking=thinking)
    terminal_overrides = {
        "TOOL_START": DSML_TOOL_START,
        "TOOL_END": DSML_TOOL_END,
        "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
        "INVOKE_END": DSML_INVOKE_END,
        "PARAM_START": DSML_PARAM_START,
        "PARAM_CLOSE": DSML_PARAM_CLOSE,
    }
    return replace(
        config,
        name="deepseek_v41",
        terminals={**config.terminals, **terminal_overrides},
        token_id_terminals={
            key: terminal_overrides.get(key, value) for key, value in config.token_id_terminals.items()
        },
        arg_converter=functools.partial(
            _dsml_arg_converter_v41,
        ),
    )


def _dsml_arg_converter_v41(raw_args: str, partial: bool) -> str:
    """Convert V4.1's spaced DSML parameters without patching vLLM's V4 parser."""
    params: dict[str, object] = {}
    last_end = 0
    for match in _PARAM_RE.finditer(raw_args):
        name, is_str, value = match.group(1), match.group(2), match.group(3)
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value
        last_end = match.end()

    if partial:
        partial_match = _PARTIAL_PARAM_RE.search(raw_args, last_end)
        if partial_match:
            name, is_str, value = partial_match.group(1), partial_match.group(2), partial_match.group(3)
            if is_str == "true":
                params[name] = value
            else:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    params[name] = json.loads(value)

    return json.dumps(params, ensure_ascii=False)


class DeepSeekV41Parser(DeepSeekV4Parser):
    """V4.1 parser compatible with the non-subclassable v0.29 V4 parser."""

    def __init__(self, tokenizer, tools=None, **kwargs) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = bool(chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking"))
        if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
            thinking = True
        thinking = thinking and chat_kwargs.get("reasoning_effort") != "none"
        # Skip DeepSeekV4Parser.__init__: v0.29 hard-codes deepseek_v4_config.
        super(DeepSeekV4Parser, self).__init__(
            tokenizer,
            tools,
            parser_engine_config=deepseek_v41_config(thinking=thinking),
            **kwargs,
        )
        self._arg_converter = self._convert_args

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        result = _dsml_arg_converter_v41(raw_args, partial)
        if not self._tools:
            return result
        func_name = next((slot.name for slot in self._tool_slots if slot.args == raw_args), None)
        return _unwrap_wrapper_args(result, self._tools, func_name)
