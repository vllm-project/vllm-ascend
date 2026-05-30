#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DeepSeek-V4 DSML tool-call fix for tool_choice == "none" (or omitted, which
# defaults to "none" per the OpenAI spec).
#
# Symptom
# -------
# Agent-style clients often send follow-up chat completions WITHOUT declaring
# `tools` / `tool_choice`. An omitted tool_choice defaults to "none". With
# DeepSeek-V4 the model may still emit DSML tool-call markup. Two failure modes:
#   1. empty content ("no_content"): DSML tokens are stripped during decode
#      because skip_special_tokens defaults to True when tool_choice == "none",
#      leaving the assistant content empty.
#   2. dsml-leak: the DSML markup survives decode and leaks verbatim into the
#      assistant content, because the tool parser is never invoked on the
#      "none" path.
#
# Fix
# ---
# Fix A (keep-special): DeepSeek DSML parsers keep DSML tokens during decode
#        unconditionally (skip_special_tokens=False), so the markup is never
#        silently dropped. Prevents the empty-content failure mode.
# Fix B (strip-on-none): when tool_choice is "none"/omitted but the model output
#        still contains the DSML tool_call start token, run the parser to strip
#        the markup out of the returned content. Raw DSML therefore never leaks
#        to the user. tool_calls stay suppressed by the downstream "none" branch,
#        consistent with OpenAI semantics (tool_choice="none" => do not call).
#        Clients that actually want tool calls must send tool_choice="auto"
#        with a tools schema, which already works through the normal path.
# Fix C (parameter-regex arity): the existing patch_deepseek_v4_tool_call_parser
#        re-implements `_parse_invoke_params` to unpack a THREE-group parameter
#        regex (name, string_attr, value), but the base DeepSeekV32ToolParser
#        builds `parameter_complete_regex` with a NON-capturing string group
#        (string="(?:true|false)"), i.e. only two groups. The arity mismatch
#        raises `ValueError: not enough values to unpack (expected 3, got 2)`
#        inside extract_tool_calls, which is swallowed and reported as
#        tools_called=False -- breaking BOTH the auto path and Fix B above.
#        Rebuild the regex with a capturing string group so the parsed calls are
#        actually recovered. Only applied when _parse_invoke_params is the
#        patched (3-group) variant, so the untouched base parser is left alone.

from __future__ import annotations

import regex as re
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.logger import init_logger
from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Fix A: always keep DSML tokens during decode for DSML tool parsers.
# ---------------------------------------------------------------------------
def _patch_adjust_request_keep_special() -> None:
    if getattr(DeepSeekV32ToolParser, "_vllm_ascend_keep_special_patched", False):
        return

    _original_adjust_request = DeepSeekV32ToolParser.adjust_request

    def _patched_adjust_request(self, request):
        request = _original_adjust_request(self, request)
        # Keep DSML tool-call tokens during decode regardless of tool_choice, so
        # the markup is never stripped (empty content) and can be parsed out.
        try:
            request.skip_special_tokens = False
        except Exception:
            pass
        return request

    DeepSeekV32ToolParser.adjust_request = _patched_adjust_request
    DeepSeekV32ToolParser._vllm_ascend_keep_special_patched = True


_patch_adjust_request_keep_special()


# ---------------------------------------------------------------------------
# Fix C: make the parameter regex arity match the patched _parse_invoke_params.
# ---------------------------------------------------------------------------
def _patch_parameter_regex_arity() -> None:
    if getattr(DeepSeekV4ToolParser, "_vllm_ascend_param_regex_patched", False):
        return

    _original_parse_invoke = DeepSeekV4ToolParser._parse_invoke_params
    # Only the patched (3-group) implementation needs the capturing string
    # group; the stock V32 implementation unpacks two values and must be left
    # untouched.
    is_patched_impl = "patch_deepseek_v4_tool_call_parser" in getattr(
        _original_parse_invoke, "__module__", ""
    )

    def _patched_parse_invoke(self, *args, **kwargs):
        if is_patched_impl:
            pat = getattr(self, "parameter_complete_regex", None)
            if pat is not None and "(?:true|false)" in pat.pattern:
                self.parameter_complete_regex = re.compile(
                    pat.pattern.replace("(?:true|false)", "(true|false)"),
                    pat.flags,
                )
        return _original_parse_invoke(self, *args, **kwargs)

    DeepSeekV4ToolParser._parse_invoke_params = _patched_parse_invoke
    DeepSeekV4ToolParser._vllm_ascend_param_regex_patched = True


_patch_parameter_regex_arity()


# ---------------------------------------------------------------------------
# Fix B: when tool_choice is "none"/omitted but the model emitted DSML
# tool-call markup, run the parser to strip the markup out of content.
# ---------------------------------------------------------------------------
_TOOL_CHOICE_NONE = (None, "none")

_original_parse_tool_calls_from_content = (
    OpenAIServing._parse_tool_calls_from_content
)


def _patched_parse_tool_calls_from_content(
    request,
    tokenizer,
    enable_auto_tools,
    tool_parser_cls,
    content=None,
):
    tool_choice = getattr(request, "tool_choice", None)
    if (
        content
        and enable_auto_tools
        and tool_parser_cls is not None
        and tokenizer is not None
        and tool_choice in _TOOL_CHOICE_NONE
    ):
        start_token = getattr(tool_parser_cls, "tool_call_start_token", "")
        if start_token and start_token in content:
            info = None
            try:
                tool_parser = tool_parser_cls(tokenizer, request.tools)
                info = tool_parser.extract_tool_calls(content, request=request)
            except Exception:
                logger.exception(
                    "vllm-ascend: DSML tool-call extraction on tool_choice="
                    "'none' path failed; falling back to default handling."
                )
            if info is not None and info.tools_called:
                calls = [
                    FunctionCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in info.tool_calls
                ]
                new_content = info.content
                if new_content is not None and new_content.strip() == "":
                    new_content = None
                return calls, new_content

    return _original_parse_tool_calls_from_content(
        request=request,
        tokenizer=tokenizer,
        enable_auto_tools=enable_auto_tools,
        tool_parser_cls=tool_parser_cls,
        content=content,
    )


OpenAIServing._parse_tool_calls_from_content = staticmethod(
    _patched_parse_tool_calls_from_content
)
