# SPDX-License-Identifier: Apache-2.0
"""Validate Qwen3 automatic function calls against the request's tools."""

from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from vllm.tool_parsers.utils import find_tool_name


class AscendQwen3ToolParser(Qwen3EngineToolParser):
    def extract_tool_calls(self, model_output, request):
        result = super().extract_tool_calls(model_output, request)
        if request.stream or request.tool_choice not in (None, "auto") or not result.tools_called:
            return result

        tools = request.tools or []
        calls = [call for call in result.tool_calls if find_tool_name(tools, call.function.name)]
        if len(calls) == len(result.tool_calls):
            return result
        return result.model_copy(
            update={
                "tools_called": bool(calls),
                "tool_calls": calls,
                "content": result.content if calls else model_output,
            }
        )
