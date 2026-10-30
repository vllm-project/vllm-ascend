# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Check GLM parser formats with the checkpoint tokenizer, without model inference."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # These imports belong to the isolated, pinned runtime, not the CPU collector.
    import torch
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    reasoning_parser = ReasoningParserManager.get_reasoning_parser("glm45")(tokenizer)
    tool_parser = ToolParserManager.get_tool_parser("glm47")(tokenizer)
    request = ChatCompletionRequest(model="glm51", messages=[{"role": "user", "content": "1+1"}])
    reasoning = reasoning_parser.extract_reasoning("<think>1+1 equals 2.</think>2", request)
    tool = tool_parser.extract_tool_calls(
        "<tool_call>weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>", request
    )
    assert reasoning == ("1+1 equals 2.", "2")
    assert tool.tools_called and len(tool.tool_calls) == 1
    assert tool.tool_calls[0].function.name == "weather"
    assert json.loads(tool.tool_calls[0].function.arguments) == {"city": "Beijing"}
    npu = getattr(torch, "npu", None)
    assert npu is None or not npu.is_initialized()
    result = {
        "reasoning_parser": type(reasoning_parser).__name__,
        "tool_parser": type(tool_parser).__name__,
        "reasoning": reasoning,
        "tool": tool.model_dump(),
        "npu_initialized": False,
        "scope": "Synthetic parser formats and actual checkpoint tokenizer; no real-model generation claim.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"parser_cases_passed": 2, "npu_initialized": False}))


if __name__ == "__main__":
    main()
