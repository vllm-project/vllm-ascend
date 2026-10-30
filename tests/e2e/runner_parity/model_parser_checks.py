# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Parse actual model generations using its own chat template and GLM parsers."""

import json


def run_model_parser_checks(llm, output):
    from vllm import SamplingParams
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    tokenizer = llm.get_tokenizer()
    eos = llm.llm_engine.model_config.hf_text_config.eos_token_id
    stop_ids = [eos] if isinstance(eos, int) else list(eos)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get the current weather for the requested city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    cases = [
        ("reasoning", "Calculate 1+1. Think in fewer than 40 tokens, then give a brief final answer.", True, None),
        ("tool", "Use the weather tool to get the current weather in Beijing. Call it now.", False, tools),
    ]
    checks = []
    for name, question, thinking, definitions in cases:
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=definitions,
            add_generation_prompt=True,
            enable_thinking=thinking,
            tokenize=True,
            return_dict=False,
        )
        assert isinstance(prompt, list) and all(isinstance(token, int) for token in prompt)
        sample = llm.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(
                temperature=0,
                seed=1024,
                max_tokens=256,
                stop_token_ids=stop_ids,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            ),
            use_tqdm=False,
        )[0].outputs[0]
        row = {
            "name": name,
            "prompt_token_ids": prompt,
            "generated_token_ids": list(sample.token_ids),
            "raw_text": sample.text,
            "finish_reason": sample.finish_reason,
            "stop_reason": sample.stop_reason,
            "enable_thinking": thinking,
            "passed": False,
        }
        try:
            request = ChatCompletionRequest(model="glm51", messages=messages, tools=definitions)
            if name == "reasoning":
                parser = ReasoningParserManager.get_reasoning_parser("glm45")(tokenizer)
                reason, content = parser.extract_reasoning(sample.text, request)
                row.update(parser=type(parser).__name__, reasoning=reason, content=content)
                assert reason and reason.strip() and content and content.strip(), row
            else:
                parser = ToolParserManager.get_tool_parser("glm47")(tokenizer)
                result = parser.extract_tool_calls(sample.text, request)
                row.update(parser=type(parser).__name__, parsed=result.model_dump())
                assert result.tools_called and len(result.tool_calls) == 1, row
                call = result.tool_calls[0].function
                assert call.name == "weather" and json.loads(call.arguments) == {"city": "Beijing"}, row
            assert sample.finish_reason == "stop", row
            row["passed"] = True
        except Exception as error:
            row["error"] = type(error).__name__ + ": " + str(error)
        checks.append(row)
        output.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"actual_model_parser": name, "passed": row["passed"]}), flush=True)
    assert all(row["passed"] for row in checks), "Inspect preserved real-model parser outputs"
    return checks
