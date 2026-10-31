# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manually check a running MiniMax-M3 service before/after the parser patch.

python tests/e2e/models/minimax_m3_reasoning_client.py \
    --base-url http://127.0.0.1:11223/v1 --model minimax-m3.1-flash
"""

import argparse
import json
import urllib.request


def check_completion(base_url, payload, streaming):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(dict(payload, stream=streaming)).encode(),
        headers={"Content-Type": "application/json"},
    )
    reasoning, content = "", ""
    finish_reason = None
    with urllib.request.urlopen(request, timeout=300) as response:
        if streaming:
            for line in response:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(event["error"])
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
                    content += delta.get("content") or ""
                    finish_reason = choice.get("finish_reason") or finish_reason
        else:
            result = json.load(response)
            choice = result["choices"][0]
            reasoning = choice["message"].get("reasoning") or choice["message"].get("reasoning_content") or ""
            content = choice["message"].get("content") or ""
            finish_reason = choice["finish_reason"]
    print(json.dumps(dict(stream=streaming, reasoning=reasoning, content=content, finish_reason=finish_reason)))
    assert finish_reason == "stop", "Incomplete response; check generation budget/server logs"
    assert content.strip(), "No visible answer"
    assert "<mm:think>" not in content and "</mm:think>" not in content, "Reasoning markers leaked into content"
    if "response_format" in payload:
        assert json.loads(content) == {"answer": 391}
    # A model may legitimately emit a leading closer with no reasoning. Report
    # reasoning above instead of requiring a nonempty field for every request.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11223/v1")
    parser.add_argument("--model", default="minimax-m3.1-flash")
    args = parser.parse_args()
    single_turn = [{"role": "user", "content": "What is 17 * 23? Give only the number in your final answer."}]
    multi_turn = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And Germany?"},
    ]
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }
    for name, messages, extra in (
        ("single-turn", single_turn, {}),
        ("multi-turn", multi_turn, {}),
        ("structured-output", single_turn, {"response_format": schema}),
    ):
        print(name)
        payload = dict(
            model=args.model,
            messages=messages,
            max_tokens=4096,
            temperature=0,
            chat_template_kwargs={"reasoning_effort": "low"},
            **extra,
        )
        for streaming in (False, True):
            check_completion(args.base_url, payload, streaming)


if __name__ == "__main__":
    main()
