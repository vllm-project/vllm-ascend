# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Semantic smoke test through the GLM-5.3 PD proxy, including repeated requests."""

import argparse
import json

import regex as re
import requests

REQUEST_TIMEOUT_SECONDS = 300
MAX_OUTPUT_TOKENS = 4096
PROBES = (("17 * 23", "391"), ("12 * 12", "144"), ("17 * 23", "391"))


def validate_response(response: dict, expected: str) -> None:
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Expected one completed answer, without truncation or tool calls")
    answer = choices[0].get("message", {}).get("content")
    if not isinstance(answer, str) or not re.fullmatch(rf"\s*{re.escape(expected)}[.!]?\s*", answer):
        raise ValueError(f"Expected answer {expected!r}, received {answer!r}")
    usage = response.get("usage", {})
    if usage.get("prompt_tokens", 0) <= 0 or usage.get("completion_tokens", 0) <= 0:
        raise ValueError("Expected positive prompt and completion token usage")


def run_probes(base_url: str) -> None:
    with requests.Session() as session:
        models = session.get(f"{base_url}/models", timeout=REQUEST_TIMEOUT_SECONDS)
        models.raise_for_status()
        model = models.json()["data"][0]["id"]
        for expression, expected in PROBES:
            response = session.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": f"Calculate {expression}. Reply with only the decimal integer."}
                    ],
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": 0,
                    "seed": 1024,
                    "stream": False,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            print(json.dumps({"expression": expression, "response": payload}), flush=True)
            validate_response(payload, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    run_probes(f"http://127.0.0.1:{args.port}/v1")
