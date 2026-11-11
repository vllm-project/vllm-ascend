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

"""Compare real-weight replay responses against a replay-disabled service.

Record a baseline before restarting the service, then run the same prompts
with replay enabled. The shared prefix exceeds the SWA window. For additional
cross-chunk coverage, use a prefill chunk smaller than the window. Inspect the service's replay
activation log as well; HTTP success alone does not prove replay was exercised.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path

import requests

LONG_CONTEXT = "This paragraph is background context for a deterministic arithmetic check.\n" * 600
CASES = {
    "short": "What is 2 plus 2? Answer with only the number.",
    "long_add": LONG_CONTEXT + "What is 13 plus 29? Answer with only the number.",
    "long_subtract": LONG_CONTEXT + "What is 91 minus 37? Answer with only the number.",
}
REQUEST_TIMEOUT = 600
REPETITIONS = 8


def request_case(base_url, model, prompt):
    response = requests.post(
        base_url.rstrip("/") + "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 1024,
            "max_tokens": 32,
            "logprobs": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    choice = result["choices"][0]
    assert choice["message"]["content"], result
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "usage": result["usage"],
        "logprobs": choice.get("logprobs"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    results = {}
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    for name, prompt in CASES.items():
        count = REPETITIONS if baseline else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results[name] = list(
                executor.map(
                    lambda _, prompt=prompt: request_case(args.base_url, args.model, prompt),
                    range(count),
                )
            )
        print(name, json.dumps(results[name][0], ensure_ascii=False), flush=True)
        if name.startswith("long"):
            assert results[name][0]["usage"]["prompt_tokens"] > 4096
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    if baseline:
        for name, responses in results.items():
            reference = baseline[name][0]
            assert all(r["text"] == reference["text"] for r in responses), name
            assert all(r["finish_reason"] == reference["finish_reason"] for r in responses), name
        print("PASS: all 24 responses match the replay-disabled baseline", flush=True)


if __name__ == "__main__":
    main()
