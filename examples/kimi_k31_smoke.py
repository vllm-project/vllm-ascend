# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record repeated full-model generation responses from a local K3.1 server."""

import argparse
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    responses = []
    for prompt in ("What is 1 + 1?", "用中文简短介绍你自己。", "What is 1 + 1?"):
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "kimi-k3.1-pro",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 128,
                    "temperature": 0,
                    "seed": 1024,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        responses.append({"prompt": prompt, "response": result})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(responses, ensure_ascii=False, indent=2) + "\n")
        message = result["choices"][0]["message"]
        content = message.get("content") or message.get("reasoning") or message.get("reasoning_content")
        if not content or result.get("usage", {}).get("completion_tokens", 0) <= 0:
            raise RuntimeError(f"Empty generation; inspect {args.output} and the server log")
        print(content)
    print(f"Three nonempty responses recorded in {args.output}; review their quality before accepting the model.")


if __name__ == "__main__":
    main()
