# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two running P/D endpoints; see the Mooncake prefill offload guide."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def complete(url, model, prompt, max_tokens):
    request = Request(
        url.rstrip("/") + "/completions",
        data=json.dumps(
            dict(model=model, prompt=prompt, temperature=0, seed=42, max_tokens=max_tokens, logprobs=1)
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=600) as response:
        result = json.load(response)
    choice = result["choices"][0]
    if choice.get("logprobs") is None:
        raise ValueError("The endpoint must return completion logprobs for token comparison")
    return choice["logprobs"]["tokens"], choice["finish_reason"], result["usage"]["prompt_tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-url", required=True, help="OpenAI endpoint including /v1")
    parser.add_argument("--candidate-url", required=True, help="OpenAI endpoint including /v1")
    parser.add_argument("--model", required=True, help="Identical served model name on both endpoints")
    parser.add_argument("--max-tokens", type=int, default=16, help="Use 1 for direct producer endpoints")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if min(args.max_tokens, args.concurrency, args.repeat) < 1:
        parser.error("max-tokens, concurrency and repeat must be positive")
    # Different lengths exercise partial blocks and several prefill chunks.
    # Repeat the common prefix with both identical and distinct suffixes.
    prefix = "A computer stores historical information in memory. " * 500
    prompts = [
        "What is one plus one?",
        prefix + "Summarize the preceding sentence.",
        prefix + "Explain the role of memory.",
        prefix[: len(prefix) // 2] + " Give a short answer.",
    ]
    expected = [complete(args.baseline_url, args.model, prompt, args.max_tokens) for prompt in prompts]
    # Populate the candidate store first, then run mixed concurrent warm hits.
    for index, prompt in enumerate(prompts):
        actual = complete(args.candidate_url, args.model, prompt, args.max_tokens)
        if actual != expected[index]:
            raise AssertionError(f"Cold request {index} differs: {actual!r} != {expected[index]!r}")
    jobs = list(range(len(prompts))) * args.repeat
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(
            executor.map(lambda index: complete(args.candidate_url, args.model, prompts[index], args.max_tokens), jobs)
        )
    for index, actual in zip(jobs, results, strict=True):
        if actual != expected[index]:
            raise AssertionError(f"Warm request {index} differs: {actual!r} != {expected[index]!r}")
    print(f"Matched generated tokens and finish reasons for {len(prompts) + len(jobs)} candidate requests")


if __name__ == "__main__":
    main()
