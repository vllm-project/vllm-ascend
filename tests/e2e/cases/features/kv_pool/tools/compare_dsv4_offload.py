#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare cold/repeated/changed-suffix/concurrent deterministic completions.

Both URLs must serve the same weights/tokenizer and sampling configuration.
Use max_num_batched_tokens=4096 on the candidate to exercise chunk continuation.
"""

import argparse
import concurrent.futures
import json
import urllib.request


def complete(url, model, prompt, output_tokens, timeout):
    payload = dict(model=model, prompt=prompt, max_tokens=output_tokens, temperature=0, seed=1023, ignore_eos=True)
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return [(choice["text"], choice.get("finish_reason")) for choice in result["choices"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model", default="dsv4")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 4097, 8193, 12289])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    if min(args.lengths) < 2 or args.concurrency < 1:
        parser.error("lengths must be >=2 and concurrency >=1")
    failed = False
    # Integer token prompts make the requested input lengths exact. IDs in
    # this small range are valid ordinary vocabulary IDs for DSV4.
    for phase in ("cold", "repeat", "different_suffix", "concurrent"):
        prompts = []
        for index, length in enumerate(args.lengths):
            prompt = [100 + i % 31 for i in range(length)]
            prompt[-1] = 140 + index + (20 if phase == "different_suffix" else 0)
            prompts.append(prompt)
        if phase == "concurrent":
            prompts *= args.concurrency

        def compare(item, phase=phase):
            index, prompt = item
            try:
                baseline = complete(args.baseline, args.model, prompt, args.output_tokens, args.timeout)
                candidate = complete(args.candidate, args.model, prompt, args.output_tokens, args.timeout)
                return dict(
                    event="comparison", phase=phase, case=index, input_tokens=len(prompt), equal=baseline == candidate
                )
            except Exception as error:
                return dict(
                    event="comparison", phase=phase, case=index, input_tokens=len(prompt), equal=False, error=str(error)
                )

        workers = args.concurrency if phase == "concurrent" else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(compare, enumerate(prompts)):
                print("DSV4_VALIDATION " + json.dumps(result), flush=True)
                failed |= not result["equal"]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
