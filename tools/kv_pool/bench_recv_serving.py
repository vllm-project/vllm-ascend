#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure a primed external-prefix-cache workload through OpenAI streaming.

Run with VLLM_SERVER_DEV_MODE=1 and a local test server. Every prompt is primed
before the local prefix cache is reset. Save server metrics before and after;
verify external cache hits before interpreting this as a receive benchmark.
"""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

CACHE_RESET_ATTEMPTS = 20
CACHE_RESET_RETRY_S = 0.1
METRICS_TIMEOUT_S = 10
METRICS_POLL_S = 0.2


def request(base, path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=payload, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=180)


def complete(base, model, prompt, max_tokens):
    start = time.perf_counter_ns()
    first = None
    text = ""
    usage = None
    request_id = None
    with request(
        base,
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            request_id = chunk.get("id", request_id)
            for choice in chunk.get("choices", []):
                delta = choice.get("text", "")
                if delta:
                    if first is None:
                        first = time.perf_counter_ns()
                    text += delta
            if chunk.get("usage"):
                usage = chunk["usage"]
    end = time.perf_counter_ns()
    if first is None or usage is None:
        raise RuntimeError("stream omitted content or usage")
    generated = usage["completion_tokens"]
    return {
        "request_id": request_id,
        "start_ns": start,
        "end_ns": end,
        "ttft_ms": (first - start) / 1e6,
        "latency_ms": (end - start) / 1e6,
        "tpot_ms": (end - first) / 1e6 / (generated - 1) if generated > 1 else None,
        "usage": usage,
        "text": text,
    }


def percentile(values, p):
    values = sorted(values)
    i = (len(values) - 1) * p / 100
    lo = int(i)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (i - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--prime-base-url", help="Optional separate writer service sharing the same Store")
    parser.add_argument("--model", default="hol-probe")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompt-repeats", type=int, default=64)
    parser.add_argument(
        "--prompt-prefix", default="Document", help="Use a distinct prefix for independent cache fixtures"
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("requests and concurrency must be positive")
    prompts = [
        f"{args.prompt_prefix} {i:08d}. "
        + "The river crosses a quiet town. " * args.prompt_repeats
        + "Summarize this document in one sentence:"
        for i in range(args.requests)
    ]
    for prompt in prompts:
        complete(args.prime_base_url or args.base_url, args.model, prompt, 1)
    # Explicit local-cache reset is outside the timed interval.
    for _ in range(CACHE_RESET_ATTEMPTS):
        with request(args.base_url, "/reset_prefix_cache", {}) as response:
            reset_status = response.status
            reset_success = json.load(response).get("success") is True
        if reset_success:
            break
        time.sleep(CACHE_RESET_RETRY_S)
    else:
        raise RuntimeError("Local prefix cache reset did not complete")
    with request(args.base_url, "/metrics") as response:
        before = response.read().decode()
    start = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(
            executor.map(lambda prompt: complete(args.base_url, args.model, prompt, args.max_tokens), prompts)
        )
    end = time.perf_counter_ns()
    seconds = (end - start) / 1e9

    def external_hits(metrics):
        return sum(
            float(line.split()[-1])
            for line in metrics.splitlines()
            if line.startswith("vllm:external_prefix_cache_hits_total")
        )

    # Engine statistics can be published after the final stream completes.
    # Polling happens outside the measured request interval.
    deadline = time.monotonic() + METRICS_TIMEOUT_S
    while True:
        with request(args.base_url, "/metrics") as response:
            after = response.read().decode()
        hits_delta = external_hits(after) - external_hits(before)
        if hits_delta > 0 or time.monotonic() >= deadline:
            break
        time.sleep(METRICS_POLL_S)
    tokens = sum(row["usage"]["completion_tokens"] for row in results)
    output = {
        "configuration": {**vars(args), "output": str(args.output)},
        "external_cache_hits_delta": hits_delta,
        "reset_status": reset_status,
        "elapsed_s": seconds,
        "measurement_start_ns": start,
        "measurement_end_ns": end,
        "clock": "perf_counter_ns",
        "requests_per_s": len(results) / seconds,
        "output_tokens_per_s": tokens / seconds,
        "ttft_ms": {f"p{p}": percentile([row["ttft_ms"] for row in results], p) for p in (50, 95, 99)},
        "requests": results,
        "metrics_before": before,
        "metrics_after": after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    if hits_delta <= 0:
        raise RuntimeError("No external prefix-cache hit observed; this run does not establish receive performance")


if __name__ == "__main__":
    main()
