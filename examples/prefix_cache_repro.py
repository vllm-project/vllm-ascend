#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Reproduce output drift between cold, warm, and concurrent requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=520)
    parser.add_argument("--leading-marker", default="")
    parser.add_argument("--serial-requests", type=int, default=3)
    parser.add_argument("--concurrent-requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    prompt = args.leading_marker + "Async prefix cache validation. " * args.repeats + "\nGive the capital of France."
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": args.max_tokens,
    }

    def request(index: int, phase: str) -> dict[str, Any]:
        start = time.perf_counter()
        body = json.dumps(payload).encode()
        http_request = urllib_request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib_request.urlopen(http_request, timeout=args.timeout) as response:
                status = response.status
                data = json.load(response)
        except urllib_error.HTTPError as response:
            status = response.code
            data = json.load(response)
        choices = data.get("choices") or [{}]
        return {
            "phase": phase,
            "index": index,
            "status": status,
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            "text": choices[0].get("message", {}).get("content"),
            "prompt_tokens": (data.get("usage") or {}).get("prompt_tokens"),
            "error": data.get("error"),
        }

    rows = [request(index, "serial") for index in range(args.serial_requests)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent_requests) as executor:
        rows.extend(
            executor.map(
                lambda index: request(index, "concurrent"),
                range(args.concurrent_requests),
            )
        )

    outputs = {row["text"] for row in rows}
    report = {
        "repeats": args.repeats,
        "leading_marker": args.leading_marker,
        "consistent": len(outputs) == 1,
        "distinct_outputs": sorted(outputs, key=repr),
        "requests": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["consistent"] else 1


if __name__ == "__main__":
    sys.exit(main())
