# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched single-instance HTTP workload; not the 3P1D SLO benchmark."""

import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

import aiohttp
import numpy as np
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput, async_request_openai_chat_completions


async def main(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[])
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:

        async def spec_metrics():
            async with session.get(args.url + "/metrics") as response:
                response.raise_for_status()
                text = await response.text()
            return {
                line.split()[0]: float(line.split()[1])
                for line in text.splitlines()
                if line and not line.startswith("#") and "spec_decode" in line and "_total" in line.split()[0]
            }

        async with session.get(args.url + "/health") as response:
            response.raise_for_status()
        for words in args.input_words:
            for concurrency in args.concurrency:
                semaphore = asyncio.Semaphore(concurrency)

                async def request(index, count, *, words=words, semaphore=semaphore):
                    rng = random.Random(5511 + index)
                    text = " ".join(
                        rng.choices(["cache", "memory", "token", "kernel", "model", "data", "graph", "table"], k=words)
                    )
                    messages = [
                        {
                            "role": "user",
                            "content": f"Request {index}. Read the following notes "
                            "and explain their implications in detail. " + text,
                        }
                    ]
                    extra = dict(temperature=0, seed=1024, ignore_eos=True)
                    body = dict(model="engram-wkv", messages=messages, max_tokens=count, **extra)
                    request_input = RequestFuncInput(
                        prompt="",
                        prompt_len=0,
                        output_len=count,
                        model="engram-wkv",
                        api_url=args.url + "/v1/chat/completions",
                        chat_messages=messages,
                        extra_body=extra,
                    )
                    async with semaphore:
                        item = await async_request_openai_chat_completions(request_input, session)
                    return dict(
                        index=index,
                        payload_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                        success=item.success,
                        error=item.error,
                        prompt_tokens=item.prompt_len,
                        output_tokens=item.output_tokens,
                        latency_s=item.latency,
                        ttft_s=item.ttft,
                        tpot_s=(item.latency - item.ttft) / (item.output_tokens - 1)
                        if item.output_tokens > 1
                        else None,
                    )

                warm = await asyncio.gather(*(request(i, 32) for i in range(concurrency * 2)))
                if not all(r["success"] for r in warm):
                    raise RuntimeError(("warmup failed", [r["error"] for r in warm if not r["success"]]))
                before = await spec_metrics()
                started = time.perf_counter()
                rows = await asyncio.gather(*(request(i, args.output_tokens) for i in range(concurrency * 3)))
                duration = time.perf_counter() - started
                case = dict(input_words=words, concurrency=concurrency, duration_s=duration, requests=rows)
                after = await spec_metrics()
                case["speculative_counter_delta"] = {key: value - before.get(key, 0) for key, value in after.items()}
                valid = [
                    r for r in rows if r["success"] and r["output_tokens"] == args.output_tokens and r["ttft_s"] > 0
                ]
                case.update(
                    completed=len(valid),
                    requested=len(rows),
                    qps=len(valid) / duration,
                    tokens_s=sum(r["output_tokens"] for r in valid) / duration,
                    ttft_p50_s=float(np.median([r["ttft_s"] for r in valid])) if valid else None,
                    tpot_p50_ms=float(np.median([r["tpot_s"] for r in valid])) * 1000 if valid else None,
                )
                results["cases"].append(case)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(
                    json.dumps({k: v for k, v in case.items() if k not in ("requests", "speculative_counter_delta")}),
                    flush=True,
                )
                if len(valid) != len(rows):
                    raise RuntimeError("invalid/failed requests; saved, do not compare as a valid workload")
    print("WKV_SERVICE_BENCH_PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8089")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-words", type=int, nargs="+", default=[256, 2048])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--output-tokens", type=int, default=128)
    asyncio.run(main(parser.parse_args()))
