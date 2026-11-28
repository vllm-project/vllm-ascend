# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure MRV2 streaming decode with logprobs and compare sampled token IDs."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

PROMPTS = (
    "Explain how an online softmax divides its calculation into tiles.",
    "List the first ten primes and explain how to check them.",
    "Describe the difference between local and global memory.",
    "Give an example of a numerically stable reduction.",
)


async def record_request(engine, prompt, params, request_id):
    started = time.perf_counter()
    first_token = None
    first_token_count = 0
    async for output in engine.generate(prompt, params, request_id):
        completion = output.outputs[0]
        now = time.perf_counter()
        if first_token is None and completion.token_ids:
            first_token = now
            first_token_count = len(completion.token_ids)
        if output.finished:
            token_ids = list(completion.token_ids)
            logprobs = completion.logprobs
            if first_token is None or not logprobs or len(logprobs) != len(token_ids):
                raise RuntimeError(f"Incomplete logprob stream for {request_id}")
            decode_count = len(token_ids) - first_token_count
            if decode_count < 1 or now <= first_token:
                raise RuntimeError(f"No observable decode interval for {request_id}")
            return {
                "token_ids": token_ids,
                "selected_logprobs": [step[token].logprob for token, step in zip(token_ids, logprobs)],
                "ttft_ms": 1000 * (first_token - started),
                "tpot_ms": 1000 * (now - first_token) / decode_count,
                "decode_tokens_per_second": decode_count / (now - first_token),
            }
    raise RuntimeError(f"Generation ended without a final response for {request_id}")


async def run(args):
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            tensor_parallel_size=1,
            max_model_len=1024,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.4,
            enforce_eager=True,
        )
    )
    try:
        results = {}
        for name, temperature in (("greedy", 0.0), ("sampled", 0.7)):
            params = SamplingParams(
                temperature=temperature,
                seed=20261126,
                logprobs=5,
                ignore_eos=True,
                max_tokens=64,
            )
            await record_request(engine, PROMPTS[0], params, f"{name}-warmup")
            results[name] = []
            for repeat in range(args.repeats):
                requests = [
                    record_request(engine, prompt, params, f"{name}-{repeat}-{i}") for i, prompt in enumerate(PROMPTS)
                ]
                results[name].append(await asyncio.gather(*requests))
        return {"model": args.model, "runs": results}
    finally:
        engine.shutdown()


def compare(result, reference):
    if result["model"] != reference["model"]:
        raise ValueError("Model paths must match")
    for name, runs in result["runs"].items():
        for actual_run, reference_run in zip(runs, reference["runs"][name], strict=True):
            for actual, expected in zip(actual_run, reference_run, strict=True):
                if actual["token_ids"] != expected["token_ids"]:
                    raise ValueError(f"Token sequence differs in {name} generation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--repeats", default=3, type=int)
    args = parser.parse_args()
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
        parser.error("set VLLM_USE_V2_MODEL_RUNNER=1 before launch")
    if args.repeats < 3:
        parser.error("use at least three repeats")
    result = asyncio.run(run(args))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.reference:
        compare(result, json.loads(args.reference.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
