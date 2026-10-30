# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Collect bounded, reproducible precision and serving performance for one arm."""

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

from client import post_json, stream_completion

SEED = 1024
MODEL = "glm51-runner-parity"
LANGUAGES = (
    ("zh", "以下是参考资料，最后请回答问题。", ("请计算17加25，只写答案。", "请用一句话说明什么是专家并行。")),
    (
        "en",
        "Reference notes follow; answer the final question.",
        ("Compute 17 plus 25. Give the answer.", "Explain expert parallelism in one sentence."),
    ),
    (
        "code",
        "# Reference source code and comments follow.",
        ("def add(a, b):\n    return", "def first_even(xs):\n    # Return the first even integer, or None.\n"),
    ),
)
LENGTHS = (128, 2048, 32768)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_inputs(base_url):
    cases = []
    for language, filler, questions in LANGUAGES:
        padding = post_json(base_url, "/tokenize", {"model": MODEL, "prompt": (filler + "\n") * 18000})["tokens"]
        for length in LENGTHS:
            for variant, question in enumerate(questions):
                suffix = post_json(base_url, "/tokenize", {"model": MODEL, "prompt": "\n\n" + question})["tokens"]
                tokens = padding[: length - len(suffix)] + suffix
                assert len(tokens) == length
                cases.append(
                    {"id": f"{language}-{length}-{variant}", "language": language, "length": length, "tokens": tokens}
                )
    return {"schema": 1, "seed": SEED, "cases": cases, "sha256": digest(cases)}


def append_record(path, record):
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")


def metrics(base_url):
    with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=20) as response:
        return response.read().decode()


def precision(base_url, inputs, output, runner, output_tokens):
    path = output / "precision.jsonl"
    completed = set()
    if path.exists():
        completed = {json.loads(line)["key"] for line in path.read_text().splitlines()}
    for concurrency in (1, 8):
        for mode in ("greedy", "sample"):

            def run(case, mode=mode, concurrency=concurrency):
                key = f"{case['id']}/{mode}/c{concurrency}"
                if key in completed:
                    return None
                body = {
                    "model": MODEL,
                    "prompt": case["tokens"],
                    "max_tokens": output_tokens,
                    "temperature": 0 if mode == "greedy" else 0.6,
                    "top_p": 1 if mode == "greedy" else 0.95,
                    "top_k": -1 if mode == "greedy" else 10,
                    "seed": SEED,
                    "ignore_eos": True,
                    "logprobs": 20,
                    "return_tokens_as_token_ids": True,
                    "return_token_ids": True,
                }
                start = time.perf_counter()
                response = post_json(base_url, "/v1/completions", body)
                choice = response["choices"][0]
                assert response["usage"]["prompt_tokens"] == case["length"]
                assert choice.get("token_ids") and choice.get("logprobs"), response
                return {
                    "key": key,
                    "runner": runner,
                    "input_sha256": digest(case["tokens"]),
                    "request": body,
                    "response": response,
                    "elapsed_s": time.perf_counter() - start,
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                for record in pool.map(run, inputs["cases"]):
                    if record:
                        append_record(path, record)
                        print(json.dumps({"precision_completed": record["key"]}), flush=True)
    # Fixed histories: do not compare logits after autoregressive histories diverge.
    for case in inputs["cases"]:
        if case["length"] != 128:
            continue
        key = f"{case['id']}/teacher"
        if key in completed:
            continue
        body = {
            "model": MODEL,
            "prompt": case["tokens"],
            "max_tokens": 1,
            "temperature": 0,
            "seed": SEED,
            "prompt_logprobs": 5,
            "logprobs": 5,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
        response = post_json(base_url, "/v1/completions", body)
        if not response["choices"][0].get("prompt_logprobs"):
            append_record(output / "errors-precision.jsonl", {"key": key, "request": body, "response": response})
            raise AssertionError("Prompt logprobs are required in the completion choice for fixed-history checks")
        append_record(
            path,
            {
                "key": key,
                "runner": runner,
                "input_sha256": digest(case["tokens"]),
                "request": body,
                "response": response,
            },
        )
        print(json.dumps({"precision_completed": key}), flush=True)


def performance(base_url, inputs, output, runner, rounds, output_tokens, round_offset=0):
    prompts = {case["length"]: case["tokens"] for case in inputs["cases"]}
    # 8192 is an explicit performance input length, created from the same corpus.
    prompts[8192] = prompts[32768][:8192]
    result = output / "performance.jsonl"
    done = set()
    if result.exists():
        done = {json.loads(line)["key"] for line in result.read_text().splitlines()}
    for round_index in range(round_offset, round_offset + rounds):
        for length in (128, 2048, 8192):
            for concurrency in (1, 8, 32):
                key = f"r{round_index}/isl{length}/c{concurrency}"
                if key in done:
                    continue
                body = {
                    "model": MODEL,
                    "prompt": prompts[length],
                    "temperature": 0,
                    "seed": SEED,
                    "max_tokens": output_tokens,
                    "ignore_eos": True,
                    "return_token_ids": True,
                }
                # Warm the actual request path. Never count warmups as measurements.
                warmups = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    warmups = list(
                        pool.map(
                            lambda _, body=body: stream_completion(base_url, body, error_dir=output), range(concurrency)
                        )
                    )
                before = metrics(base_url)
                count = max(concurrency, 8)
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    samples = list(
                        pool.map(lambda _, body=body: stream_completion(base_url, body, error_dir=output), range(count))
                    )
                elapsed = time.perf_counter() - start
                after = metrics(base_url)
                for sample in samples:
                    assert sample["usage"]["prompt_tokens"] == length
                    assert sample["usage"]["completion_tokens"] == output_tokens
                safe_key = key.replace("/", "-")
                (output / f"metrics-{safe_key}-before.txt").write_text(before)
                (output / f"metrics-{safe_key}-after.txt").write_text(after)
                append_record(
                    result,
                    {
                        "key": key,
                        "runner": runner,
                        "input_sha256": digest(prompts[length]),
                        "client_concurrency": concurrency,
                        "request_count": count,
                        "input_tokens": length,
                        "output_tokens": output_tokens,
                        "wall_s": elapsed,
                        "output_tps": sum(s["usage"]["completion_tokens"] for s in samples) / elapsed,
                        "input_tps_serving": sum(s["usage"]["prompt_tokens"] for s in samples) / elapsed,
                        "ttft_median_s": statistics.median(s["ttft_s"] for s in samples),
                        "tpot_median_s": statistics.median(s["tpot_s"] for s in samples),
                        "e2e_median_s": statistics.median(s["e2e_s"] for s in samples),
                        "warmup_count": len(warmups),
                        "request": body,
                        "samples": samples,
                        "scope": "Serving wall clock; concurrency is not asserted to equal actual engine batch.",
                    },
                )
                print(
                    json.dumps({"performance_completed": key, "output_tps": count * output_tokens / elapsed}),
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8131")
    parser.add_argument("--runner", choices=("v1", "v2"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("precision", "performance", "all"), default="all")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--round-offset", type=int, default=0)
    parser.add_argument("--precision-output-tokens", type=int, default=32)
    parser.add_argument("--performance-output-tokens", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.inputs.exists():
        args.inputs.write_text(json.dumps(make_inputs(args.base_url), ensure_ascii=False))
    inputs = json.loads(args.inputs.read_text())
    assert digest(inputs["cases"]) == inputs["sha256"], "Input corpus has changed"
    if args.stage in ("precision", "all"):
        precision(args.base_url, inputs, args.output, args.runner, args.precision_output_tokens)
    if args.stage in ("performance", "all"):
        performance(
            args.base_url,
            inputs,
            args.output,
            args.runner,
            args.rounds,
            args.performance_output_tokens,
            args.round_offset,
        )


if __name__ == "__main__":
    main()
