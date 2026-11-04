#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build curl payloads and retain request-level results using the standard library."""

import argparse
import csv
import json
import time
from pathlib import Path

WINDOW_TOKENS = 128
BACKGROUND = "Background record: verify bounded replay using a long input and a simple arithmetic task.\n"


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def build(args):
    if args.kind == "short":
        prompt = "What is 2 plus 2? Reply with only the number."
    else:
        # The index appears only in the suffix, so repeated requests share a
        # long token prefix even when their tails differ. No random timestamps.
        prompt = BACKGROUND * args.repeats
        prompt += f"\nTask {args.index}: What is 13 plus 29? Reply with only the number."
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 1024,
        "max_tokens": args.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def capture(args):
    errors = []
    timing = {}
    response = {}
    for filename, target in (("curl.json", timing), ("response.body", response)):
        try:
            value = json.loads((args.directory / filename).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            target.update(value)
        except (OSError, ValueError) as exc:
            errors.append(f"{filename}: {exc}")
    http_code = str(timing.get("http_code", "000"))
    if args.curl_exit:
        errors.append(f"curl exited with status {args.curl_exit}; see curl.stderr")
    if not http_code.isdecimal() or not 200 <= int(http_code) < 300:
        errors.append(f"HTTP {http_code}")
    if response.get("error"):
        errors.append("API error: " + json.dumps(response["error"], ensure_ascii=False))
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    if not isinstance(message, dict):
        errors.append("Response has no choices[0].message")
        message = {}
    if not message.get("content") and not message.get("reasoning_content"):
        errors.append("Response contains no answer or reasoning text")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        errors.append("Response has no usage object")
        usage = {}
    prompt_tokens = usage.get("prompt_tokens")
    if args.kind != "short" and (not isinstance(prompt_tokens, int) or prompt_tokens <= WINDOW_TOKENS):
        errors.append("Long-input test needs usage.prompt_tokens > 128; increase LONG_REPEATS if necessary")
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    result = {
        "index": args.index,
        "kind": args.kind,
        "request_ok": not errors,
        "http_code": http_code,
        "curl_exit": args.curl_exit,
        "time_total_s": timing.get("time_total_s"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": cached,
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "errors": errors,
    }
    (args.directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    preview = str(result["content"] or result["reasoning_content"] or "")[:120]
    print(
        f"request={args.index} ok={result['request_ok']} HTTP={http_code} "
        f"prompt={prompt_tokens} completion={result['completion_tokens']} "
        f"cached={cached} total={result['time_total_s']}s "
        f"finish={result['finish_reason']} text={json.dumps(preview, ensure_ascii=False)}",
        flush=True,
    )
    for error in errors:
        print("  " + error[:300], flush=True)
    return int(bool(errors))


def report(args):
    results = sorted(
        (json.loads(path.read_text()) for path in args.directory.glob("request-*/result.json")),
        key=lambda result: result["index"],
    )
    successful = sum(result["request_ok"] for result in results)
    summary = {
        "kind": args.kind,
        "base_url": args.base_url,
        "model": args.model,
        "expected_requests": args.expected,
        "recorded_requests": len(results),
        "successful_requests": successful,
        "failed_or_missing_requests": args.expected - successful,
        "client_run_elapsed_s": round((time.time_ns() - args.started_ns) / 1e9, 6),
        "timing_scope": "Non-streaming curl time_total; includes all output generation, not TTFT",
        "requests": results,
    }
    (args.directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    columns = (
        "index",
        "kind",
        "request_ok",
        "http_code",
        "curl_exit",
        "time_total_s",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "finish_reason",
        "errors",
    )
    with (args.directory / "summary.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({**result, "errors": "; ".join(result["errors"])})
    print(f"Completed: {successful}/{args.expected} successful. Results: {args.directory}", flush=True)
    return int(successful != args.expected or len(results) != args.expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--kind", choices=("short", "long", "prefix", "concurrent"), required=True)
    build_parser.add_argument("--index", type=int, required=True)
    build_parser.add_argument("--model", required=True)
    build_parser.add_argument("--repeats", type=positive, required=True)
    build_parser.add_argument("--max-tokens", type=positive, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.set_defaults(run=build)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--directory", type=Path, required=True)
    capture_parser.add_argument("--index", type=int, required=True)
    capture_parser.add_argument("--kind", required=True)
    capture_parser.add_argument("--curl-exit", type=int, required=True)
    capture_parser.set_defaults(run=capture)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--directory", type=Path, required=True)
    report_parser.add_argument("--expected", type=positive, required=True)
    report_parser.add_argument("--kind", required=True)
    report_parser.add_argument("--started-ns", type=int, required=True)
    report_parser.add_argument("--base-url", required=True)
    report_parser.add_argument("--model", required=True)
    report_parser.set_defaults(run=report)
    args = parser.parse_args()
    raise SystemExit(args.run(args))


if __name__ == "__main__":
    main()
