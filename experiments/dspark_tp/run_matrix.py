#!/usr/bin/env python3
"""Run a resumable TP matrix using run_case.sh."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--ascend-source", required=True)
    parser.add_argument("--vllm-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--tps", nargs="+", type=int, default=(1, 2, 4, 8, 16))
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-prompts", type=int, default=16)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def passed(metadata_path: Path, args: argparse.Namespace) -> bool:
    if args.rerun or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("status") == "passed"
        and metadata.get("num_prompts") == args.num_prompts
        and metadata.get("runs") == args.runs
    )


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = Path(args.output_root) / args.label
    failures: list[dict] = []

    for tp in args.tps:
        for case in args.cases:
            metadata_path = root / f"tp{tp}" / case / "metadata.json"
            if passed(metadata_path, args):
                print(f"SKIP label={args.label} tp={tp} case={case}: already passed", flush=True)
                continue
            print(f"START label={args.label} tp={tp} case={case} at={time.strftime('%FT%T%z')}", flush=True)
            command = [
                "bash",
                str(script_dir / "run_case.sh"),
                "--case",
                case,
                "--tp",
                str(tp),
                "--ascend-source",
                args.ascend_source,
                "--vllm-source",
                args.vllm_source,
                "--output-root",
                str(root),
                "--num-prompts",
                str(args.num_prompts),
                "--runs",
                str(args.runs),
                "--warmup-prompts",
                str(args.warmup_prompts),
                "--startup-timeout",
                str(args.startup_timeout),
            ]
            result = subprocess.run(command, check=False)
            if result.returncode:
                failure = {"label": args.label, "tp": tp, "case": case, "returncode": result.returncode}
                failures.append(failure)
                print(f"FAIL {failure}", flush=True)
                break
            print(f"PASS label={args.label} tp={tp} case={case} at={time.strftime('%FT%T%z')}", flush=True)
        if failures:
            break

    failure_path = root / "matrix-failures.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
