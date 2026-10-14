#!/usr/bin/env python3
"""Run adjacent old/new control pairs without changing the measured workload."""

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
    parser.add_argument("--base-source", required=True)
    parser.add_argument("--current-source", required=True)
    parser.add_argument("--vllm-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tps", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument(
        "--cases", nargs="+", choices=("target", "adaptive"), default=("target", "adaptive")
    )
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-prompts", type=int, default=16)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--device-offset", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def commit(source: str) -> str:
    result = subprocess.run(
        ["git", "-C", source, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def already_passed(metadata_path: Path, args: argparse.Namespace, tp: int, case: str, source: str) -> bool:
    if args.rerun or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    devices = ",".join(str(device) for device in range(args.device_offset, args.device_offset + tp))
    return (
        metadata.get("status") == "passed"
        and metadata.get("case") == case
        and metadata.get("tp") == tp
        and metadata.get("ascend_commit") == commit(source)
        and metadata.get("visible_devices") == devices
        and metadata.get("num_prompts") == args.num_prompts
        and metadata.get("runs") == args.runs
    )


def main() -> int:
    args = parse_args()
    root = Path(args.output_root) / args.label
    script = Path(__file__).resolve().parent / "run_case.sh"
    sources = {"base": args.base_source, "current": args.current_source}
    records: list[dict] = []
    failure = False

    for tp_index, tp in enumerate(args.tps):
        for case_index, case in enumerate(args.cases):
            order = ("base", "current") if (tp_index + case_index) % 2 == 0 else ("current", "base")
            for source_name in order:
                source = sources[source_name]
                metadata_path = root / source_name / f"tp{tp}" / case / "metadata.json"
                if already_passed(metadata_path, args, tp, case, source):
                    print(f"SKIP tp={tp} case={case} source={source_name}: already passed", flush=True)
                    continue
                command = [
                    "bash",
                    str(script),
                    "--case",
                    case,
                    "--tp",
                    str(tp),
                    "--ascend-source",
                    source,
                    "--vllm-source",
                    args.vllm_source,
                    "--output-root",
                    str(root / source_name),
                    "--num-prompts",
                    str(args.num_prompts),
                    "--runs",
                    str(args.runs),
                    "--warmup-prompts",
                    str(args.warmup_prompts),
                    "--startup-timeout",
                    str(args.startup_timeout),
                    "--port",
                    str(args.port),
                    "--device-offset",
                    str(args.device_offset),
                ]
                record = {
                    "tp": tp,
                    "case": case,
                    "source": source_name,
                    "commit": commit(source),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                print(f"START {record}", flush=True)
                result = subprocess.run(command, check=False)
                record["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                record["returncode"] = result.returncode
                records.append(record)
                root.mkdir(parents=True, exist_ok=True)
                (root / "pair-sequence.json").write_text(
                    json.dumps(records, indent=2), encoding="utf-8"
                )
                if result.returncode:
                    print(f"FAIL {record}", flush=True)
                    failure = True
                    break
                print(f"PASS {record}", flush=True)
            if failure:
                break
        if failure:
            break

    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
