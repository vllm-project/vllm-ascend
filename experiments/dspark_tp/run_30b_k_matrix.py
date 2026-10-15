#!/usr/bin/env python3
"""Run TP/K DSpark comparisons serially on one 30B Ascend host."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def planned_cases(tps: list[int], widths: list[int]) -> list[tuple[int, str, int | None]]:
    cases = []
    for tp in tps:
        cases.append((tp, "target", None))
        for width in widths:
            cases.extend(((tp, "adaptive", width), (tp, "physical", width)))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ascend-source", required=True)
    parser.add_argument("--vllm-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", default="/home/vllm/weights/Qwen3-30B-A3B/")
    parser.add_argument("--draft-model", default="/home/vllm/weights/CMB-Qwen3-30B-A3B-DSpark-preview")
    parser.add_argument("--tp", type=int, nargs="+", choices=(2, 4), default=[4, 2])
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.93)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    return parser.parse_args()


def status_for(path: Path) -> str | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("status")


def main() -> int:
    args = parse_args()
    if not args.k or any(width < 1 or width > 8 for width in args.k):
        raise ValueError("K values must be in 1..8 for this checkpoint")
    if args.max_concurrency > args.max_num_seqs:
        raise ValueError("max-concurrency must not exceed max-num-seqs")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("run_case.py")
    cases = planned_cases(args.tp, args.k)
    skipped_tp: set[int] = set()
    summary = []
    for index, (tp, case, width) in enumerate(cases, start=1):
        tag = f"k{width}" if width is not None else None
        case_dir = root / f"tp{tp}" / (case if tag is None else f"{case}-{tag}")
        metadata = case_dir / "metadata.json"
        prior_status = status_for(metadata)
        if tp in skipped_tp:
            summary.append({"tp": tp, "case": case, "k": width, "status": "skipped_after_target_failure"})
            continue
        if prior_status == "passed":
            summary.append({"tp": tp, "case": case, "k": width, "status": "passed", "resumed": True})
            continue
        command = [
            sys.executable,
            str(script),
            "--case", case,
            "--tp", str(tp),
            "--ascend-source", args.ascend_source,
            "--vllm-source", args.vllm_source,
            "--output-root", str(root),
            "--model", args.model,
            "--draft-model", args.draft_model,
            "--max-model-len", "8192",
            "--max-num-seqs", str(args.max_num_seqs),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--input-len", "7168",
            "--output-len", "1024",
            "--max-concurrency", str(args.max_concurrency),
            "--num-prompts", str(args.num_prompts),
            "--runs", "1",
            "--warmup-prompts", "4",
            "--startup-timeout", str(args.startup_timeout),
        ]
        if width is not None:
            command.extend(("--num-speculative-tokens", str(width), "--result-tag", tag))
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(cases)}] START tp={tp} case={case} K={width}", flush=True)
        started = time.monotonic()
        with (case_dir / "driver.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed = time.monotonic() - started
        final_status = status_for(metadata) or "failed_before_metadata"
        if result.returncode != 0 and final_status == "passed":
            final_status = "failed_after_metadata"
        summary.append(
            {"tp": tp, "case": case, "k": width, "status": final_status,
             "exit_code": result.returncode, "elapsed_seconds": elapsed}
        )
        (root / "matrix-status.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"[{index}/{len(cases)}] END tp={tp} case={case} K={width} "
              f"status={final_status} elapsed={elapsed:.1f}s", flush=True)
        if case == "target" and final_status != "passed":
            skipped_tp.add(tp)
        # Let the previous service's worker processes release NPU memory.
        time.sleep(30)
    return 0 if all(item["status"] == "passed" for item in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
