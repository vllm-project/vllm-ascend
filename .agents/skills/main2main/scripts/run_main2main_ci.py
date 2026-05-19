#!/usr/bin/env python3
"""Run main2main CI, summarize failures, and classify the step result.

This wrapper replaces shell pipelines like:

    python3 run_suite.py ... 2>&1 | tee round-1.log

Shell pipelines are easy to get wrong because tee usually returns 0 even when
run_suite.py fails. This script writes run_suite.py output only to the step log,
runs ci_log_summary.py on that log, and records a machine-readable result JSON.
It intentionally does not stream raw CI logs to stdout, so workflow logs and
agent context stay small.

The raw run_suite.py status is always stored as run_suite_exit_code. The wrapper
process exits 0 only when main2main may proceed: passed CI or env_flake_pass.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PASS_RESULTS = {"passed", "env_flake_pass"}
DEFAULT_SUITES = ["e2e-singlecard-light"]


def _run_to_log(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        return process.wait()


def _run_summary(
    ci_log_summary: Path,
    log_path: Path,
    summary_path: Path,
    step_id: str,
    round_number: int,
) -> dict:
    if not ci_log_summary.exists():
        return {
            "summary_exit_code": 1,
            "summary_error": f"ci_log_summary.py not found: {ci_log_summary}",
            "summary": None,
        }

    command = [
        sys.executable,
        str(ci_log_summary),
        "--log-file",
        str(log_path),
        "--format",
        "llm-json",
        "--output",
        str(summary_path),
        "--step-name",
        f"main2main {step_id} round {round_number}",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return {
            "summary_exit_code": result.returncode,
            "summary_error": result.stderr.strip() or result.stdout.strip(),
            "summary": None,
        }
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return {
            "summary_exit_code": result.returncode,
            "summary_error": f"summary output was not written: {summary_path}",
            "summary": None,
        }
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "summary_exit_code": result.returncode,
            "summary_error": f"invalid summary JSON: {exc}",
            "summary": None,
        }
    return {
        "summary_exit_code": result.returncode,
        "summary_error": None,
        "summary": summary,
    }


def _count(summary: dict | None, field: str) -> int:
    if not summary:
        return 0
    count_field = f"{field}_count"
    if count_field in summary:
        return int(summary[count_field])
    value = summary.get(field, [])
    return len(value) if isinstance(value, list) else 0


def _classify_result(run_suite_exit_code: int, summary: dict | None, summary_error: str | None) -> str:
    if run_suite_exit_code == 0:
        return "passed"
    if summary_error or summary is None:
        return "summary_error"

    code_bugs_count = len(summary.get("code_bugs", []))
    env_flakes_count = len(summary.get("env_flakes", []))
    if code_bugs_count == 0 and env_flakes_count > 0:
        return "env_flake_pass"
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run e2e-main2main CI for one main2main step.",
    )
    parser.add_argument("--ascend-path", type=Path, required=True,
                        help="Path to the vllm-ascend repository")
    parser.add_argument("--step-id", required=True,
                        help="Step identifier, for example step-1")
    parser.add_argument("--round", type=int, default=1,
                        help="CI round number for this step")
    parser.add_argument("--suite", action="append",
                        help="run_suite.py suite name. Can be specified multiple times. "
                             "Defaults to e2e-singlecard-light.")
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/main2main"),
                        help="main2main workspace directory")
    args = parser.parse_args()

    ascend_path = args.ascend_path.resolve()
    run_suite = ascend_path / ".github" / "workflows" / "scripts" / "run_suite.py"
    ci_log_summary = ascend_path / ".github" / "workflows" / "scripts" / "ci_log_summary.py"
    if not run_suite.exists():
        print(f"Error: run_suite.py not found: {run_suite}", file=sys.stderr)
        sys.exit(1)

    ci_dir = args.workspace / "steps" / args.step_id / "ci"
    log_path = ci_dir / f"round-{args.round}.log"
    summary_path = ci_dir / f"round-{args.round}-summary.json"
    result_path = ci_dir / f"round-{args.round}-result.json"

    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("VLLM_USE_MODELSCOPE", "true")

    suite_names = args.suite or DEFAULT_SUITES
    suite_label = suite_names[0] if len(suite_names) == 1 else "+".join(suite_names)
    command = [
        sys.executable,
        str(run_suite),
    ]
    for suite_name in suite_names:
        command.extend(["--suite", suite_name])
    command.append("--continue-on-error")
    print(f"main2main CI raw log: {log_path}", flush=True)
    run_suite_exit_code = _run_to_log(command, ascend_path, log_path, env)
    print(
        f"main2main run_suite exited with {run_suite_exit_code}; summarizing log",
        flush=True,
    )

    summary_result = _run_summary(
        ci_log_summary=ci_log_summary,
        log_path=log_path,
        summary_path=summary_path,
        step_id=args.step_id,
        round_number=args.round,
    )
    summary = summary_result["summary"]
    summary_error = summary_result["summary_error"]
    ci_result = _classify_result(run_suite_exit_code, summary, summary_error)
    can_commit = ci_result in PASS_RESULTS

    result = {
        "step_id": args.step_id,
        "round": args.round,
        "suite": suite_label,
        "suites": suite_names,
        "run_suite_exit_code": run_suite_exit_code,
        "exit_code": run_suite_exit_code,
        "summary_exit_code": summary_result["summary_exit_code"],
        "ci_result": ci_result,
        "passed": ci_result == "passed",
        "can_commit": can_commit,
        "requires_fix": ci_result == "failed" and len((summary or {}).get("code_bugs", [])) > 0,
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "summary_error": summary_error,
        "code_bugs_count": len((summary or {}).get("code_bugs", [])),
        "env_flakes_count": len((summary or {}).get("env_flakes", [])),
        "failed_test_files_count": _count(summary, "failed_test_files"),
        "failed_test_cases_count": _count(summary, "failed_test_cases"),
        "command": command,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"main2main CI summary written to {summary_path}", flush=True)
    print(f"main2main CI result written to {result_path}", flush=True)
    sys.exit(0 if can_commit else run_suite_exit_code or 1)


if __name__ == "__main__":
    main()
