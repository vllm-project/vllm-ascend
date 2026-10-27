# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run complete single-card test files on disjoint, container-visible NPUs."""

import argparse
import contextlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import regex as re

RUNTIME_PREFIX = "SINGLECARD_RUNTIME="
TERMINATE_TIMEOUT_SECONDS = 10
PROBE_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 0.1
OUTCOMES = ("passed", "skipped", "failed", "errors")


@dataclass
class WorkerRun:
    rank: int
    device: str
    files: list[Path]
    process: subprocess.Popen | None = None
    reader: threading.Thread | None = None
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    stopped: bool = False
    log_error: str | None = None


def discover_files(tests: Path) -> list[Path]:
    if not tests.is_dir():
        raise ValueError(f"Expected a single-card test directory: {tests}")
    files = sorted(set(tests.rglob("test_*.py")) | set(tests.rglob("*_test.py")))
    if not files:
        raise ValueError(f"No test files found in {tests}")
    return files


def runner_device_count(runner: str) -> int:
    labels = json.loads(Path(__file__).with_name("runner_label.json").read_text(encoding="utf-8"))
    if runner in labels:
        return int(labels[runner]["npu_num"])
    # The PR runner registry does not include every nightly label (e.g. A5-8).
    # Only recognize the existing A3/A5 NPU convention, never an arbitrary suffix.
    match = re.fullmatch(r"linux-(?:aarch64|amd64)-(?:a3(?:-800[it])?|a5)-([1-9][0-9]*)", runner)
    if match is None:
        raise ValueError(f"Unknown NPU runner label: {runner!r}")
    return int(match.group(1))


def resolve_workers(setting: str, runner: str, device_count: int, file_count: int) -> int:
    capacity = runner_device_count(runner) if runner else None
    if device_count < 1 or file_count < 1 or capacity == 0:
        raise ValueError("Single-card sharding requires NPU resources and at least one test file")
    if setting == "auto":
        if capacity is None:
            raise ValueError("--workers auto requires --runner")
        if capacity != device_count:
            raise ValueError(f"Runner expects {capacity} NPUs, but the container exposes {device_count}")
        requested = capacity
    elif re.fullmatch(r"[1-9][0-9]*", setting):
        requested = int(setting)
    else:
        raise ValueError("--workers must be 'auto' or a positive integer")
    if requested > device_count or (capacity is not None and requested > capacity):
        raise ValueError(f"Requested {requested} workers, runner capacity={capacity}, visible NPUs={device_count}")
    return min(requested, file_count)


def select_devices(workers: int, device_count: int, visible: str | None) -> list[str]:
    # The runtime count is from inside the job container, never from host npu-smi.
    devices = visible.split(",") if visible is not None else [str(i) for i in range(device_count)]
    if any(not device.isdecimal() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError(f"Invalid ASCEND_RT_VISIBLE_DEVICES: {visible!r}")
    if len(devices) != device_count or not 1 <= workers <= device_count:
        raise ValueError(f"Requested {workers} workers, but runtime exposes {device_count} NPUs: {devices}")
    return devices[:workers]


def runtime_environment() -> dict:
    # Probe imports in a short-lived child; the controller never initializes NPU.
    code = """
import importlib.metadata
import json
import platform
import torch
import torch_npu
from vllm.triton_utils import triton

count = torch.npu.device_count()
packages = {}
for name in ("torch", "torch-npu", "triton-ascend", "vllm", "vllm-ascend"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
packages.update({"torch": torch.__version__, "torch-npu": torch_npu.__version__, "triton": triton.__version__})
print("SINGLECARD_RUNTIME=" + json.dumps({
    "device_count": count,
    "device_name": torch.npu.get_device_name(0) if count else None,
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "packages": packages,
    "triton_path": triton.__file__,
}))
"""
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("NPU environment probe timed out") from error
    if probe.returncode:
        raise RuntimeError(f"NPU environment probe failed:\n{probe.stdout}\n{probe.stderr}")
    for line in reversed(probe.stdout.splitlines()):
        if line.startswith(RUNTIME_PREFIX):
            try:
                result = json.loads(line.removeprefix(RUNTIME_PREFIX))
                if not isinstance(result, dict) or type(result.get("device_count")) is not int:
                    raise ValueError("device_count must be an integer")
                return result
            except (ValueError, TypeError) as error:
                raise RuntimeError(f"Invalid NPU environment probe result: {line}") from error
    raise RuntimeError(f"NPU environment probe did not report device_count:\n{probe.stdout}")


def partition_files(files: list[Path], workers: int, timings: dict[str, float]) -> list[list[Path]]:
    if not 1 <= workers <= len(files):
        raise ValueError("Each worker must receive at least one file")
    known_times = [float(timings[path.as_posix()]) for path in files if path.as_posix() in timings]
    # Like select_tests.py, use equal weights for unknown files. When history is
    # available, use its mean as the default instead of mixing seconds and bytes.
    default_weight = sum(known_times) / len(known_times) if known_times else 1.0
    weights = {path: float(timings.get(path.as_posix(), default_weight)) for path in files}
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Scheduling weights must be nonnegative finite numbers")
    shards: list[list[Path]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for path in sorted(files, key=lambda path: (-weights[path], path.as_posix())):
        rank = min(range(workers), key=lambda rank: (loads[rank], len(shards[rank]), rank))
        shards[rank].append(path)
        loads[rank] += weights[path]
    return shards


def build_metadata(runtime: dict, args: argparse.Namespace, files: list[Path], devices: list[str]) -> dict:
    def git_output(*arguments: str) -> str | None:
        try:
            result = subprocess.run(["git", *arguments], capture_output=True, text=True, check=True, timeout=10)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    environment = {
        "runner": args.runner,
        "soc_version": args.soc_version,
        "image": args.image,
        "device_name": runtime.get("device_name"),
        "architecture": runtime.get("architecture", platform.machine()),
        "python": runtime.get("python", platform.python_version()),
        "packages": runtime.get("packages", {}),
        "workers": len(devices),
        "cache": {
            "triton_root": os.environ.get("TRITON_CACHE_DIR"),
            "per_worker_namespaces": False,
            "TRITON_HOME": os.environ.get("TRITON_HOME"),
            "TRITON_ALWAYS_COMPILE": os.environ.get("TRITON_ALWAYS_COMPILE"),
            "TRITON_DISABLE_FFTS": os.environ.get("TRITON_DISABLE_FFTS"),
        },
    }
    dirty = git_output("status", "--porcelain")
    return {
        "commit_sha": git_output("rev-parse", "HEAD"),
        "source_dirty": dirty != "" if dirty is not None else None,
        "workflow_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requested_workers": args.workers,
        "visible_device_count": runtime["device_count"],
        "devices": devices,
        "test_files": [path.as_posix() for path in files],
        "environment": environment,
        "triton_path": runtime.get("triton_path"),
        "timings_source": str(args.timings) if args.timings else None,
    }


def load_timings(path: Path, metadata: dict) -> dict[str, float]:
    previous = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(previous, dict) or previous.get("success") is not True:
        raise ValueError("Timing history must come from a successful, complete run")
    previous_metadata = previous.get("metadata")
    if not isinstance(previous_metadata, dict):
        raise ValueError("Timing history must include metadata")
    for key in ("environment", "test_files"):
        if not metadata.get(key) or previous_metadata.get(key) != metadata[key]:
            raise ValueError(f"Timing history {key} does not match this run")
    durations = previous.get("file_durations")
    if not isinstance(durations, dict):
        raise ValueError("Timing history must contain file_durations")
    values = {name: float(value) for name, value in durations.items() if name in metadata["test_files"]}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Timing history must contain nonnegative finite durations")
    return values


def read_report(report: Path) -> dict:
    counts = dict.fromkeys(OUTCOMES, 0)
    file_durations: dict[str, float] = {}
    failed_cases = []
    for case in ET.parse(report).getroot().iter("testcase"):
        if case.find("error") is not None:
            outcome = "errors"
        elif case.find("failure") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        counts[outcome] += 1
        filename = case.get("file", "<collection>")
        seconds = float(case.get("time", "0"))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("JUnit durations must be nonnegative finite numbers")
        file_durations[filename] = file_durations.get(filename, 0.0) + seconds
        if outcome in ("failed", "errors"):
            failed_cases.append(f"{filename}::{case.get('name', case.get('classname', '<unknown>'))}")
    return {
        "counts": counts,
        "file_durations": file_durations,
        "cumulative_case_seconds": sum(file_durations.values()),
        "failed_cases": failed_cases,
    }


def stream_log(worker: WorkerRun, log: Path) -> None:
    try:
        with log.open("w", encoding="utf-8") as output:
            for line in worker.process.stdout:
                output.write(line)
                output.flush()
                print(f"[worker {worker.rank}] {line}", end="", flush=True)
    except (OSError, ValueError) as error:
        worker.log_error = str(error)


def start_worker(worker: WorkerRun, output_dir: Path) -> None:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = worker.device
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-sv",
        "-o",
        "junit_family=xunit1",
        "-o",
        "junit_duration_report=total",
        "-o",
        f"cache_dir={output_dir / f'pytest-cache-{worker.rank}'}",
        f"--junitxml={output_dir / f'worker-{worker.rank}.xml'}",
        *[path.as_posix() for path in worker.files],
    ]
    print(f"Starting worker {worker.rank}: visible NPU {worker.device}, {len(worker.files)} files", flush=True)
    worker.started = time.monotonic()
    worker.process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=os.name == "posix",
    )
    worker.reader = threading.Thread(
        target=stream_log,
        args=(worker, output_dir / f"worker-{worker.rank}.log"),
        daemon=True,
    )
    worker.reader.start()


def poll_workers(workers: list[WorkerRun]) -> bool:
    pending = False
    for worker in workers:
        if worker.process is None or worker.finished is not None:
            continue
        worker.returncode = worker.process.poll()
        if worker.returncode is None:
            pending = True
        else:
            worker.finished = time.monotonic()
    return pending


def wait_workers(workers: list[WorkerRun]) -> None:
    # Observe all workers so their elapsed times are not inflated by a slower
    # earlier worker. Keep running other shards even when one shard fails.
    while poll_workers(workers):
        if any(worker.log_error for worker in workers):
            raise RuntimeError("A worker log stream failed")
        time.sleep(POLL_INTERVAL_SECONDS)


def stop_workers(processes: list[subprocess.Popen]) -> None:
    # Only clean up private groups from this invocation, including descendants.
    for process in processes:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + TERMINATE_TIMEOUT_SECONDS
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            if os.name != "posix":
                process.kill()
    for process in processes:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def collect_result(worker: WorkerRun, output_dir: Path, interrupted: bool) -> dict:
    try:
        result = read_report(output_dir / f"worker-{worker.rank}.xml")
    except (OSError, ET.ParseError, ValueError) as error:
        result = {
            "report_error": str(error),
            "counts": dict.fromkeys(OUTCOMES, 0),
            "file_durations": {},
            "cumulative_case_seconds": 0.0,
            "failed_cases": [],
        }
    if worker.process is None:
        status = "not_started"
    elif worker.stopped:
        status = "interrupted" if interrupted else "aborted"
    elif (
        worker.returncode != 0
        or "report_error" in result
        or worker.log_error
        or result["counts"]["failed"]
        or result["counts"]["errors"]
    ):
        status = "failed"
    else:
        status = "passed"
    return {
        "worker": worker.rank,
        "device": worker.device,
        "files": [path.as_posix() for path in worker.files],
        **result,
        "returncode": worker.returncode,
        "status": status,
        "wall_seconds": worker.finished - worker.started if worker.finished is not None else None,
        "log_error": worker.log_error,
        "log": f"worker-{worker.rank}.log",
    }


def write_summary(
    results: list[dict],
    output_dir: Path,
    metadata: dict,
    started: float,
    interrupted: bool,
    controller_error: str | None,
) -> dict:
    file_durations: dict[str, float] = {}
    for result in results:
        for filename, seconds in result["file_durations"].items():
            file_durations[filename] = file_durations.get(filename, 0.0) + seconds
    summary = {
        "metadata": metadata,
        "workers": results,
        "wall_seconds": time.monotonic() - started,
        "cumulative_case_seconds": sum(result["cumulative_case_seconds"] for result in results),
        "counts": {key: sum(result["counts"][key] for result in results) for key in OUTCOMES},
        "file_durations": file_durations,
        "interrupted": interrupted,
        "controller_error": controller_error,
        "partial": any(
            result["status"] in ("not_started", "interrupted", "aborted")
            or result["returncode"] not in (0, 1)
            or "report_error" in result
            for result in results
        ),
        "success": not interrupted and controller_error is None and all(r["status"] == "passed" for r in results),
    }
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "summary.json")
    return summary


def print_summary(summary: dict, output_dir: Path) -> None:
    metadata = summary["metadata"]
    environment = metadata.get("environment", {})
    rows = [
        "## Single-card operator tests",
        "",
        f"Source commit: `{metadata.get('commit_sha', 'unknown')}`; runner: `{environment.get('runner', 'unknown')}`.",
        f"Test phase: **{summary['wall_seconds']:.3f}s**; "
        f"cumulative cases: **{summary['cumulative_case_seconds']:.3f}s**.",
        f"Partial results: **{summary['partial']}**; interrupted: **{summary['interrupted']}**.",
        "",
        "| Worker | Device | Status | Pass | Skip | Fail | Error | Wall seconds |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in summary["workers"]:
        counts = result["counts"]
        seconds = f"{result['wall_seconds']:.3f}" if result["wall_seconds"] is not None else "-"
        rows.append(
            f"| {result['worker']} | {result['device']} | {result['status']} | {counts['passed']} | "
            f"{counts['skipped']} | {counts['failed']} | {counts['errors']} | {seconds} |"
        )
        if result["status"] != "passed":
            print(f"Worker {result['worker']}: {result['status']}; log: {output_dir / result['log']}", flush=True)
    failures = [case for result in summary["workers"] for case in result["failed_cases"]]
    if failures:
        rows.extend(["", "Failed cases:", *[f"- {case}" for case in failures]])
    if summary["controller_error"]:
        rows.extend(["", f"Controller error: {summary['controller_error']}"])
    markdown = "\n".join(rows) + "\n"
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if destination := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(destination).open("a", encoding="utf-8") as output:
            output.write(markdown)
    print(markdown, flush=True)
    counts = ", ".join(f"{count} {name}" for name, count in summary["counts"].items())
    # Keep the last pytest-shaped line compatible with existing log consumers;
    # the workflow outcome/exit code, not these counts, determines success.
    print(
        f"================ {counts}{', shard execution error' if not summary['success'] else ''} "
        f"in {summary['wall_seconds']:.3f}s ================",
        flush=True,
    )


def run_shards(shards: list[list[Path]], devices: list[str], output_dir: Path, metadata: dict | None = None) -> int:
    workers = [WorkerRun(rank, device, paths) for rank, (device, paths) in enumerate(zip(devices, shards, strict=True))]
    if not workers or any(not worker.files for worker in workers):
        raise ValueError("Each worker must receive at least one file")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = [{"worker": w.rank, "device": w.device, "files": [p.as_posix() for p in w.files]} for w in workers]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.monotonic()
    interrupted = False
    controller_error = None
    try:
        for worker in workers:
            start_worker(worker, output_dir)
        wait_workers(workers)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        controller_error = f"{type(error).__name__}: {error}"
    finally:
        poll_workers(workers)
        for worker in workers:
            worker.stopped = worker.process is not None and worker.finished is None
        try:
            if interrupted or controller_error is not None:
                # Persist results before cleanup in case CI escalates to SIGKILL.
                results = [collect_result(worker, output_dir, interrupted) for worker in workers]
                write_summary(results, output_dir, metadata or {}, started, interrupted, controller_error)
        except OSError as error:
            controller_error = "; ".join(filter(None, [controller_error, f"Cannot save partial summary: {error}"]))
            print(controller_error, file=sys.stderr, flush=True)
        finally:
            # A full disk or unavailable artifact directory must not leak workers.
            stop_workers([worker.process for worker in workers if worker.process is not None])
        poll_workers(workers)
        for worker in workers:
            if worker.reader is not None:
                worker.reader.join(timeout=TERMINATE_TIMEOUT_SECONDS)
                if worker.reader.is_alive():
                    worker.log_error = "Worker log stream did not close"
        # Read every report after cleanup, including completed shards that were
        # not yet observed when the controller received SIGTERM or Ctrl-C.
        results = [collect_result(worker, output_dir, interrupted) for worker in workers]
        summary = write_summary(results, output_dir, metadata or {}, started, interrupted, controller_error)
        print_summary(summary, output_dir)
    return 130 if interrupted else int(not summary["success"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", required=True, type=Path)
    parser.add_argument("--workers", default="auto", help="Worker count; defaults to the runner allocation")
    parser.add_argument("--runner", default=os.environ.get("VLLM_CI_RUNNER", ""))
    parser.add_argument("--image", default="", help="Image reference used by the workflow")
    parser.add_argument("--soc-version", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timings", type=Path, help="Successful summary.json from the same environment and test scope")
    args = parser.parse_args()
    files = discover_files(args.tests)
    runtime = runtime_environment()
    workers = resolve_workers(args.workers, args.runner, runtime["device_count"], len(files))
    devices = select_devices(workers, runtime["device_count"], os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    metadata = build_metadata(runtime, args, files, devices)
    timings = load_timings(args.timings, metadata) if args.timings else {}
    shards = partition_files(files, workers, timings)
    return run_shards(shards, devices, args.output_dir.resolve(), metadata)


def interrupt(_signum: int, _frame) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
