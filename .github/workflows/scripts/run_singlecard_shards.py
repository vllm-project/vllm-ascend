# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run complete single-card test files on disjoint, container-visible NPUs."""

import argparse
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

DEVICE_COUNT_PREFIX = "SINGLECARD_DEVICE_COUNT="
TERMINATE_TIMEOUT_SECONDS = 10
OUTCOMES = ("passed", "skipped", "failed", "errors")


def discover_files(tests: Path) -> list[Path]:
    if not tests.is_dir():
        raise ValueError(f"Expected a single-card test directory: {tests}")
    files = sorted(set(tests.rglob("test_*.py")) | set(tests.rglob("*_test.py")))
    if not files:
        raise ValueError(f"No test files found in {tests}")
    return files


def select_devices(workers: int, device_count: int, visible: str | None) -> list[str]:
    # The runtime count is from inside the job container, never from host npu-smi.
    devices = visible.split(",") if visible is not None else [str(i) for i in range(device_count)]
    if any(not device.isdecimal() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError(f"Invalid ASCEND_RT_VISIBLE_DEVICES: {visible!r}")
    if len(devices) != device_count or not 1 <= workers <= device_count:
        raise ValueError(f"Requested {workers} workers, but runtime exposes {device_count} NPUs: {devices}")
    return devices[:workers]


def runtime_device_count() -> int:
    # Probe in a separate process so the controller never initializes a device.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import torch, torch_npu; print('{DEVICE_COUNT_PREFIX}' + str(torch.npu.device_count()))",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if probe.returncode:
        raise RuntimeError(f"NPU visibility probe failed:\n{probe.stdout}\n{probe.stderr}")
    return int(
        next(
            line.removeprefix(DEVICE_COUNT_PREFIX)
            for line in probe.stdout.splitlines()
            if line.startswith(DEVICE_COUNT_PREFIX)
        )
    )


def partition_files(files: list[Path], workers: int, timings: dict[str, float]) -> list[list[Path]]:
    if not 1 <= workers <= len(files):
        raise ValueError("Each worker must receive at least one file")
    # Use a previous summary from the same environment when available. On the
    # first run, source size is only a deterministic scheduling estimate.
    known_times = [float(timings[path.as_posix()]) for path in files if path.as_posix() in timings]
    default_weight = sum(known_times) / len(known_times) if known_times else None
    weights = {
        path: float(timings.get(path.as_posix(), default_weight if known_times else path.stat().st_size))
        for path in files
    }
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Scheduling weights must be nonnegative finite numbers")
    shards: list[list[Path]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for path in sorted(files, key=lambda path: (-weights[path], path.as_posix())):
        rank = min(range(workers), key=lambda rank: (loads[rank], len(shards[rank]), rank))
        shards[rank].append(path)
        loads[rank] += weights[path]
    return shards


def read_report(report: Path) -> dict:
    counts = dict.fromkeys(OUTCOMES, 0)
    file_durations: dict[str, float] = {}
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
        file_durations[filename] = file_durations.get(filename, 0.0) + float(case.get("time", "0"))
    return {"counts": counts, "file_durations": file_durations, "cumulative_case_seconds": sum(file_durations.values())}


def stream_log(process: subprocess.Popen, log: Path, rank: int) -> None:
    with log.open("w", encoding="utf-8") as output:
        for line in process.stdout:
            output.write(line)
            output.flush()
            print(f"[worker {rank}] {line}", end="", flush=True)


def stop_workers(processes: list[subprocess.Popen]) -> None:
    # Only signal process groups created by this invocation, including their
    # pytest children; never kill other jobs or processes by name.
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
    # A pytest process may have exited while a descendant still holds its log
    # pipe open. Clean up that private group too, then reap all direct children.
    for process in processes:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_shards(shards: list[list[Path]], devices: list[str], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = [
        {"worker": rank, "device": device, "files": [path.as_posix() for path in paths]}
        for rank, (device, paths) in enumerate(zip(devices, shards, strict=True))
    ]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    processes: list[subprocess.Popen] = []
    readers: list[threading.Thread] = []
    results = []
    started = time.monotonic()
    try:
        for item in manifest:
            rank = item["worker"]
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = item["device"]
            cache_root = Path(env.get("TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache")))
            env["TRITON_CACHE_DIR"] = str(cache_root / "singlecard-shards" / f"worker-{rank}")
            report = output_dir / f"worker-{rank}.xml"
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
                f"cache_dir={output_dir / f'pytest-cache-{rank}'}",
                f"--junitxml={report}",
                *item["files"],
            ]
            print(f"Starting worker {rank}: visible NPU {item['device']}, {len(item['files'])} files", flush=True)
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=os.name == "posix",
            )
            processes.append(process)
            reader = threading.Thread(
                target=stream_log, args=(process, output_dir / f"worker-{rank}.log", rank), daemon=True
            )
            readers.append(reader)
            reader.start()
        # Wait for every shard, even if another shard fails, to retain coverage.
        for rank, process in enumerate(processes):
            returncode = process.wait()
            readers[rank].join(timeout=TERMINATE_TIMEOUT_SECONDS)
            if readers[rank].is_alive():
                stop_workers([process])
                readers[rank].join(timeout=TERMINATE_TIMEOUT_SECONDS)
                if readers[rank].is_alive():
                    raise RuntimeError(f"Worker {rank} log stream did not close")
            try:
                result = read_report(output_dir / f"worker-{rank}.xml")
            except (OSError, ET.ParseError, ValueError) as error:
                result = {
                    "report_error": str(error),
                    "counts": dict.fromkeys(OUTCOMES, 0),
                    "file_durations": {},
                    "cumulative_case_seconds": 0.0,
                }
            results.append({**manifest[rank], **result, "returncode": returncode})
    finally:
        stop_workers(processes)
        for reader in readers:
            reader.join(timeout=TERMINATE_TIMEOUT_SECONDS)
    file_durations: dict[str, float] = {}
    for result in results:
        for filename, seconds in result["file_durations"].items():
            file_durations[filename] = file_durations.get(filename, 0.0) + seconds
    summary = {
        "workers": results,
        "wall_seconds": time.monotonic() - started,
        "cumulative_case_seconds": sum(result["cumulative_case_seconds"] for result in results),
        "counts": {key: sum(result["counts"][key] for result in results) for key in OUTCOMES},
        "file_durations": file_durations,
    }
    failed = any(
        result["returncode"] != 0
        or "report_error" in result
        or result["counts"]["failed"]
        or result["counts"]["errors"]
        for result in results
    )
    summary["success"] = not failed
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    counts = ", ".join(f"{count} {name}" for name, count in summary["counts"].items())
    print(
        f"Singlecard total: {counts}; cumulative cases {summary['cumulative_case_seconds']:.3f}s; "
        f"wall {summary['wall_seconds']:.3f}s",
        flush=True,
    )
    # Existing AOP captures the last pytest-shaped summary; do not expose just
    # the last shard's counts as the result of the whole directory.
    print(
        f"================ {counts}{', shard execution error' if failed else ''} "
        f"in {summary['wall_seconds']:.3f}s ================",
        flush=True,
    )
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timings", type=Path, help="Previous summary.json from the same hardware and environment")
    args = parser.parse_args()
    files = discover_files(args.tests)
    timings = json.loads(args.timings.read_text(encoding="utf-8"))["file_durations"] if args.timings else {}
    shards = partition_files(files, args.workers, timings)
    devices = select_devices(args.workers, runtime_device_count(), os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    return run_shards(shards, devices, args.output_dir.resolve())


def interrupt(_signum: int, _frame) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
