"""Collect a bounded worker profiling window from one AISBench benchmark."""

import csv
import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import regex as re
import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300
CONNECT_TIMEOUT_SECONDS = 10
MANIFEST_FILENAME = "manifest.json"
KERNEL_TABLE_FILENAMES = ("kernel_details.csv", "task_time.csv")
WORKER_RANK_PATTERN = re.compile(r"(?:^|[_-])rank[_-]?(\d+)(?:[_-]|$)")


@dataclass(frozen=True)
class BenchmarkProfilingConfig:
    benchmark_key: str
    output_dir: Path
    delay_iterations: int
    max_iterations: int
    expected_workers: int = 1
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def _integer(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"profiling.{name} must be an integer >= {minimum}")
    return value


def _profiler_config_from_command(server_cmd: list[str]) -> dict[str, Any]:
    values = []
    for index, argument in enumerate(server_cmd):
        if argument == "--profiler-config":
            if index + 1 >= len(server_cmd):
                raise ValueError("--profiler-config requires a JSON value")
            values.append(server_cmd[index + 1])
        elif argument.startswith("--profiler-config="):
            values.append(argument.split("=", 1)[1])
    if len(values) != 1:
        raise ValueError("profiling requires exactly one --profiler-config JSON value")
    try:
        result = json.loads(values[0])
    except (TypeError, ValueError) as error:
        raise ValueError("--profiler-config must contain valid JSON") from error
    if not isinstance(result, dict):
        raise ValueError("--profiler-config must contain a JSON object")
    return result


def parse_benchmark_profiling_config(
    value: Any,
    *,
    benchmarks: Mapping[str, Any],
    server_cmd: list[str],
    service_mode: str,
) -> BenchmarkProfilingConfig | None:
    """Validate opt-in YAML configuration before launching the server."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("profiling must be a mapping")
    fields = {
        "benchmark_key",
        "output_dir",
        "delay_iterations",
        "max_iterations",
        "expected_workers",
        "timeout_seconds",
    }
    unknown = set(value) - fields
    if unknown:
        raise ValueError(f"Unknown profiling fields: {sorted(unknown)}")
    if service_mode != "openai":
        raise ValueError("benchmark profiling requires the OpenAI service mode")
    benchmark_key = value.get("benchmark_key")
    if not isinstance(benchmark_key, str) or not benchmark_key.strip():
        raise ValueError("profiling.benchmark_key must be a non-empty string")
    selected = benchmarks.get(benchmark_key)
    if not isinstance(selected, dict) or selected.get("case_type") != "performance":
        raise ValueError("profiling.benchmark_key must select an enabled performance benchmark")
    if "warm" in benchmark_key.lower():
        raise ValueError("profiling.benchmark_key must select the formal benchmark, excluding warmup")
    output_dir_value = value.get("output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value.strip():
        raise ValueError("profiling.output_dir must be a non-empty path")
    output_dir = Path(output_dir_value).resolve()
    profiling_root = Path("profiling").resolve()
    if output_dir == profiling_root or not output_dir.is_relative_to(profiling_root):
        raise ValueError("profiling.output_dir must be a case-specific directory under profiling/")
    delay_iterations = _integer(value.get("delay_iterations"), "delay_iterations", 0)
    max_iterations = _integer(value.get("max_iterations"), "max_iterations", 1)
    expected_workers = _integer(value.get("expected_workers", 1), "expected_workers", 1)
    timeout_seconds = value.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("profiling.timeout_seconds must be a finite positive number")

    profiler = _profiler_config_from_command(server_cmd)
    required = {
        "profiler": "torch",
        "torch_profiler_with_stack": False,
        "torch_profiler_with_memory": False,
        "ignore_frontend": True,
        "delay_iterations": delay_iterations,
        "max_iterations": max_iterations,
    }
    for key, expected in required.items():
        actual = profiler.get(key)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"--profiler-config {key} must equal {expected!r}")
    profiler_dir = profiler.get("torch_profiler_dir")
    if not isinstance(profiler_dir, str) or not Path(profiler_dir).is_absolute():
        raise ValueError("--profiler-config torch_profiler_dir must be an absolute path")
    if Path(profiler_dir).resolve() != output_dir:
        raise ValueError("profiling.output_dir and --profiler-config torch_profiler_dir must match")
    return BenchmarkProfilingConfig(
        benchmark_key=benchmark_key,
        output_dir=output_dir,
        delay_iterations=delay_iterations,
        max_iterations=max_iterations,
        expected_workers=expected_workers,
        timeout_seconds=float(timeout_seconds),
    )


def _error_text(error: BaseException | None) -> str | None:
    return f"{type(error).__name__}: {error}" if error is not None else None


class BenchmarkProfiler:
    """Arm worker-step profiling, flush it, and check its exported kernel tables."""

    def __init__(self, config: BenchmarkProfilingConfig, port: int, benchmark: Mapping[str, Any] | None = None):
        self.config = config
        self.base_url = f"http://127.0.0.1:{port}"
        self.start_attempted = False
        self.manifest: dict[str, Any] = {
            "benchmark_key": config.benchmark_key,
            "benchmark": dict(benchmark) if benchmark is not None else None,
            "output_dir": str(config.output_dir),
            "delay_iterations": config.delay_iterations,
            "max_iterations": config.max_iterations,
            "expected_workers": config.expected_workers,
            "timeout_seconds": config.timeout_seconds,
            "start_profile": {"status": "not_attempted"},
            "stop_profile": {"status": "not_attempted"},
            "benchmark_error": None,
            "benchmark_status": "not_started",
            "traces": [],
            "trace_directory_count": 0,
            "valid_worker_trace_count": 0,
            "unique_worker_ranks": [],
        }

    def _post(self, endpoint: str) -> None:
        record = self.manifest[endpoint]
        record["requested_at"] = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        try:
            response = requests.post(
                f"{self.base_url}/{endpoint}",
                timeout=(min(CONNECT_TIMEOUT_SECONDS, self.config.timeout_seconds), self.config.timeout_seconds),
            )
            record["http_status"] = response.status_code
            response.raise_for_status()
            record["status"] = "success"
        except BaseException as error:
            record["status"] = "failed"
            record["error"] = _error_text(error)
            raise
        finally:
            record["duration_seconds"] = round(time.monotonic() - started, 3)

    def __enter__(self) -> "BenchmarkProfiler":
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.config.output_dir / MANIFEST_FILENAME).exists() or list(self.config.output_dir.glob("*_ascend_pt")):
            raise ValueError(f"Profiling output already exists in {self.config.output_dir}; use a fresh directory")
        self.start_attempted = True
        try:
            # Starting here only arms the worker counter. AISBench setup and idle
            # time cannot consume the delayed window because no model steps run.
            self._post("start_profile")
        except BaseException as error:
            self._finish(error, benchmark_started=False)
            raise
        self.manifest["benchmark_status"] = "running"
        return self

    def _validate_traces(self) -> None:
        trace_dirs = sorted(path for path in self.config.output_dir.glob("*_ascend_pt") if path.is_dir())
        for trace_dir in trace_dirs:
            rank_match = WORKER_RANK_PATTERN.search(trace_dir.name)
            worker_rank = int(rank_match.group(1)) if rank_match else None
            kernel_tables = []
            for filename in KERNEL_TABLE_FILENAMES:
                for table in trace_dir.rglob(filename):
                    if not table.is_file() or table.stat().st_size == 0:
                        continue
                    with table.open(encoding="utf-8-sig", newline="") as stream:
                        rows = csv.reader(stream)
                        next(rows, None)
                        if any(any(cell.strip() for cell in row) for row in rows):
                            kernel_tables.append(str(table.relative_to(self.config.output_dir)))
            self.manifest["traces"].append(
                {
                    "directory": trace_dir.name,
                    "worker_rank": worker_rank,
                    "kernel_tables": kernel_tables,
                    "valid": bool(kernel_tables) and worker_rank is not None,
                }
            )
        valid_ranks = sorted({trace["worker_rank"] for trace in self.manifest["traces"] if trace["valid"]})
        valid_count = len(valid_ranks)
        self.manifest["trace_directory_count"] = len(trace_dirs)
        self.manifest["valid_worker_trace_count"] = valid_count
        self.manifest["unique_worker_ranks"] = valid_ranks
        if valid_count != self.config.expected_workers or len(trace_dirs) != self.config.expected_workers:
            raise RuntimeError(
                f"Expected {self.config.expected_workers} worker traces with kernel data, "
                f"found {valid_count} valid traces in {len(trace_dirs)} directories"
            )

    def _finish(self, primary_error: BaseException | None, *, benchmark_started: bool = True) -> None:
        if benchmark_started:
            self.manifest["benchmark_error"] = _error_text(primary_error)
            self.manifest["benchmark_status"] = "failed" if primary_error else "success"
        errors = []
        if self.start_attempted:
            try:
                # A timed-out start may have activated some workers. Always stop
                # before leaving the server context, even when start failed.
                self._post("stop_profile")
            except BaseException as error:
                errors.append(error)
        if self.manifest["start_profile"]["status"] == "success":
            try:
                self._validate_traces()
            except BaseException as error:
                errors.append(error)
        self.manifest["profiling_errors"] = [_error_text(error) for error in errors]
        self.manifest["status"] = "failed" if primary_error or errors else "success"
        try:
            manifest_path = self.config.output_dir / MANIFEST_FILENAME
            manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
            logger.info("Profiling manifest saved to %s", manifest_path)
        except BaseException as error:
            errors.append(error)
        if primary_error is not None:
            for error in errors:
                logger.error("Profiling cleanup failed: %s", error)
                # Python 3.10 lacks add_note; the manifest and log still retain
                # cleanup failures while the original exception propagates.
                add_note = getattr(primary_error, "add_note", None)
                if add_note is not None:
                    add_note(f"Profiling cleanup failed: {_error_text(error)}")
        elif errors:
            message = "Benchmark profiling failed: " + "; ".join(str(error) for error in errors)
            raise RuntimeError(message) from errors[0]

    def __exit__(self, exc_type: Any, exc_value: BaseException | None, traceback: Any) -> None:
        self._finish(exc_value)


def run_profiled_benchmarks(
    *,
    model: str,
    port: int,
    benchmark_keys: list[str],
    benchmarks: Mapping[str, Any],
    profiling: BenchmarkProfilingConfig,
    run_cases: Callable[..., list[Any]],
) -> list[Any]:
    """Keep warmup outside the profile and preserve benchmark result ordering."""
    results = []
    for key in benchmark_keys:
        if key == profiling.benchmark_key:
            with BenchmarkProfiler(profiling, port, benchmarks[key]):
                results.extend(run_cases(model=model, port=port, aisbench_cases=[benchmarks[key]]))
        else:
            results.extend(run_cases(model=model, port=port, aisbench_cases=[benchmarks[key]]))
    return results
