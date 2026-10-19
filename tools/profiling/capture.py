"""Capture runtime profiles and publish raw artifacts from benchmark jobs."""

import argparse
import json
import os
import shlex
import shutil
import tarfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, cast

from tools.profiling import safe_name
from tools.profiling.storage import (
    DEFAULT_BUCKET,
    DEFAULT_ENDPOINT,
    DEFAULT_REGION,
    GZIP_COMPRESSION_LEVEL,
    ArtifactError,
    create_obs_client,
    create_transfer_config,
    upload_archive,
)

# =============================================================================
# Configuration and serving topology
# =============================================================================

ProfileOutput = Literal["parsed", "raw"]
DEFAULT_START_AFTER_SECONDS = 60


def _parse_bool(value: bool | str) -> bool:
    """Accept booleans or case-insensitive true/false strings, and nothing else."""
    if isinstance(value, bool):
        return value
    normalized = str(value).lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError("Expected true or false")


@dataclass(frozen=True)
class ProfileSpec:
    """User-selectable settings for one scheduled profiling session.

    ``start_after`` is measured from the first formal AISBench request, after
    warmup; ``start_after_requests`` instead counts successful formal requests.
    ``duration`` is the capture length in seconds. All serving instances
    are captured; ``benchmark_names`` optionally selects performance benchmark names
    from the test YAML, not case types or serving instance names.
    """

    enabled: bool = False
    start_after: int | None = DEFAULT_START_AFTER_SECONDS
    start_after_requests: int | None = None
    duration: int = 8
    with_stack: bool = False
    output: ProfileOutput = "parsed"
    benchmark_names: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "ProfileSpec":
        """Build and validate a specification from ``ASCEND_PROFILE_*`` values."""
        return cls.from_options(
            {
                field.name: os.environ[f"ASCEND_PROFILE_{field.name.upper()}"]
                for field in fields(cls)
                if f"ASCEND_PROFILE_{field.name.upper()}" in os.environ
            }
        )

    @classmethod
    def from_options(cls, options: dict) -> "ProfileSpec":
        """Normalize CI JSON and environment inputs using one set of defaults."""
        defaults = cls()
        if not isinstance(options, dict) or options.keys() - {field.name for field in fields(cls)}:
            raise ValueError("Unknown profiling options; expected a JSON object with ProfileSpec fields")
        benchmark_names = options.get("benchmark_names", ",".join(defaults.benchmark_names))
        if not isinstance(benchmark_names, str):
            raise ValueError("Profile benchmark_names must be a comma-separated string")
        if "\n" in benchmark_names or "\r" in benchmark_names:
            raise ValueError("Profile benchmark_names must be single-line")
        output = options.get("output", defaults.output)
        if output not in ("parsed", "raw"):
            raise ValueError("Invalid profile output mode")
        if "start_after" in options and "start_after_requests" in options:
            raise ValueError("Choose either start_after or start_after_requests")
        for name in ("start_after", "start_after_requests", "duration"):
            if name in options and (isinstance(options[name], bool) or not isinstance(options[name], (int, str))):
                raise ValueError("Profile times and request counts must be integers")
        spec = cls(
            enabled=_parse_bool(options.get("enabled", defaults.enabled)),
            start_after=None
            if "start_after_requests" in options
            else int(options.get("start_after", DEFAULT_START_AFTER_SECONDS)),
            start_after_requests=int(options["start_after_requests"]) if "start_after_requests" in options else None,
            duration=int(options.get("duration", defaults.duration)),
            with_stack=_parse_bool(options.get("with_stack", defaults.with_stack)),
            output=cast(ProfileOutput, output),
            benchmark_names=tuple(name.strip() for name in benchmark_names.split(",") if name.strip()),
        )
        if (spec.start_after is not None and spec.start_after < 0) or spec.duration <= 0:
            raise ValueError("Profile times must be positive (start-after may be zero)")
        if spec.start_after_requests is not None and spec.start_after_requests <= 0:
            raise ValueError("Profile start_after_requests must be positive")
        return spec

    def options(self) -> dict:
        """Return the canonical JSON options passed between CI workflows."""
        return {
            **{name: value for name, value in asdict(self).items() if value is not None},
            "benchmark_names": ",".join(self.benchmark_names),
        }

    def environment(self) -> dict[str, str]:
        """Expand options only at the test process/container boundary."""
        return {
            f"ASCEND_PROFILE_{name.upper()}": str(value).lower() if isinstance(value, bool) else str(value)
            for name, value in self.options().items()
        }

    def includes_benchmark(self, benchmark_name: str, case_type: str) -> bool:
        """Return whether a named performance benchmark should be profiled."""
        return (
            self.enabled
            and case_type == "performance"
            and (not self.benchmark_names or benchmark_name in self.benchmark_names)
        )


@dataclass(frozen=True)
class ServeInstance:
    """Address and artifact location for one independently profiled server."""

    name: str
    endpoint: str
    raw_trace_dir: str
    node_index: int = 0

    @classmethod
    def from_endpoint(
        cls,
        name: str,
        endpoint: str,
        *,
        node_index: int = 0,
        root: Path | None = None,
    ) -> "ServeInstance":
        """Create an instance with a filesystem-safe raw trace directory."""
        import regex as re

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        artifact_root = profile_root() if root is None else root.resolve()
        return cls(name, endpoint, str(artifact_root / "raw" / safe_name), node_index)


@dataclass(frozen=True)
class ServeManifest:
    """Serializable inventory of server instances started by a benchmark job."""

    instances: tuple[ServeInstance, ...]

    def write(self, path: Path) -> None:
        """Write the versioned server inventory to ``path`` as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "instances": [asdict(i) for i in self.instances]}, indent=2))

    @classmethod
    def read(cls, path: Path) -> "ServeManifest":
        """Load a server inventory previously written by :meth:`write`."""
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            raise ValueError(f"Unsupported serve manifest version: {data.get('version')!r}")
        return cls(tuple(ServeInstance(**item) for item in data["instances"]))


def profile_root() -> Path:
    """Return the absolute root used for manifests, traces, and archives."""
    return Path(os.getenv("ASCEND_PROFILE_ROOT", "profile_artifact")).resolve()


def initialize_profile_run(root: Path, spec: ProfileSpec) -> None:
    """Initialize one job in a fresh directory, never overwriting prior results."""
    root.mkdir(parents=True, exist_ok=False)
    manifest_path = root / "profile_manifest.json"
    manifest_path.write_text(
        json.dumps({"version": 2, "requested": asdict(spec), "benchmarks": [], "status": "skipped"})
    )


def prepare_capture(
    root: Path, benchmark_name: str, benchmark_artifact_name: str, targets: tuple[ServeInstance, ...]
) -> None:
    """Snapshot the serving topology before capture and retain interrupted attempts."""
    directory = root / "benchmarks" / safe_name(benchmark_artifact_name)
    directory.mkdir(parents=True, exist_ok=False)
    serve_path = directory / "serve_manifest.json"
    ServeManifest(targets).write(serve_path)
    manifest_path = root / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["benchmarks"].append(
        {
            "benchmark_name": benchmark_name,
            "benchmark_artifact_name": benchmark_artifact_name,
            "serve_manifest": str(serve_path.relative_to(root)),
            "status": "partial",
            "reason": "capture_not_completed",
            "targets": [{"name": target.name, "node_index": target.node_index} for target in targets],
        }
    )
    manifest["status"] = "partial"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def configure_profiler_command(command: list[str] | str, instance: ServeInstance, spec: ProfileSpec) -> list[str] | str:
    """Merge only profiler-owned options into the existing server command."""
    command_args = shlex.split(command) if isinstance(command, str) else list(command)
    profiler_flag = "--profiler-config"
    existing = json.loads(command_args[command_args.index(profiler_flag) + 1]) if profiler_flag in command_args else {}
    existing.update(
        profiler="torch",
        torch_profiler_dir=instance.raw_trace_dir,
        torch_profiler_with_stack=spec.with_stack,
        ignore_frontend=True,
        max_iterations=0,
    )
    if profiler_flag in command_args:
        command_args[command_args.index(profiler_flag) + 1] = json.dumps(existing)
    else:
        command_args.extend((profiler_flag, json.dumps(existing)))
    return shlex.join(command_args) if isinstance(command, str) else command_args


# =============================================================================
# Runtime profiling control
# =============================================================================

STOP_TIMEOUT_SECONDS = 900
POLL_INTERVAL_SECONDS = 1


def record_first_request(marker: Path) -> None:
    """Record the capture origin using AISBench's clock, once."""
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return
    with os.fdopen(fd, "w") as file:
        file.write(str(time.perf_counter()))


class _ProfileClient:
    """HTTP client for vLLM's runtime profiling control endpoints."""

    def start_profile(self, endpoint: str) -> None:
        import requests

        requests.post(f"{endpoint.rstrip('/')}/start_profile", timeout=30).raise_for_status()

    def stop_profile(self, endpoint: str) -> None:
        import requests

        requests.post(f"{endpoint.rstrip('/')}/stop_profile", timeout=STOP_TIMEOUT_SECONDS).raise_for_status()


class ProfileController:
    """Own one benchmark case's start/stop lifecycle and report capture failures."""

    def __init__(
        self, spec: ProfileSpec, targets: tuple[ServeInstance, ...], marker: Path, client: _ProfileClient | None = None
    ):
        self.spec, self.targets, self.marker = spec, targets, marker
        self.client = client or _ProfileClient()
        self.finished = threading.Event()
        self.result: dict = {"status": "partial", "targets": {}}
        self.thread = threading.Thread(target=self._run_profile_window, daemon=True)

    def start(self) -> None:
        """Start monitoring for the first benchmark request."""
        self.thread.start()

    def finish(self) -> dict:
        """Signal benchmark completion, wait for trace shutdown, and return status."""
        self.finished.set()
        self.thread.join(STOP_TIMEOUT_SECONDS + 60)
        if self.thread.is_alive():
            raise ArtifactError("Profiling controller timed out while stopping; traces are still in use")
        return self.result

    def _call_profile_endpoints(
        self, operation: Callable[[str], None], targets: tuple[ServeInstance, ...]
    ) -> dict[str, str | None]:
        """Call an operation once per unique endpoint and map errors to targets."""
        endpoints: dict[str, list[str]] = {}
        for target in targets:
            endpoints.setdefault(target.endpoint, []).append(target.name)
        with ThreadPoolExecutor(max_workers=len(endpoints) or 1) as pool:
            futures = {pool.submit(operation, endpoint): names for endpoint, names in endpoints.items()}
            errors_by_target = {}
            for future in as_completed(futures):
                try:
                    future.result()
                    error = None
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                errors_by_target.update(dict.fromkeys(futures[future], error))
            return errors_by_target

    def _run_profile_window(self) -> None:
        """Wait for the selected trigger, then capture a fixed-duration window."""
        try:
            while not self.finished.wait(POLL_INTERVAL_SECONDS):
                if self.marker.exists() and self.marker.stat().st_size:
                    first_request = float(self.marker.read_text())
                    break
            else:
                self.result = {"status": "partial", "reason": "no_request_before_benchmark_end", "targets": {}}
                return
            completed_requests = None
            if self.spec.start_after_requests is not None:
                while not self.finished.is_set():
                    completed_requests = sum(
                        int(path.read_text()) for path in (self.marker.parent / "completed_requests").glob("*.count")
                    )
                    if completed_requests >= self.spec.start_after_requests:
                        break
                    self.finished.wait(POLL_INTERVAL_SECONDS)
                else:
                    self.result = {
                        "status": "partial",
                        "reason": "benchmark_ended_before_request_threshold",
                        "completed_requests": completed_requests,
                        "targets": {},
                    }
                    return
            elif self.finished.wait(max(0, first_request + (self.spec.start_after or 0) - time.perf_counter())):
                self.result = {"status": "partial", "reason": "benchmark_ended_before_start", "targets": {}}
                return
            actual_start_after = time.perf_counter() - first_request
            print(f"[Profiling] start_profile -> {', '.join(t.name for t in self.targets)}", flush=True)
            start_errors = self._call_profile_endpoints(self.client.start_profile, self.targets)
            started_at = time.perf_counter()
            active_targets = tuple(target for target in self.targets if start_errors[target.name] is None)
            reason = "start_failed"
            try:
                if active_targets:
                    reason = "benchmark_ended" if self.finished.wait(self.spec.duration) else "duration_reached"
            finally:
                actual_duration = time.perf_counter() - started_at
                print(f"[Profiling] stop_profile ({reason}) -> {', '.join(t.name for t in self.targets)}", flush=True)
                # A failed start response may still have started some workers.
                stop_errors = self._call_profile_endpoints(self.client.stop_profile, self.targets)
            self.result = {
                "status": "success"
                if len(active_targets) == len(self.targets)
                and not any(stop_errors.values())
                and reason == "duration_reached"
                and actual_duration >= self.spec.duration
                else "partial",
                "reason": reason,
                "actual_start_after": actual_start_after,
                "completed_requests": completed_requests,
                "actual_duration": actual_duration,
                "targets": {
                    target.name: {
                        "start_error": start_errors[target.name],
                        "stop_error": stop_errors.get(target.name),
                    }
                    for target in self.targets
                },
            }
        except Exception as exc:
            self.result = {"status": "partial", "reason": f"{type(exc).__name__}: {exc}", "targets": {}}


# =============================================================================
# Raw artifact collection and upload
# =============================================================================

RAW_COMPRESSION_CONCURRENCY = 4
RAW_UPLOAD_CONCURRENCY = 2


def collect_raw_artifacts(
    root: Path,
    benchmark_artifact_name: str,
    targets: tuple[ServeInstance, ...],
    profile_result: dict,
) -> None:
    """Archive raw traces and update the prepared benchmark's result manifest.

    Successful target directories are cleared only after their gzip archives
    have closed. Failures are raised after the case manifest has been saved.
    """
    case_dir = root / safe_name(benchmark_artifact_name)
    case_dir.mkdir(parents=True, exist_ok=True)

    def compress_target(target: ServeInstance) -> dict:
        raw_trace_dir = Path(target.raw_trace_dir)
        archive_path = case_dir / f"{target.name}.tar.gz"
        target_state = dict(profile_result.get("targets", {}).get(target.name, {}))
        try:
            if not raw_trace_dir.exists() or not any(raw_trace_dir.iterdir()):
                raise ArtifactError(f"No profiler output in {raw_trace_dir}")
            started_at = time.monotonic()
            with tarfile.open(archive_path, "w:gz", compresslevel=GZIP_COMPRESSION_LEVEL) as tar:
                tar.add(raw_trace_dir, arcname=target.name)
            target_state.update(
                archive=str(archive_path.relative_to(root)),
                size_bytes=archive_path.stat().st_size,
                output="raw",
            )
            print(
                f"[Profiling] Compressed raw {benchmark_artifact_name}/{target.name}: "
                f"{target_state['size_bytes']} bytes "
                f"in {time.monotonic() - started_at:.1f}s",
                flush=True,
            )
            for child in raw_trace_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        except Exception as exc:
            target_state["artifact_error"] = f"{type(exc).__name__}: {exc}"
        return {
            "name": target.name,
            "node_index": target.node_index,
            **target_state,
        }

    with ThreadPoolExecutor(max_workers=RAW_COMPRESSION_CONCURRENCY) as pool:
        target_records = list(pool.map(compress_target, targets))
    manifest_path = root / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    benchmark_record = next(
        record for record in manifest["benchmarks"] if record["benchmark_artifact_name"] == benchmark_artifact_name
    )
    benchmark_record.update(
        {
            "actual_duration": profile_result.get("actual_duration", 0),
            "actual_start_after": profile_result.get("actual_start_after"),
            "completed_requests": profile_result.get("completed_requests"),
            "status": "partial"
            if profile_result.get("status") != "success"
            or any(record.get("artifact_error") for record in target_records)
            else "success",
            "reason": profile_result.get("reason"),
            "targets": target_records,
        }
    )
    manifest["status"] = "partial" if any(case["status"] != "success" for case in manifest["benchmarks"]) else "success"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if benchmark_record["status"] != "success":
        failures = [
            f"{target['name']}: {field}: {target[field]}"
            for target in target_records
            for field in ("start_error", "stop_error", "artifact_error")
            if target.get(field)
        ]
        raise ArtifactError(
            f"Profiling {benchmark_artifact_name} incomplete ({benchmark_record['reason']}): {'; '.join(failures)}"
        )


def upload_raw_artifacts(root: Path, prefix: str, bucket: str, endpoint: str, region: str) -> dict:
    """Upload raw archives concurrently, keeping their metadata local."""
    manifest_path = root / "profile_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError("Profiling job manifest is missing; initialize the job before testing")
    manifest = json.loads(manifest_path.read_text())
    manifest["storage"] = {"prefix": prefix, "bucket": bucket, "endpoint": endpoint, "region": region}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    client = create_obs_client(endpoint, region)
    transfer_config = create_transfer_config()
    raw_prefix = f"{prefix.strip('/')}/raw"

    def upload_target(target_record: dict) -> None:
        if "archive" not in target_record:
            return
        target_record.pop("obs_url", None)
        target_record.pop("upload_error", None)
        try:
            upload_archive(client, transfer_config, root, raw_prefix, bucket, target_record, "raw")
        except Exception as exc:
            target_record["upload_error"] = f"{type(exc).__name__}: {exc}"

    target_records = [target for case in manifest["benchmarks"] for target in case["targets"]]
    with ThreadPoolExecutor(max_workers=RAW_UPLOAD_CONCURRENCY) as pool:
        list(pool.map(upload_target, target_records))
    for target_record in target_records:
        if target_record.get("upload_error"):
            print(
                f"Profiling artifact upload failed: {target_record['archive']}: {target_record['upload_error']}",
                flush=True,
            )
    for case in manifest["benchmarks"]:
        if not case["targets"] or any(not target.get("obs_url") for target in case["targets"]):
            case["status"] = "partial"
            manifest["status"] = "partial"
    if not manifest["benchmarks"] and manifest["status"] != "skipped":
        manifest["status"] = "partial"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


# =============================================================================
# Command-line entry point
# =============================================================================


def main() -> None:
    """Normalize CI options, initialize a job, or upload its raw artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["configure", "initialize", "upload"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--options-json", default="{}")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-output", type=Path)
    parser.add_argument("--env-json", type=Path)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    args = parser.parse_args()
    if args.command == "configure":
        try:
            spec = ProfileSpec.from_options(json.loads(args.options_json))
        except (ValueError, TypeError) as exc:
            parser.error(f"Invalid profiling options: {exc}")
        if args.output:
            with args.output.open("a") as file:
                file.write(f"value={json.dumps(spec.options())}\n")
                file.write(f"enabled={str(spec.enabled).lower()}\n")
                file.write(f"parse={str(spec.enabled and spec.output == 'parsed').lower()}\n")
        if args.env_output:
            with args.env_output.open("a") as file:
                for name, value in spec.environment().items():
                    file.write(f"{name}={value}\n")
        if args.env_json:
            args.env_json.write_text(json.dumps({"profile_env": spec.environment()}))
        return
    if args.root is None:
        parser.error(f"{args.command} requires --root")
    if args.command == "initialize":
        spec = ProfileSpec.from_env()
        if spec.enabled:
            initialize_profile_run(args.root, spec)
        return
    if not args.prefix:
        parser.error("upload requires --prefix")

    from boto3.exceptions import S3UploadFailedError
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        result = upload_raw_artifacts(args.root, args.prefix, args.bucket, args.endpoint, args.region)
        if result["status"] == "partial":
            raise ArtifactError("Profiling raw artifact/upload incomplete; see manifest and per-target errors above")
    except ArtifactError as exc:
        raise SystemExit(f"[Profiling] {exc}") from None
    except (BotoCoreError, ClientError, S3UploadFailedError) as exc:
        raise SystemExit(f"[Profiling] OBS operation failed: {type(exc).__name__}: {exc}") from None


if __name__ == "__main__":
    main()
