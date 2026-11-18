"""Parse raw profiling artifacts and publish parsed results from offline jobs."""

import argparse
import json
import shutil
import tarfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.profiling.storage import (
    GZIP_COMPRESSION_LEVEL,
    ArtifactError,
    create_obs_client,
    create_transfer_config,
    make_transfer_progress_callback,
    upload_archive,
)

# =============================================================================
# Raw artifact discovery and shard planning
# =============================================================================


def plan_artifact_shards(manifest_path: Path) -> list[int]:
    """Return one parsing shard per node from the downloaded capture index."""
    manifest = json.loads(manifest_path.read_text())
    node_indexes = sorted(
        {target["node_index"] for benchmark in manifest["benchmarks"] for target in benchmark["targets"]}
    )
    if not node_indexes and manifest["status"] != "skipped":
        raise ArtifactError("Profiling manifest has no targets but is not skipped")
    return node_indexes


# =============================================================================
# Rank parsing and publication
# =============================================================================


@dataclass(frozen=True)
class _ParseContext:
    """Shared paths, clients, and settings for one node's parsing work."""

    root: Path
    raw_root: Path
    raw_prefix: str
    parsed_root: Path
    parsed_prefix: str
    bucket: str
    node_index: int
    analyse: Callable
    download_client: Any
    upload_client: Any
    transfer_config: Any


def _analyse_rank_trace(
    context: _ParseContext,
    *,
    raw_archive: str,
    trace_index: int,
    trace_count: int,
    trace_dir: Path,
) -> Path:
    """Run torch-npu analysis for one rank and validate its parsed trace."""
    print(
        f"[Profiling] Parse node={context.node_index} {raw_archive} trace {trace_index}/{trace_count}: {trace_dir}",
        flush=True,
    )
    started_at = time.monotonic()
    context.analyse(str(trace_dir))
    parsed_trace_dir = trace_dir / "ASCEND_PROFILER_OUTPUT"
    trace_view = parsed_trace_dir / "trace_view.json"
    if not (parsed_trace_dir / "analyse.done").is_file() or not trace_view.is_file() or not trace_view.stat().st_size:
        raise ArtifactError(f"Incomplete parsed trace: {trace_dir}")
    print(f"[Profiling] Parsed {trace_dir} in {time.monotonic() - started_at:.1f}s", flush=True)
    return parsed_trace_dir


def _publish_parsed_rank(
    context: _ParseContext,
    *,
    benchmark_artifact_name: str,
    target_name: str,
    rank_record: dict,
    trace_dir: Path,
    rank_relative_path: Path,
    parsed_trace_dir: Path,
    raw_archive: str,
) -> None:
    """Compress and upload one parsed rank, then release its local workspace."""
    archive_relative_path = (
        Path(raw_archive).parent / target_name / rank_relative_path.with_name(f"{rank_relative_path.name}.tar.gz")
    )
    archive_path = context.parsed_root / archive_relative_path
    try:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.monotonic()
        with tarfile.open(archive_path, "w:gz", compresslevel=GZIP_COMPRESSION_LEVEL) as tar:
            tar.add(parsed_trace_dir, arcname=str(Path(target_name) / rank_relative_path / parsed_trace_dir.name))
            for metadata in (*trace_dir.glob("profiler_info*.json"), *trace_dir.glob("profiler_metadata.json")):
                tar.add(metadata, arcname=str(Path(target_name) / rank_relative_path / metadata.name))
        rank_record.update(
            archive=str(archive_relative_path),
            size_bytes=archive_path.stat().st_size,
            output="parsed",
        )
        print(
            f"[Profiling] Compressed parsed {archive_relative_path}: {rank_record['size_bytes']} bytes "
            f"in {time.monotonic() - started_at:.1f}s",
            flush=True,
        )
        upload_archive(
            context.upload_client,
            context.transfer_config,
            context.parsed_root,
            context.parsed_prefix,
            context.bucket,
            rank_record,
            "parsed",
        )
    except Exception as exc:
        rank_record["upload_error"] = f"{type(exc).__name__}: {exc}"
        failure_message = f"{benchmark_artifact_name}/{target_name}/{rank_relative_path}: {rank_record['upload_error']}"
        print(f"[Profiling] Publish failed: {failure_message}", flush=True)
    # Raw traces remain in OBS; failed attempts must not accumulate local output either.
    archive_path.unlink(missing_ok=True)
    shutil.rmtree(trace_dir)


def _parse_target_ranks(
    context: _ParseContext,
    *,
    benchmark_artifact_name: str,
    raw_archive: str,
    target_record: dict,
    target_dir: Path,
) -> None:
    """Overlap parsing with publication, keeping at most one rank waiting to publish."""
    trace_dirs = sorted(target_dir.rglob("*_ascend_pt"))
    if not trace_dirs:
        raise ArtifactError(f"No Ascend trace directories in {raw_archive}")

    target_record["ranks"] = []
    publish_future = None
    with ThreadPoolExecutor(max_workers=1) as publish_pool:
        for trace_index, trace_dir in enumerate(trace_dirs, 1):
            rank_relative_path = trace_dir.relative_to(target_dir)
            rank_record = {"name": str(rank_relative_path)}
            target_record["ranks"].append(rank_record)
            try:
                parsed_trace_dir = _analyse_rank_trace(
                    context,
                    raw_archive=raw_archive,
                    trace_index=trace_index,
                    trace_count=len(trace_dirs),
                    trace_dir=trace_dir,
                )
            except Exception as exc:
                rank_record["parse_error"] = f"{type(exc).__name__}: {exc}"
                failure_message = (
                    f"{benchmark_artifact_name}/{target_record['name']}/{rank_relative_path}: "
                    f"{rank_record['parse_error']}"
                )
                print(f"[Profiling] Parse failed: {failure_message}", flush=True)
                shutil.rmtree(trace_dir)
                continue
            # Do not parse another rank until the previous publication has released its files.
            if publish_future is not None:
                publish_future.result()
            publish_future = publish_pool.submit(
                _publish_parsed_rank,
                context,
                benchmark_artifact_name=benchmark_artifact_name,
                target_name=target_record["name"],
                rank_record=rank_record,
                trace_dir=trace_dir,
                rank_relative_path=rank_relative_path,
                parsed_trace_dir=parsed_trace_dir,
                raw_archive=raw_archive,
            )

        if publish_future is not None:
            publish_future.result()


def _parse_target_artifact(
    context: _ParseContext,
    *,
    benchmark_artifact_name: str,
    benchmark_index: int,
    target_index: int,
    raw_target_record: dict,
) -> dict:
    """Download, extract, parse, and publish one target's raw artifact."""
    target_record = deepcopy(raw_target_record)
    raw_archive = raw_target_record.get("archive")
    raw_obs_url = raw_target_record.get("obs_url")
    work_dir = context.root / "work" / f"benchmark_{benchmark_index}_target_{target_index}"
    try:
        if not raw_archive or not raw_obs_url:
            raise ArtifactError("Raw archive was not uploaded")
        raw_archive_path = context.raw_root / raw_archive
        raw_archive_path.parent.mkdir(parents=True, exist_ok=True)
        object_key = f"{context.raw_prefix}/{raw_archive}"
        remote_size_bytes = raw_target_record["size_bytes"]
        print(
            f"[Profiling] Download raw node={context.node_index} {raw_archive}: {remote_size_bytes} bytes", flush=True
        )
        context.download_client.download_file(
            context.bucket,
            object_key,
            str(raw_archive_path),
            Config=context.transfer_config,
            Callback=make_transfer_progress_callback(f"Download raw {raw_archive}", remote_size_bytes),
        )
        with tarfile.open(raw_archive_path) as tar:
            tar.extractall(work_dir, filter="data")
        raw_archive_path.unlink()
        _parse_target_ranks(
            context,
            benchmark_artifact_name=benchmark_artifact_name,
            raw_archive=raw_archive,
            target_record=target_record,
            target_dir=work_dir / target_record["name"],
        )
    except Exception as exc:
        target_record["parse_error"] = f"{type(exc).__name__}: {exc}"
        failure_message = f"{benchmark_artifact_name}/{target_record['name']}: {target_record['parse_error']}"
        print(f"[Profiling] Parse failed: {failure_message}", flush=True)
    finally:
        if raw_archive:
            (context.raw_root / raw_archive).unlink(missing_ok=True)
        shutil.rmtree(work_dir, ignore_errors=True)
    return target_record


def _target_failures(target: dict) -> list[str]:
    """Derive parse/upload failures from target and rank records, once each."""
    failures = []
    for record in [target, *target.get("ranks", [])]:
        label = target["name"] if record is target else f"{target['name']}/{record['name']}"
        for field in ("parse_error", "upload_error"):
            if record.get(field):
                failures.append(f"{label}: {field}: {record[field]}")
    return failures


def parse_artifact_shard(root: Path, manifest_path: Path, node_index: int) -> dict:
    """Parse and publish the target artifacts assigned to one node."""
    raw_manifest = json.loads(manifest_path.read_text())
    if raw_manifest["status"] == "skipped" and not raw_manifest["benchmarks"]:
        return {"status": "skipped", "targets": [], "failures": []}
    storage = raw_manifest["storage"]
    raw_root = root / "raw"
    raw_prefix = f"{storage['prefix'].strip('/')}/raw"
    download_client = create_obs_client(storage["endpoint"], storage["region"])
    transfer_config = create_transfer_config()
    from torch_npu.profiler.profiler import analyse

    parsed_root = root / "parsed"
    parsed_root.mkdir(parents=True, exist_ok=True)
    parsed_prefix = f"{storage['prefix'].strip('/')}/parsed"
    context = _ParseContext(
        root=root,
        raw_root=raw_root,
        raw_prefix=raw_prefix,
        parsed_root=parsed_root,
        parsed_prefix=parsed_prefix,
        bucket=storage["bucket"],
        node_index=node_index,
        analyse=analyse,
        download_client=download_client,
        upload_client=create_obs_client(storage["endpoint"], storage["region"]),
        transfer_config=transfer_config,
    )
    shard_target_records = []

    for benchmark_index, benchmark_record in enumerate(raw_manifest["benchmarks"]):
        for target_index, raw_target_record in enumerate(benchmark_record["targets"]):
            if raw_target_record["node_index"] != node_index:
                continue
            target_record = _parse_target_artifact(
                context,
                benchmark_artifact_name=benchmark_record["benchmark_artifact_name"],
                benchmark_index=benchmark_index,
                target_index=target_index,
                raw_target_record=raw_target_record,
            )
            shard_target_records.append(
                {"benchmark_index": benchmark_index, "target_index": target_index, "record": target_record}
            )

    failures = [
        f"{raw_manifest['benchmarks'][target['benchmark_index']]['benchmark_artifact_name']}/{failure}"
        for target in shard_target_records
        for failure in _target_failures(target["record"])
    ]
    shard = {
        "node_index": node_index,
        "status": "partial" if failures else "success",
        "targets": shard_target_records,
        "failures": failures,
    }
    shard_path = parsed_root / "shards" / f"node-{node_index}.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text(json.dumps(shard, indent=2))
    print(f"[Profiling] Node {node_index}: {shard['status']}; result saved to {shard_path}", flush=True)
    return shard


# =============================================================================
# Parsed manifest finalization
# =============================================================================


def _merge_parsed_shards(raw_manifest: dict, shards: list[dict]) -> tuple[dict, list[str]]:
    """Merge shard target records into a copied raw manifest."""
    parsed_manifest = deepcopy(raw_manifest)
    if raw_manifest["status"] == "skipped" and not raw_manifest["benchmarks"]:
        return parsed_manifest, []
    parsed_records_by_position = {}
    failures: list[str] = []
    for shard in shards:
        for shard_target in shard["targets"]:
            position = shard_target["benchmark_index"], shard_target["target_index"]
            parsed_records_by_position[position] = shard_target["record"]

    for benchmark_index, benchmark_record in enumerate(parsed_manifest["benchmarks"]):
        target_records = []
        benchmark_failures = []
        for target_index, raw_target_record in enumerate(benchmark_record["targets"]):
            target_record = parsed_records_by_position.get((benchmark_index, target_index))
            if target_record is None:
                target_record = deepcopy(raw_target_record)
                target_record["parse_error"] = "Parsed shard result is missing"
            else:
                target_record = deepcopy(target_record)
            benchmark_failures.extend(
                f"{benchmark_record['benchmark_artifact_name']}/{failure}"
                for failure in _target_failures(target_record)
            )
            target_records.append(target_record)
        benchmark_record["targets"] = target_records
        failures.extend(benchmark_failures)
        if benchmark_record.get("status") != "success" or benchmark_failures:
            benchmark_record["status"] = "partial"

    if parsed_manifest.get("status") != "success" or failures:
        parsed_manifest["status"] = "partial"
    return parsed_manifest, failures


def finalize_parsed_manifest(root: Path, shard_root: Path) -> dict:
    """Merge local shard results into the capture index, retaining raw links."""
    manifest_path = root / "profile_manifest.json"
    raw_manifest = json.loads(manifest_path.read_text())
    shards = [json.loads(path.read_text()) for path in sorted(shard_root.glob("node-*.json"))]
    parsed_manifest, failures = _merge_parsed_shards(raw_manifest, shards)
    manifest_path.write_text(json.dumps(parsed_manifest, indent=2))
    if parsed_manifest["status"] == "partial":
        raise ArtifactError(
            f"Offline profiling incomplete: {len(failures)} parse/upload failures; see manifest for raw/parsed status"
        )
    return parsed_manifest


# =============================================================================
# Command-line entry point
# =============================================================================


def main() -> None:
    """Plan, execute, or finalize parsing with local GitHub Artifact metadata."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--plan-output", type=Path, required=True)
    parse = commands.add_parser("parse")
    parse.add_argument("--manifest", type=Path, required=True)
    parse.add_argument("--root", type=Path, required=True)
    parse.add_argument("--node-index", type=int, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--root", type=Path, required=True)
    finalize.add_argument("--shard-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        nodes = plan_artifact_shards(args.manifest)
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(json.dumps({"node_index": nodes}))
    elif args.command == "parse":
        shard = parse_artifact_shard(args.root, args.manifest, args.node_index)
        if shard["status"] == "partial":
            raise ArtifactError(
                f"Offline profiling node {args.node_index}: {len(shard['failures'])} parse/upload failures"
            )
    else:
        finalize_parsed_manifest(args.root, args.shard_root)


if __name__ == "__main__":
    try:
        main()
    except ArtifactError as exc:
        raise SystemExit(f"[Profiling] {exc}") from None
