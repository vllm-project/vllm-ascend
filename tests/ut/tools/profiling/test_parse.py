"""Offline parsing, bounded publication backlog, and partial-result indexes."""

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import ModuleType
from unittest.mock import Mock

import pytest

from tools.profiling import artifacts, capture, parse
from tools.profiling.storage import ArtifactError


@pytest.mark.parametrize("failure", [None, "parse", "upload", "incomplete"])
def test_parse_and_finalize_preserve_raw_and_publish_successful_ranks(tmp_path, monkeypatch, failure):
    root = tmp_path / "capture"
    capture.initialize_profile_run(root, capture.ProfileSpec(enabled=True))
    target = capture.ServeInstance.from_endpoint("serve-0", "http://server:8000", root=root)
    capture.prepare_capture(root, "perf", "model__perf", (target,))
    for rank in ("rank0_ascend_pt", "rank1_ascend_pt"):
        directory = Path(target.raw_trace_dir) / rank
        directory.mkdir(parents=True)
        (directory / "raw.bin").write_bytes(b"raw")
    capture.collect_raw_artifacts(root, "model__perf", (target,), {"status": "success"})
    manifest_path = root / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    raw = manifest["benchmarks"][0]["targets"][0]
    raw["obs_url"] = f"obs://bucket/job/raw/{raw['archive']}"
    manifest["storage"] = {"bucket": "bucket", "prefix": "job", "endpoint": "https://obs.example", "region": "region"}
    manifest_path.write_text(json.dumps(manifest))

    def analyse(directory):
        path = Path(directory)
        if failure == "parse" and path.name.startswith("rank1"):
            raise RuntimeError("analysis failed")
        output = path / "ASCEND_PROFILER_OUTPUT"
        output.mkdir()
        (output / "trace_view.json").write_text('{"traceEvents":[]}')
        if failure != "incomplete" or not path.name.startswith("rank1"):
            (output / "analyse.done").touch()

    def upload(source, bucket, key, **kwargs):
        if failure == "upload" and key.endswith("rank1_ascend_pt.tar.gz"):
            raise OSError("upload failed")

    client = Mock(upload_file=upload)
    client.download_file.side_effect = lambda bucket, key, destination, **kwargs: shutil.copyfile(
        root / raw["archive"], destination
    )
    monkeypatch.setattr(parse, "create_obs_client", lambda *_: client)
    monkeypatch.setattr(parse, "create_transfer_config", lambda: None)
    module = ModuleType("torch_npu.profiler.profiler")
    monkeypatch.setattr(module, "analyse", analyse, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    workspace = tmp_path / "parse"
    assert parse.plan_artifact_shards(manifest_path) == [0]
    shard = parse.parse_artifact_shard(workspace, manifest_path, 0)
    assert shard["status"] == ("partial" if failure else "success")
    with pytest.raises(ArtifactError, match="incomplete") if failure else nullcontext():
        parse.finalize_parsed_manifest(root, workspace / "parsed/shards")
    bundle = tmp_path / "index"
    artifacts.write_artifact_bundle(manifest_path, bundle)
    saved = json.loads((bundle / "profile_manifest.json").read_text())
    record = saved["benchmarks"][0]["targets"][0]
    assert record["obs_url"] == raw["obs_url"]
    assert saved["status"] == ("partial" if failure else "success")
    links = (bundle / "parsed/downloads.md").read_text()
    assert "rank0_ascend_pt.tar.gz" in links
    assert ("rank1_ascend_pt.tar.gz" in links) is not bool(failure)
    assert not list((workspace / "work").rglob("raw.bin"))
    assert not list(workspace.rglob("*.tar.gz"))


def test_publication_limits_completed_rank_backlog_to_one(tmp_path, monkeypatch):
    for index in range(3):
        (tmp_path / f"rank{index}_ascend_pt").mkdir()
    publishing, release, second, third = (Event() for _ in range(4))

    def analyse(context, *, trace_dir, **kwargs):
        if trace_dir.name.startswith("rank1"):
            second.set()
        elif trace_dir.name.startswith("rank2"):
            third.set()
        return trace_dir

    def publish(*args, **kwargs):
        publishing.set()
        assert release.wait(5)

    monkeypatch.setattr(parse, "_analyse_rank_trace", analyse)
    monkeypatch.setattr(parse, "_publish_parsed_rank", publish)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            parse._parse_target_ranks,
            Mock(),
            benchmark_artifact_name="perf",
            raw_archive="raw.tar.gz",
            target_record={"name": "serve-0"},
            target_dir=tmp_path,
        )
        try:
            assert publishing.wait(5) and second.wait(5)
            assert not third.wait(0.05)
        finally:
            release.set()
        future.result(timeout=5)
    assert third.is_set()


def test_missing_shard_records_failure_without_losing_raw_url(tmp_path):
    raw = {"name": "serve-0", "node_index": 0, "obs_url": "obs://bucket/raw.tar.gz"}
    manifest = {
        "status": "success",
        "benchmarks": [{"status": "success", "benchmark_artifact_name": "perf", "targets": [raw]}],
    }
    path = tmp_path / "profile_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="incomplete"):
        parse.finalize_parsed_manifest(tmp_path, tmp_path / "absent")
    saved = json.loads(path.read_text())
    assert saved["status"] == "partial"
    record = saved["benchmarks"][0]["targets"][0]
    assert record["obs_url"] == raw["obs_url"] and "missing" in record["parse_error"]


@pytest.mark.parametrize("status", ["skipped", "partial"])
def test_no_targets_is_valid_only_when_capture_was_skipped(tmp_path, status):
    path = tmp_path / "profile_manifest.json"
    path.write_text(json.dumps({"status": status, "benchmarks": []}))
    with nullcontext() if status == "skipped" else pytest.raises(ArtifactError):
        assert parse.plan_artifact_shards(path) == []
        assert parse.finalize_parsed_manifest(tmp_path, tmp_path / "absent")["status"] == "skipped"
