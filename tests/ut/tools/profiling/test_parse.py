"""Parser output validation, bounded backlog, and partial-result preservation."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.profiling import parse
from tools.profiling.storage import ArtifactError


@pytest.mark.parametrize("complete", [False, True])
def test_analysis_requires_complete_profiler_output(tmp_path, complete):
    output = tmp_path / "ASCEND_PROFILER_OUTPUT"
    output.mkdir()
    (output / "trace_view.json").write_text('{"traceEvents":[]}')
    if complete:
        (output / "analyse.done").touch()
    context = SimpleNamespace(analyse=Mock(), node_index=0)
    with nullcontext() if complete else pytest.raises(ArtifactError, match="Incomplete parsed trace"):
        assert (
            parse._analyse_rank_trace(
                context, raw_archive="raw.tar.gz", trace_index=1, trace_count=1, trace_dir=tmp_path
            )
            == output
        )


def test_parser_keeps_at_most_one_completed_rank_waiting(tmp_path, monkeypatch):
    for index in range(3):
        (tmp_path / f"rank{index}_ascend_pt").mkdir()
    publishing, release, second_parsed, third_started = (Event() for _ in range(4))

    def analyse(context, *, trace_dir, **kwargs):
        if trace_dir.name.startswith("rank1"):
            second_parsed.set()
        elif trace_dir.name.startswith("rank2"):
            third_started.set()
        return trace_dir

    def publish(*args, **kwargs):
        publishing.set()
        assert release.wait(5)

    monkeypatch.setattr(parse, "_analyse_rank_trace", analyse)
    monkeypatch.setattr(parse, "_publish_parsed_rank", publish)
    target = {"name": "serve-0"}
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            parse._parse_target_ranks,
            Mock(),
            benchmark_artifact_name="load",
            raw_archive="raw.tar.gz",
            target_record=target,
            target_dir=tmp_path,
        )
        try:
            assert publishing.wait(5) and second_parsed.wait(5)
            assert not third_started.wait(0.05)
        finally:
            release.set()
        result.result(timeout=5)
    assert third_started.is_set()


@pytest.mark.parametrize("failure", [None, "parse_error", "upload_error", "missing_shard"])
def test_finalization_preserves_raw_and_available_parsed_results(tmp_path, failure):
    raw = {"name": "serve-0", "node_index": 0, "obs_url": "obs://bucket/raw.tar.gz"}
    manifest = {
        "status": "success",
        "benchmarks": [{"status": "success", "benchmark_artifact_name": "load", "targets": [raw]}],
    }
    path = tmp_path / "profile_manifest.json"
    path.write_text(json.dumps(manifest))
    shards = tmp_path / "shards"
    shards.mkdir()
    good_rank = {"name": "rank0", "obs_url": "obs://bucket/parsed.tar.gz"}
    record: dict = {**raw, "ranks": [good_rank]}
    if failure in ("parse_error", "upload_error"):
        record["ranks"].append({"name": "rank1", failure: "unavailable"})
    if failure != "missing_shard":
        shard = {"targets": [{"benchmark_index": 0, "target_index": 0, "record": record}]}
        (shards / "node-0.json").write_text(json.dumps(shard))
    assert parse.plan_artifact_shards(path) == [0]
    with pytest.raises(ArtifactError, match="incomplete") if failure else nullcontext():
        parse.finalize_parsed_manifest(tmp_path, shards)
    saved = json.loads(path.read_text())
    assert saved["status"] == ("partial" if failure else "success")
    target = saved["benchmarks"][0]["targets"][0]
    assert target["obs_url"] == raw["obs_url"]
    if failure == "missing_shard":
        assert "missing" in target["parse_error"]
    else:
        assert target["ranks"][0] == good_rank


@pytest.mark.parametrize("status", ["skipped", "partial"])
def test_empty_manifest_is_skipped_only_when_no_capture_was_selected(tmp_path, status):
    path = tmp_path / "profile_manifest.json"
    path.write_text(json.dumps({"status": status, "benchmarks": []}))
    with nullcontext() if status == "skipped" else pytest.raises(ArtifactError):
        assert parse.plan_artifact_shards(path) == []
        assert parse.finalize_parsed_manifest(tmp_path, tmp_path / "absent")["status"] == "skipped"
