"""Raw archive retention, publication, and the downloadable metadata index."""

import json
import tarfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.profiling import artifacts, capture


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "capture"
    capture.initialize_profile_run(root, capture.ProfileSpec(enabled=True, output="raw"))
    targets = tuple(
        capture.ServeInstance.from_endpoint(f"dp-{i}", "http://leader:8000", node_index=i, root=root) for i in range(2)
    )
    capture.prepare_capture(root, "perf", "model__perf", targets)
    for target in targets:
        trace = Path(target.raw_trace_dir)
        trace.mkdir(parents=True)
        (trace / "raw.bin").write_bytes(b"trace")
    return SimpleNamespace(root=root, manifest=root / "profile_manifest.json", targets=targets)


@pytest.mark.parametrize("failure", [False, True])
def test_raw_files_are_released_only_after_successful_archiving(run, monkeypatch, failure):
    if failure:
        monkeypatch.setattr(capture.tarfile, "open", Mock(side_effect=OSError("disk full")))
    with pytest.raises(capture.ArtifactError, match="disk full") if failure else nullcontext():
        capture.collect_raw_artifacts(run.root, "model__perf", run.targets, {"status": "success"})
    saved = json.loads(run.manifest.read_text())
    assert saved["status"] == ("partial" if failure else "success")
    for target, record in zip(run.targets, saved["benchmarks"][0]["targets"]):
        assert (Path(target.raw_trace_dir) / "raw.bin").exists() is failure
        if not failure:
            with tarfile.open(run.root / record["archive"]) as archive:
                data = archive.extractfile(f"{target.name}/raw.bin")
                assert data is not None and data.read() == b"trace"


@pytest.mark.parametrize("failure", [False, True])
def test_index_lists_confirmed_uploads_and_meaningful_errors(run, tmp_path, monkeypatch, failure):
    capture.collect_raw_artifacts(
        run.root,
        "model__perf",
        run.targets,
        {
            "status": "success",
            "targets": {target.name: {"start_error": None, "stop_error": None} for target in run.targets},
        },
    )

    def upload(source, bucket, key, **kwargs):
        if failure and key.endswith("dp-1.tar.gz"):
            raise OSError("upload unavailable")

    monkeypatch.setattr(capture, "create_obs_client", lambda *_: Mock(upload_file=upload))
    monkeypatch.setattr(capture, "create_transfer_config", lambda: None)
    capture.upload_raw_artifacts(run.root, "job", "bucket", "https://obs.example", "region")
    bundle = tmp_path / "index"
    artifacts.write_artifact_bundle(run.manifest, bundle)
    saved = json.loads((bundle / "profile_manifest.json").read_text())
    assert saved["status"] == ("partial" if failure else "success")
    assert (bundle / saved["benchmarks"][0]["serve_manifest"]).is_file()
    assert not list(bundle.rglob("*.tar.gz"))
    for name in ("summary.txt", "download.sh", "raw/downloads.md"):
        content = (bundle / name).read_text()
        assert "https://bucket.obs.example/job/raw/model__perf/dp-0.tar.gz" in content
        assert ("https://bucket.obs.example/job/raw/model__perf/dp-1.tar.gz" in content) is not failure
    summary = (bundle / "summary.txt").read_text()
    assert ("FAILURES" in summary) is failure
    assert ": None" not in summary
    if failure:
        assert summary.count("upload_error: OSError: upload unavailable") == 1


def test_benchmarks_keep_independent_serving_snapshots(run):
    other = capture.ServeInstance.from_endpoint("other", "http://other:8000", root=run.root)
    capture.prepare_capture(run.root, "perf", "other__perf", (other,))
    before = run.manifest.read_text()
    with pytest.raises(FileExistsError):
        capture.prepare_capture(run.root, "perf", "model__perf", (other,))
    assert run.manifest.read_text() == before
    first, second = json.loads(before)["benchmarks"]
    assert capture.ServeManifest.read(run.root / first["serve_manifest"]).instances == run.targets
    assert capture.ServeManifest.read(run.root / second["serve_manifest"]).instances == (other,)
