"""Archive retention, upload status, and the downloadable metadata index."""

import json
import sys
import tarfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.profiling import artifacts, capture
from tools.profiling.storage import ArtifactError


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "capture"
    capture.initialize_profile_run(root, capture.ProfileSpec(enabled=True))
    targets = tuple(
        capture.ServeInstance.from_endpoint(f"dp-{node}", "http://leader:8000", node_index=node, root=root)
        for node in (0, 1)
    )
    capture.prepare_capture(root, "load", "model__load", targets)
    for target in targets:
        trace = Path(target.raw_trace_dir)
        trace.mkdir(parents=True)
        (trace / "raw.bin").write_bytes(b"trace")
    return SimpleNamespace(root=root, manifest=root / "profile_manifest.json", targets=targets)


@pytest.fixture
def archived(run):
    capture.collect_raw_artifacts(run.root, "model__load", run.targets, {"status": "success"})
    return run


@pytest.mark.parametrize("failed", [False, True])
def test_compression_releases_raw_only_after_success(run, monkeypatch, failed):
    if failed:
        monkeypatch.setattr(capture.tarfile, "open", Mock(side_effect=OSError("disk full")))
    with pytest.raises(ArtifactError, match="disk full") if failed else nullcontext():
        capture.collect_raw_artifacts(run.root, "model__load", run.targets, {"status": "success"})
    manifest = json.loads(run.manifest.read_text())
    assert manifest["status"] == ("partial" if failed else "success")
    for target, record in zip(run.targets, manifest["benchmarks"][0]["targets"]):
        assert (Path(target.raw_trace_dir) / "raw.bin").exists() is failed
        if not failed:
            with tarfile.open(run.root / record["archive"]) as archive:
                data = archive.extractfile(f"{target.name}/raw.bin")
                assert data is not None and data.read() == b"trace"


@pytest.mark.parametrize("failed", [False, True])
def test_upload_records_successes_and_failures_without_cloud_dependencies(archived, monkeypatch, failed):
    def upload(source, bucket, key, **kwargs):
        if failed and key.endswith("dp-1.tar.gz"):
            raise OSError("upload unavailable")

    monkeypatch.setattr(capture, "create_obs_client", lambda *_: Mock(upload_file=upload))
    monkeypatch.setattr(capture, "create_transfer_config", lambda: None)
    manifest = capture.upload_raw_artifacts(archived.root, "job", "bucket", "https://obs.example", "region")
    first, second = manifest["benchmarks"][0]["targets"]
    assert first["obs_url"] == "obs://bucket/job/raw/model__load/dp-0.tar.gz"
    assert ("obs_url" in second) is not failed
    assert manifest["status"] == ("partial" if failed else "success")
    if failed:
        assert "upload unavailable" in second["upload_error"]
    assert json.loads(archived.manifest.read_text()) == manifest


@pytest.mark.parametrize("mode", ["raw", "parsed"])
def test_index_preserves_metadata_and_lists_only_confirmed_uploads(archived, tmp_path, mode):
    manifest = json.loads(archived.manifest.read_text())
    manifest["requested"]["output"] = mode
    manifest["storage"] = {"prefix": "job", "bucket": "bucket", "endpoint": "https://obs.example"}
    target = manifest["benchmarks"][0]["targets"][0]
    target["obs_url"] = "obs://bucket/job/raw/model__load/dp-0.tar.gz"
    if mode == "parsed":
        target["ranks"] = [{"name": "tp0", "archive": "tp0.tar.gz", "obs_url": "obs://bucket/job/parsed/tp0.tar.gz"}]
    archived.manifest.write_text(json.dumps(manifest))
    bundle = tmp_path / "index"
    artifacts.write_artifact_bundle(archived.manifest, bundle)
    saved = json.loads((bundle / "profile_manifest.json").read_text())
    assert saved["status"] == "partial"  # dp-1 has not been uploaded.
    assert (bundle / saved["benchmarks"][0]["serve_manifest"]).is_file()
    assert not list(bundle.rglob("*.tar.gz")) and not list(bundle.rglob("*.bin"))
    for name in ("summary.txt", "download.sh", "raw/downloads.md"):
        text = (bundle / name).read_text()
        assert "https://bucket.obs.example/job/raw/model__load/dp-0.tar.gz" in text
        assert "https://bucket.obs.example/job/raw/model__load/dp-1.tar.gz" not in text
    assert (bundle / "parsed").exists() is (mode == "parsed")
    if mode == "parsed":
        assert "https://bucket.obs.example/job/parsed/tp0.tar.gz" in (bundle / "parsed/downloads.md").read_text()


def test_each_benchmark_keeps_its_serving_snapshot(run):
    replacement = capture.ServeInstance.from_endpoint("other", "http://other:8000", root=run.root)
    capture.prepare_capture(run.root, "load", "other_model__load", (replacement,))
    before = run.manifest.read_text()
    with pytest.raises(FileExistsError):
        capture.prepare_capture(run.root, "load", "model__load", (replacement,))
    assert run.manifest.read_text() == before
    first, second = json.loads(before)["benchmarks"]
    assert capture.ServeManifest.read(run.root / first["serve_manifest"]).instances == run.targets
    assert capture.ServeManifest.read(run.root / second["serve_manifest"]).instances == (replacement,)


@pytest.mark.parametrize("missing", [False, True])
def test_package_signals_upload_ready_only_after_success(archived, tmp_path, monkeypatch, missing):
    manifest = tmp_path / "missing.json" if missing else archived.manifest
    output = tmp_path / "github-output"
    argv = ["artifacts", "package", "--manifest", str(manifest), "--output", str(tmp_path / "index")]
    monkeypatch.setattr(sys, "argv", argv + ["--prefix", "job", "--github-output", str(output)])
    with pytest.raises(FileNotFoundError) if missing else nullcontext():
        artifacts.main()
    assert output.exists() is not missing
    if not missing:
        assert output.read_text() == "artifact_key=job\n"
