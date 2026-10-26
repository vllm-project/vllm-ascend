# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github/workflows/scripts/run_singlecard_shards.py"
SPEC = importlib.util.spec_from_file_location("singlecard_shards", SCRIPT)
SHARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARDS)


def test_complete_files_partitioned_once(tmp_path):
    (tmp_path / "triton").mkdir()
    paths = [tmp_path / "test_same.py", tmp_path / "triton/test_same.py", tmp_path / "triton/test_other.py"]
    for path in paths:
        path.write_text("# test\n", encoding="utf-8")
    files = SHARDS.discover_files(tmp_path)
    timings = {paths[0].as_posix(): 30.0, paths[1].as_posix(): 10.0, paths[2].as_posix(): 20.0}
    parts = SHARDS.partition_files(files, 2, timings)
    flattened = [path for part in parts for path in part]
    assert sorted(flattened) == sorted(paths)
    assert len(set(flattened)) == len(paths)
    assert [sum(timings[path.as_posix()] for path in part) for part in parts] == [30.0, 30.0]
    assert SHARDS.partition_files(files, 2, timings) == parts
    with pytest.raises(ValueError, match="at least one file"):
        SHARDS.partition_files(files, 4, timings)


@pytest.mark.parametrize("visible,count,expected", [("5,2,7,1", 4, ["5", "2"]), (None, 4, ["0", "1"])])
def test_allocated_device_order(visible, count, expected):
    assert SHARDS.select_devices(2, count, visible) == expected


@pytest.mark.parametrize("workers,count,visible", [(4, 2, None), (2, 2, "5,5"), (2, 2, "5,2,7"), (2, 0, "")])
def test_invalid_allocation_rejected(workers, count, visible):
    with pytest.raises(ValueError):
        SHARDS.select_devices(workers, count, visible)


@pytest.mark.parametrize("fail", [False, True])
def test_parallel_execution_and_failure_summary(tmp_path, monkeypatch, fail):
    # Two real pytest subprocesses rendezvous before proceeding. This checks
    # concurrent launch, disjoint devices/caches, and end-to-end JUnit accounting
    # without importing an NPU runtime or any repository conftest.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    rendezvous = """import json, os, time
from pathlib import Path
import pytest

def test_rendezvous():
    device = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    allocation = {"device": device, "cache": os.environ["TRITON_CACHE_DIR"]}
    Path("device-" + device + ".json").write_text(json.dumps(allocation))
    deadline = time.monotonic() + 15
    while len(list(Path.cwd().glob("device-*.json"))) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(list(Path.cwd().glob("device-*.json"))) == 2
"""
    Path("test_a.py").write_text(rendezvous + "\ndef test_pass(): assert True\n", encoding="utf-8")
    Path("triton").mkdir()
    Path("triton/test_b.py").write_text(
        rendezvous + f"\ndef test_result(): assert {not fail}\n\ndef test_skip(): pytest.skip('hardware gate')\n",
        encoding="utf-8",
    )
    parts = SHARDS.partition_files(SHARDS.discover_files(Path(".")), 2, {})
    result = SHARDS.run_shards(parts, ["5", "2"], tmp_path / "results")
    assert result == int(fail)
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    assert summary["success"] is not fail
    assert summary["counts"] == {"passed": 4 - int(fail), "skipped": 1, "failed": int(fail), "errors": 0}
    assert sum(summary["file_durations"].values()) == pytest.approx(summary["cumulative_case_seconds"])
    assert summary["wall_seconds"] > 0
    allocations = [json.loads(path.read_text()) for path in Path.cwd().glob("device-*.json")]
    assert {item["device"] for item in allocations} == {"5", "2"}
    assert len({item["cache"] for item in allocations}) == 2
    assert all(os.path.commonpath([item["cache"], str(tmp_path)]) == str(tmp_path) for item in allocations)
    manifest = json.loads(Path("results/manifest.json").read_text())
    assert sorted(filename for item in manifest for filename in item["files"]) == ["test_a.py", "triton/test_b.py"]


def test_collection_error_is_not_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    Path("test_bad.py").write_text("raise RuntimeError('collection failure')\n", encoding="utf-8")
    assert SHARDS.run_shards([[Path("test_bad.py")]], ["0"], tmp_path / "results") == 1
    summary = json.loads(Path("results/summary.json").read_text())
    assert summary["success"] is False
    assert summary["workers"][0]["returncode"] != 0
    assert summary["counts"]["errors"] == 1


def test_native_exit_without_report_does_not_hide_other_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    Path("test_crash.py").write_text("import os\ndef test_crash(): os._exit(7)\n", encoding="utf-8")
    Path("test_ok.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    parts = [[Path("test_crash.py")], [Path("test_ok.py")]]
    assert SHARDS.run_shards(parts, ["0", "1"], tmp_path / "results") == 1
    summary = json.loads(Path("results/summary.json").read_text())
    assert summary["success"] is False
    assert summary["counts"]["passed"] == 1
    assert summary["workers"][0]["returncode"] == 7
    assert "report_error" in summary["workers"][0]
