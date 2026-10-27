# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github/workflows/scripts/run_singlecard_shards.py"
SPEC = importlib.util.spec_from_file_location("singlecard_shards", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SHARDS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SHARDS
SPEC.loader.exec_module(SHARDS)


@pytest.fixture(autouse=True)
def isolate_child_pytest(monkeypatch):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    # Nested runner tests must not append to the outer CI job's real summary.
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


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


@pytest.mark.parametrize(
    "label,count",
    [
        ("linux-aarch64-a3-800i-8", 8),
        ("linux-aarch64-a5-8", 8),
        ("linux-aarch64-a2b1b3-1-hk-001", 1),
    ],
)
def test_runner_capacity_uses_registered_or_supported_labels(label, count):
    assert SHARDS.runner_device_count(label) == count


@pytest.mark.parametrize(
    "setting,runner,visible,files,expected",
    [
        ("auto", "linux-aarch64-a3-800i-8", 8, 112, 8),
        ("auto", "linux-aarch64-a5-8", 8, 3, 3),
        ("4", "linux-aarch64-a5-8", 8, 112, 4),
        ("2", "", 4, 112, 2),
    ],
)
def test_resolve_workers_respects_capacity_and_avoids_empty_shards(setting, runner, visible, files, expected):
    assert SHARDS.resolve_workers(setting, runner, visible, files) == expected


@pytest.mark.parametrize(
    "setting,runner,visible,files",
    [
        ("auto", "linux-aarch64-a5-8", 4, 112),
        ("auto", "linux-aarch64-a5-8", 16, 112),
        ("auto", "", 8, 112),
        ("auto", "arbitrary-runner-8", 8, 112),
        ("auto", "linux-amd64-cpu-8-hk", 0, 112),
        ("9", "linux-aarch64-a5-8", 16, 112),
        ("4", "linux-aarch64-a5-8", 2, 112),
        ("0", "", 8, 112),
        ("-1", "", 8, 112),
        ("1.5", "", 8, 112),
        ("1", "", 8, 0),
    ],
)
def test_invalid_worker_configuration_is_rejected(setting, runner, visible, files):
    with pytest.raises(ValueError):
        SHARDS.resolve_workers(setting, runner, visible, files)


def test_initial_partition_balances_file_count_independent_of_source_size(tmp_path):
    paths = [tmp_path / f"test_{index}.py" for index in range(7)]
    for index, path in enumerate(paths):
        path.write_text("# comment\n" * (10000 if index == 0 else 1), encoding="utf-8")
    original = SHARDS.partition_files(paths, 3, {})
    assert sorted(map(len, original)) == [2, 2, 3]
    paths[0].write_text("# small\n", encoding="utf-8")
    paths[-1].write_text("# large\n" * 10000, encoding="utf-8")
    assert SHARDS.partition_files(paths, 3, {}) == original


def _timing_metadata():
    return {
        "commit_sha": "tested-commit",
        "environment": {"soc_version": "a5", "python": "3.12.9", "triton_ascend": "3.6.0"},
        "test_files": ["test_a.py", "triton/test_b.py"],
    }


def test_timings_require_success_and_matching_environment_and_scope(tmp_path):
    metadata = _timing_metadata()
    payload = {
        "success": True,
        "metadata": metadata,
        "file_durations": {"test_a.py": 9.5, "triton/test_b.py": 0.5},
    }
    report = tmp_path / "summary.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert SHARDS.load_timings(report, metadata) == payload["file_durations"]

    for invalid in (
        {**payload, "success": False},
        {**payload, "metadata": {}},
        {**payload, "metadata": {**metadata, "environment": {"soc_version": "a3"}}},
        {**payload, "metadata": {**metadata, "test_files": ["test_a.py"]}},
    ):
        report.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError):
            SHARDS.load_timings(report, metadata)


def test_metadata_records_tested_revision_and_runtime_separately_from_workflow(monkeypatch):
    def git_command(command, **kwargs):
        value = "actual-tested-commit\n" if command[1] == "rev-parse" else " M local-change.py\n"
        return subprocess.CompletedProcess(command, 0, value, "")

    monkeypatch.setattr(SHARDS.subprocess, "run", git_command)
    monkeypatch.setenv("GITHUB_SHA", "workflow-commit")
    monkeypatch.setenv("GITHUB_RUN_ID", "1234")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/explicit-cache")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    runtime = {
        "device_count": 8,
        "device_name": "Ascend950",
        "architecture": "aarch64",
        "python": "3.12.9",
        "packages": {"torch": "2.10.0", "triton-ascend": "3.6.0"},
        "triton_path": "/runtime/triton/__init__.py",
    }
    args = argparse.Namespace(
        runner="linux-aarch64-a5-8", soc_version="a5", image="nightly-image", workers="auto", timings=None
    )
    metadata = SHARDS.build_metadata(runtime, args, [Path("test_a.py"), Path("test_b.py")], ["5", "2"])
    assert metadata["commit_sha"] == "actual-tested-commit"
    assert metadata["workflow_sha"] == "workflow-commit"
    assert metadata["source_dirty"] is True
    assert metadata["github_run_id"] == "1234"
    assert metadata["visible_device_count"] == 8
    assert metadata["devices"] == ["5", "2"]
    assert metadata["environment"]["workers"] == 2
    assert metadata["environment"]["packages"] == runtime["packages"]
    assert metadata["environment"]["cache"]["triton_root"] == "/tmp/explicit-cache"
    assert metadata["environment"]["cache"]["TRITON_ALWAYS_COMPILE"] == "1"


def test_runtime_probe_accepts_runtime_warnings_before_metadata(monkeypatch):
    runtime = {"device_count": 8, "device_name": "Ascend950"}
    monkeypatch.setattr(
        SHARDS.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "runtime warning\n" + SHARDS.RUNTIME_PREFIX + json.dumps(runtime) + "\n", ""
        ),
    )
    assert SHARDS.runtime_environment() == runtime


@pytest.mark.parametrize("returncode,stdout", [(1, ""), (0, "missing marker\n"), (0, "SINGLECARD_RUNTIME={invalid}\n")])
def test_runtime_probe_failure_is_explicit(monkeypatch, returncode, stdout):
    monkeypatch.setattr(
        SHARDS.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, stdout, "probe diagnostic"),
    )
    with pytest.raises(RuntimeError):
        SHARDS.runtime_environment()


def test_runtime_probe_timeout_is_explicit(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 60)

    monkeypatch.setattr(SHARDS.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="[Tt]ime|timed"):
        SHARDS.runtime_environment()


@pytest.mark.parametrize("fail", [False, True])
def test_parallel_execution_and_failure_summary(tmp_path, monkeypatch, fail):
    # Two real pytest subprocesses rendezvous before proceeding. This checks
    # concurrent launch, disjoint devices/caches, and end-to-end JUnit accounting
    # without importing an NPU runtime or any repository conftest.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))
    step_summary = tmp_path / "job-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
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
    metadata = _timing_metadata()
    result = SHARDS.run_shards(parts, ["5", "2"], tmp_path / "results", metadata=metadata)
    assert result == int(fail)
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    assert summary["success"] is not fail
    assert summary["counts"] == {"passed": 4 - int(fail), "skipped": 1, "failed": int(fail), "errors": 0}
    assert sum(summary["file_durations"].values()) == pytest.approx(summary["cumulative_case_seconds"])
    assert summary["wall_seconds"] > 0
    assert summary["metadata"] == metadata
    assert summary["interrupted"] is False
    assert summary["partial"] is False
    assert all(worker["wall_seconds"] > 0 for worker in summary["workers"])
    expected_statuses = ["failed", "passed"] if fail else ["passed", "passed"]
    assert sorted(worker["status"] for worker in summary["workers"]) == expected_statuses
    markdown = step_summary.read_text(encoding="utf-8")
    assert "| Pass | Skip | Fail | Error |" in markdown
    if fail:
        assert any("test_result" in case for worker in summary["workers"] for case in worker["failed_cases"])
        assert "test_result" in markdown
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
    Path("test_ok.py").write_text(
        "from pathlib import Path\ndef test_ok(): Path('executed').write_text('yes')\n", encoding="utf-8"
    )
    assert SHARDS.run_shards([[Path("test_bad.py"), Path("test_ok.py")]], ["0"], tmp_path / "results") == 1
    summary = json.loads(Path("results/summary.json").read_text())
    assert summary["success"] is False
    assert summary["partial"] is True
    assert not Path("executed").exists()
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


def test_partial_launch_failure_still_writes_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("test_a.py").write_text("def test_pass(): assert True\n", encoding="utf-8")
    Path("test_b.py").write_text("def test_pass(): assert True\n", encoding="utf-8")
    original_popen = SHARDS.subprocess.Popen
    launched: list[subprocess.Popen[str]] = []

    def fail_second_launch(*args, **kwargs):
        if launched:
            raise OSError("simulated process launch failure")
        process = original_popen(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(SHARDS.subprocess, "Popen", fail_second_launch)
    result = SHARDS.run_shards([[Path("test_a.py")], [Path("test_b.py")]], ["0", "1"], tmp_path / "results")
    assert result == 1
    assert launched[0].poll() is not None
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    assert summary["success"] is False
    assert summary["partial"] is True
    assert summary["workers"][1]["status"] == "not_started"
    assert "simulated process launch failure" in json.dumps(summary)


def test_interrupt_retains_completed_shards_and_stops_running_children(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("test_wait.py").write_text(
        "import time\nfrom pathlib import Path\n"
        "def test_wait():\n    Path('waiting').write_text('ready')\n    time.sleep(60)\n",
        encoding="utf-8",
    )
    Path("test_ok.py").write_text("def test_pass(): assert True\n", encoding="utf-8")
    observed: list[subprocess.Popen[str]] = []

    def interrupt_after_one_completes(workers):
        observed.extend(worker.process for worker in workers)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if Path("waiting").exists() and workers[1].process.poll() == 0:
                raise KeyboardInterrupt
            time.sleep(0.01)
        raise AssertionError("The fast worker did not finish while the other worker was running")

    monkeypatch.setattr(SHARDS, "wait_workers", interrupt_after_one_completes)
    result = SHARDS.run_shards([[Path("test_wait.py")], [Path("test_ok.py")]], ["0", "1"], tmp_path / "results")
    assert result == 130
    assert len(observed) == 2 and all(process.poll() is not None for process in observed)
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    assert summary["success"] is False
    assert summary["interrupted"] is True
    assert summary["partial"] is True
    assert summary["counts"] == {"passed": 1, "skipped": 0, "failed": 0, "errors": 0}
    assert [worker["status"] for worker in summary["workers"]] == ["interrupted", "passed"]


def test_partial_report_write_failure_does_not_skip_worker_cleanup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("test_wait.py").write_text("import time\ndef test_wait(): time.sleep(60)\n", encoding="utf-8")
    observed: list[subprocess.Popen[str]] = []
    original_write = SHARDS.write_summary
    writes = 0

    def fail_wait(workers):
        observed.extend(worker.process for worker in workers)
        raise RuntimeError("simulated controller failure")

    def fail_first_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 1:
            raise OSError("simulated disk full")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(SHARDS, "wait_workers", fail_wait)
    monkeypatch.setattr(SHARDS, "write_summary", fail_first_write)
    assert SHARDS.run_shards([[Path("test_wait.py")]], ["0"], tmp_path / "results") == 1
    assert observed and all(process.poll() is not None for process in observed)
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    assert summary["success"] is False
    assert summary["partial"] is True
    assert "simulated disk full" in summary["controller_error"]
