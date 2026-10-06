# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re
import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/_e2e_nightly_single_node_560t.yaml"


@pytest.fixture(scope="module")
def nightly_job():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["e2e-nightly"]


@pytest.fixture(scope="module")
def bash():
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
    if executable is None:
        pytest.skip("Bash is required to exercise the nightly workflow scripts")
    return executable


def _step(job, name):
    return next(step for step in job["steps"] if step["name"] == name)


def _localize(value):
    # Run the workflow commands in a sandbox, including the YAML step's bashrc write.
    value = value.replace("/tmp/test-logs", "./test-logs")
    value = value.replace("/vllm-workspace/vllm-ascend", "./workspace")
    value = value.replace("~/.bashrc", "./bashrc")
    return re.sub(r"\$\{\{.*?\}\}", "test-input", value)


def _run_step(bash, tmp_path, job, name, *, commands="", extra_env=None):
    step = _step(job, name)
    environment = os.environ.copy()
    for values in (job.get("env") or {}, step.get("env") or {}, extra_env or {}):
        environment.update({key: _localize(str(value)) for key, value in values.items()})
    if os.name == "nt":
        commands = 'export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"\n' + commands
    return subprocess.run(
        [bash, "-e", "-c", commands + "\n" + _localize(step["run"])],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_run_step_accepts_empty_yaml_environment_blocks(bash, tmp_path):
    job = yaml.safe_load("""env:
steps:
  - name: Empty environment
    env:
    run: printf '%s' "$EXTRA_VALUE"
""")
    result = _run_step(bash, tmp_path, job, "Empty environment", extra_env={"EXTRA_VALUE": "present"})

    assert result.returncode == 0, result.stderr
    assert result.stdout == "present"


def test_log_artifact_is_available_for_both_test_entries(nightly_job):
    environment = nightly_job["env"]
    assert environment["ASCEND_GLOBAL_LOG_LEVEL"] == "1"
    assert environment["ASCEND_SLOG_PRINT_TO_STDOUT"] == "0"
    assert environment["ASCEND_PROCESS_LOG_PATH"] == "/tmp/test-logs/ascend"
    steps = nightly_job["steps"]
    assert steps[0]["name"] == "Prepare log directories"

    collect = _step(nightly_job, "Collect benchmark logs")
    upload = _step(nightly_job, "Upload test and Ascend logs (GitHub Artifacts)")
    for name in ("Run Pytest (py-driven)", "Run Pytest (YAML-driven)"):
        assert steps.index(_step(nightly_job, name)) < steps.index(collect)
    assert steps.index(collect) < steps.index(upload)
    assert collect["if"] == upload["if"] == "${{ always() }}"
    assert upload["continue-on-error"] is True
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["path"].rstrip("/") == "/tmp/test-logs"
    assert upload["with"]["retention-days"] == 7
    assert upload["with"]["if-no-files-found"] == "warn"
    # File-path inputs can contain characters prohibited in artifact names.
    assert "inputs.tests" not in upload["with"]["name"]
    assert "inputs.config_file_path" not in upload["with"]["name"]


@pytest.mark.parametrize("exit_code", [0, 3])
@pytest.mark.parametrize(
    ("step_name", "log_name"),
    [("Run Pytest (py-driven)", "pytest-driven.log"), ("Run Pytest (YAML-driven)", "yaml-test.log")],
)
def test_pytest_preserves_result_and_collects_console_and_plog(
    bash, tmp_path, nightly_job, step_name, log_name, exit_code
):
    prepared = _run_step(bash, tmp_path, nightly_job, "Prepare log directories")
    assert prepared.returncode == 0, prepared.stderr
    assert (tmp_path / "test-logs/ascend").is_dir()
    (tmp_path / "mock-pytest.sh").write_text(
        """printf 'pytest stdout\\n'
printf 'pytest stderr\\n' >&2
printf '%s\\n' "$ASCEND_GLOBAL_LOG_LEVEL" "$ASCEND_SLOG_PRINT_TO_STDOUT" "$ASCEND_PROCESS_LOG_PATH" > pytest-env.log
mkdir -p "$ASCEND_PROCESS_LOG_PATH/plog"
printf 'INFO mock CANN process log\\n' > "$ASCEND_PROCESS_LOG_PATH/plog/plog-mock.log"
exit "$MOCK_PYTEST_EXIT_CODE"
""",
        encoding="utf-8",
        newline="\n",
    )
    result = _run_step(
        bash,
        tmp_path,
        nightly_job,
        step_name,
        commands='pytest() { "$BASH" ./mock-pytest.sh "$@"; }',
        extra_env={"MOCK_PYTEST_EXIT_CODE": str(exit_code)},
    )

    assert result.returncode == exit_code, result.stderr
    expected_console = "pytest stdout\npytest stderr\n"
    assert expected_console in result.stdout
    assert (tmp_path / "test-logs" / log_name).read_text(encoding="utf-8") == expected_console
    assert (tmp_path / "pytest-env.log").read_text(encoding="utf-8").splitlines() == [
        "1",
        "0",
        "./test-logs/ascend",
    ]
    assert (tmp_path / "test-logs/ascend/plog/plog-mock.log").read_text(encoding="utf-8").startswith("INFO ")


def test_hardware_and_cann_console_are_saved(bash, tmp_path, nightly_job):
    prepared = _run_step(bash, tmp_path, nightly_job, "Prepare log directories")
    assert prepared.returncode == 0, prepared.stderr
    result = _run_step(
        bash,
        tmp_path,
        nightly_job,
        "Check npu and CANN info",
        commands="""npu-smi() { printf 'NPU information\\n'; printf 'NPU diagnostic\\n' >&2; }
cat() { printf 'CANN installation information\\n'; }
sed() { :; }
pip() { :; }
""",
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test-logs/npu-smi.log").read_text(encoding="utf-8") == ("NPU information\nNPU diagnostic\n")
    assert (tmp_path / "test-logs/cann-info.log").read_text(encoding="utf-8") == "CANN installation information\n"


@pytest.mark.parametrize("filenames", [(), ("output_model.txt", "output_model with spaces.txt")])
def test_benchmark_logs_are_collected_when_present(bash, tmp_path, nightly_job, filenames):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("not a benchmark log", encoding="utf-8")
    for name in filenames:
        (workspace / name).write_text(f"benchmark log: {name}\n", encoding="utf-8")

    result = _run_step(bash, tmp_path, nightly_job, "Collect benchmark logs")

    assert result.returncode == 0, result.stderr
    destination = tmp_path / "test-logs/aisbench"
    if filenames:
        assert sorted(path.name for path in destination.iterdir()) == sorted(filenames)
        for name in filenames:
            assert (destination / name).read_text(encoding="utf-8") == (workspace / name).read_text(encoding="utf-8")
    else:
        assert not destination.exists()
