# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise CLI lifetime and round isolation with real CPU subprocesses."""

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from tools import aisbench


@pytest.fixture
def cli(tmp_path, monkeypatch):
    work = tmp_path / "work with spaces"
    work.mkdir()
    monkeypatch.chdir(work)
    script = tmp_path / "fake_aisbench.py"
    script.write_text(
        """import argparse, pathlib, signal, subprocess, sys, time
p = argparse.ArgumentParser()
p.add_argument('--work-dir')
a, _ = p.parse_known_args()
root = pathlib.Path(a.work_dir)
experiment = root / 'experiment'
experiment.mkdir()
print(f'Current exp folder: {experiment}', flush=True)
print(f'write csv to {experiment / "summary.csv"}', flush=True)
mode = sys.argv[sys.argv.index('--models') + 1]
if mode == 'stubborn_custom':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / 'ready').touch()
    time.sleep(60)
if mode == 'child_custom':
    child_code = ('import pathlib, signal, time; '
                  'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                  f'pathlib.Path({str(root / "child-ready")!r}).touch(); '
                  'time.sleep(60)')
    child = subprocess.Popen([sys.executable, '-c', child_code])
    (root / 'child.pid').write_text(str(child.pid))
    while not (root / 'child-ready').exists():
        time.sleep(0.01)
    sys.exit(7)
if mode == 'failed_custom':
    sys.exit(9)
if mode == 'error_custom':
    print('ERROR: response anomaly detection failed', flush=True)
time.sleep(0.2)
(root / 'finished').touch()
""",
        encoding="utf-8",
    )
    popen = subprocess.Popen
    launched = []

    def launch(command, **kwargs):
        assert not kwargs.get("shell")
        assert "&" not in command
        proc = popen([sys.executable, str(script), *command[1:]], **kwargs)
        launched.append(proc)
        return proc

    monkeypatch.setattr(aisbench.subprocess, "Popen", launch)
    yield work
    for proc in launched:
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        elif proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def runner(mode="success"):
    instance = aisbench.AisbenchRunner.__new__(aisbench.AisbenchRunner)
    instance.__dict__.update(
        task_type="accuracy",
        dataset_conf="gsm8k/test",
        request_conf=mode,
        num_prompts=100,
        benchmark_artifact_name="model__round",
        profile_marker=None,
        proc=None,
        exp_folder=None,
        result_line=None,
    )
    return instance


def test_rounds_ignore_stale_log_and_wait_past_result_marker(cli):
    (cli / "output_accuracy.txt").write_text("Current exp folder: /old\nwrite csv to /old/summary.csv\n")
    runs = []
    for _ in range(2):
        current = runner()
        current._run_aisbench_task()
        current._wait_for_task()
        assert (current.work_dir / "finished").exists()
        assert current.proc.returncode == 0
        assert Path(current.exp_folder).is_relative_to(current.work_dir)
        assert "/old/" not in current.result_line
        runs.append(current)
    assert runs[0].work_dir != runs[1].work_dir
    assert runs[0].stdout_file != runs[1].stdout_file
    assert all(r.stdout_file.exists() for r in runs)


@pytest.mark.parametrize("mode, message", [("failed", "exited with code 9"), ("error", "Aisbench runtime")])
def test_exit_code_and_errors_override_early_completion_marker(cli, mode, message):
    current = runner(mode)
    current._run_aisbench_task()
    with pytest.raises(RuntimeError, match=message):
        current._wait_for_task()
    assert current._process_cleaned_up
    assert current.result_line is None


@pytest.mark.parametrize("output", ["", "Current exp folder: {work}\n", "Current exp folder: /old\n"])
def test_successful_exit_without_current_results_fails(cli, monkeypatch, output):
    current = runner()
    current._run_aisbench_task()
    monkeypatch.setattr(current, "_check_runtime_stdout", lambda: output.format(work=current.work_dir))
    with pytest.raises(RuntimeError, match="without|outside this run"):
        current._wait_for_task()


def test_result_reader_rejects_another_round(cli):
    current = runner()
    current._run_aisbench_task()
    current._wait_for_task()
    current.result_line = f"write csv to {cli / 'old-summary.csv'}"
    with pytest.raises(RuntimeError, match="outside this run"):
        current._get_result_accuracy()


@pytest.mark.skipif(os.name != "posix", reason="Nightly workers use POSIX process groups")
def test_failed_cli_kills_worker_even_after_parent_exits(cli):
    current = runner("child")
    current._run_aisbench_task()
    with pytest.raises(RuntimeError, match="exited with code 7"):
        current._wait_for_task()
    pid = int((current.work_dir / "child.pid").read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("AISBench worker survived cleanup")


@pytest.mark.skipif(os.name != "posix", reason="Nightly workers use POSIX process groups")
def test_cleanup_escalates_when_cli_ignores_termination(cli, monkeypatch):
    current = runner("stubborn")
    current._run_aisbench_task()
    deadline = time.monotonic() + 5
    while not (current.work_dir / "ready").exists():
        assert time.monotonic() < deadline, "CLI did not start"
        time.sleep(0.01)
    monkeypatch.setattr(aisbench, "AISBENCH_STOP_TIMEOUT_SECONDS", 0.2)
    current._stop_aisbench_task()
    assert current.proc.returncode == -signal.SIGKILL
    assert current._process_cleaned_up


def test_constructor_cleans_up_before_next_case(cli, monkeypatch):
    instances: list[aisbench.AisbenchRunner] = []
    original = aisbench.AisbenchRunner._run_aisbench_task

    def start(current):
        if instances:
            assert instances[-1]._process_cleaned_up
            assert instances[-1].proc.poll() is not None
        current.request_conf = "failed" if not instances else "success"
        original(current)
        instances.append(current)

    monkeypatch.setattr(aisbench.AisbenchRunner, "_run_aisbench_task", start)
    monkeypatch.setattr(aisbench.AisbenchRunner, "_init_dataset_conf", lambda self: None)
    monkeypatch.setattr(aisbench.AisbenchRunner, "_init_request_conf", lambda self: None)
    monkeypatch.setattr(aisbench.AisbenchRunner, "_accuracy_verify", lambda self: setattr(self, "result", 67))
    config = dict(
        case_type="accuracy",
        benchmark_name="round",
        dataset_path_local=str(cli),
        model_path=str(cli),
        request_conf="success",
        dataset_conf="gsm8k/test",
        max_out_len=1024,
        batch_size=60,
    )
    with pytest.raises(AssertionError, match="some aisbench cases failed"):
        aisbench.run_aisbench_cases("model", 8000, [dict(config), dict(config)])
    assert len(instances) == 2
    assert all(r._process_cleaned_up for r in instances)


def test_cleanup_failure_stops_scheduling(cli, monkeypatch):
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise aisbench.AisbenchCleanupError("cleanup failed")

    monkeypatch.setattr(aisbench, "AisbenchRunner", fail)
    with pytest.raises(aisbench.AisbenchCleanupError):
        aisbench.run_aisbench_cases("model", 8000, [{"case_type": "accuracy"}] * 2)
    assert len(calls) == 1
