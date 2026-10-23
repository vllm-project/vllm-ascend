# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import signal
import subprocess
import sys
import textwrap
import time
from contextlib import suppress
from pathlib import Path

import pytest
import regex as re

fcntl = pytest.importorskip("fcntl", reason="The CI mypy runner uses POSIX process groups")

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / ".github/workflows/scripts/run_mypy_parallel.py"
RUNNER_COMMAND = (
    sys.executable,
    "-c",
    "import runpy, sys; raise SystemExit(runpy.run_path(sys.argv[1])['main'](heartbeat_seconds=0.05))",
    str(RUNNER),
)
VERSIONS = ("3.10", "3.11", "3.12")
PROCESS_TIMEOUT_SECONDS = 15
pytestmark = pytest.mark.skipif(os.name != "posix", reason="The CI mypy runner uses Bash and POSIX process groups")


def fake_mypy_environment(tmp_path: Path, body: str) -> dict[str, str]:
    binary_dir = tmp_path / "fake bin"
    binary_dir.mkdir()
    binary = binary_dir / "mypy"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import fcntl, os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"state = Path({str(tmp_path)!r})\n"
        "version = sys.argv[sys.argv.index('--python-version') + 1]\n"
        "target = sys.argv[-1]\n" + textwrap.dedent(body),
        encoding="utf-8",
    )
    binary.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = str(binary_dir) + os.pathsep + environment["PATH"]
    return environment


@pytest.mark.parametrize("failed_version", [None, *VERSIONS])
def test_parallel_versions_preserve_checks_logs_and_failure(tmp_path: Path, failed_version: str | None):
    environment = fake_mypy_environment(
        tmp_path,
        textwrap.dedent(r"""
        with (state / f"calls-{version}").open("a") as calls:
            calls.write(target + "\n")
        (state / f"cache-{version}").write_text(os.environ["MYPY_CACHE_DIR"])
        assert sys.argv[1:-1] == [
            '--follow-imports', 'skip', '--check-untyped-defs', '--python-version', version,
        ]
        if target == 'vllm_ascend':
            (state / f"started-{version}").touch()
            deadline = time.monotonic() + 5
            while len(list(state.glob('started-*'))) != 3:
                if time.monotonic() > deadline:
                    sys.exit(90)  # A serial implementation cannot pass the barrier.
                time.sleep(0.01)
            time.sleep(0.2)  # Allow the heartbeat to observe active checks.
        print(f"checked {version} {target}", flush=True)
        """)
        + f"\nif version == {failed_version!r} and target == 'tests':\n"
        + "    print('tests/example.py:7: error: simulated type error', flush=True)\n"
        + "    sys.exit(2)\n",
    )
    result = subprocess.run(
        RUNNER_COMMAND,
        env=environment,
        capture_output=True,
        text=True,
        timeout=PROCESS_TIMEOUT_SECONDS,
    )

    assert result.returncode == (1 if failed_version else 0), result.stdout + result.stderr
    assert "[mypy] Running Python 3.10, 3.11, 3.12;" in result.stdout
    groups = re.findall(r"::group::mypy Python (\S+) \([^\n]+\)\n(.*?)::endgroup::", result.stdout, re.S)
    assert len(groups) == 3
    assert {version for version, _ in groups} == set(VERSIONS)
    for version, log in groups:
        assert (tmp_path / f"calls-{version}").read_text().splitlines() == ["vllm_ascend", "examples", "tests"]
        assert all(f"checked {version} {target}" in log for target in ("vllm_ascend", "examples", "tests"))
        assert not any(f"checked {other} " in log for other in VERSIONS if other != version)
    assert len({(tmp_path / f"cache-{version}").read_text() for version in VERSIONS}) == 3
    if failed_version:
        assert "\ntests/example.py:7: error: simulated type error\n" in result.stdout
        assert f"::error title=mypy Python {failed_version}::" in result.stdout


def test_cancellation_stops_descendants_and_prints_partial_logs(tmp_path: Path):
    environment = fake_mypy_environment(
        tmp_path,
        """
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        parent_lock = (state / f"parent-lock-{version}").open("w")
        fcntl.flock(parent_lock, fcntl.LOCK_EX)
        child = subprocess.Popen([
            sys.executable, '-c',
            'import fcntl, signal, sys, time; from pathlib import Path; '
            'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
            'lock = open(sys.argv[1], "w"); fcntl.flock(lock, fcntl.LOCK_EX); '
            'Path(sys.argv[2]).touch(); time.sleep(60)',
            str(state / f"child-lock-{version}"), str(state / f"child-ready-{version}"),
        ])
        while not (state / f"child-ready-{version}").exists():
            time.sleep(0.01)
        print(f"partial output for {version}", flush=True)
        (state / f"pids-{version}").write_text(f"{os.getpgrp()} {os.getpid()} {child.pid}")
        time.sleep(60)
        """,
    )
    process = subprocess.Popen(
        RUNNER_COMMAND,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
        while len(list(tmp_path.glob("pids-*"))) != 3:
            assert process.poll() is None
            assert time.monotonic() < deadline, "mypy children did not start"
            time.sleep(0.01)
        process.terminate()
        stdout, stderr = process.communicate(timeout=PROCESS_TIMEOUT_SECONDS)
        assert process.returncode != 0, stdout + stderr
        for version in VERSIONS:
            assert f"partial output for {version}" in stdout
        assert stdout.count("(cancelled)") == 3

        # Process exit releases locks, even if a killed descendant has not yet
        # been reaped. This also avoids platform-specific process inspection.
        deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
        for version in VERSIONS:
            for owner in ("parent", "child"):
                with (tmp_path / f"{owner}-lock-{version}").open() as lock:
                    while True:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            assert time.monotonic() < deadline, f"{owner} for Python {version} is still running"
                            time.sleep(0.01)
    finally:
        for path in tmp_path.glob("pids-*"):
            with suppress(ProcessLookupError):
                os.killpg(int(path.read_text().split()[0]), signal.SIGKILL)
        if process.poll() is None:
            process.kill()
        process.communicate()
