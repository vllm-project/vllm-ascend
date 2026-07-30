import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _find_bash() -> Path | None:
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            git_bash = Path(git).resolve().parents[1] / "bin" / "bash.exe"
            if git_bash.is_file():
                return git_bash
        return None
    bash = shutil.which("bash")
    return Path(bash) if bash else None


def _shell_path(bash: Path, path: Path) -> str:
    if os.name != "nt":
        return str(path)
    return subprocess.check_output(
        [str(bash), "-lc", 'cygpath -u "$1"', "_", str(path)],
        text=True,
    ).strip()


def test_aop_shell_forwards_complete_bisect_contract(tmp_path: Path):
    """Run the real AOP shell entry and inspect its two Python invocations."""
    bash = _find_bash()
    if bash is None:
        pytest.skip("bash is required to exercise aop_process.sh")
    repo = Path(__file__).resolve().parents[4]
    capture = tmp_path / "python-calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf 'CALL\\n' >> \"$AOP_CAPTURE\"\n"
        'for arg in "$@"; do printf \'ARG=%s\\n\' "$arg" >> "$AOP_CAPTURE"; done\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    good_table = tmp_path / "nightly" / "good_table.csv"
    env_table = good_table.with_name("env_table.csv")
    good_table.parent.mkdir()
    good_table.write_text("placeholder\n", encoding="utf-8")
    shell_fake_bin = _shell_path(bash, fake_bin)
    shell_capture = _shell_path(bash, capture)
    shell_good_table = _shell_path(bash, good_table)
    shell_env_table = _shell_path(bash, env_table)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{shell_fake_bin}:/usr/bin:/bin" if os.name == "nt" else f"{fake_bin}{os.pathsep}{env['PATH']}",
            "AOP_CAPTURE": shell_capture,
            "GOOD_TABLE": shell_good_table,
            "ENV_TABLE": shell_env_table,
        }
    )
    args = [
        "application",
        "1",
        "runner-a2",
        "",
        "case.yaml",
        "1 failed",
        "yaml failed",
        "single_node",
        "badbad1",
        "",
        "",
        "aop-case",
        "a2",
        "good123",
        "3",
        "120",
        "60",
        "true",
        "true",
        "true",
        "true",
        "since-build",
        "tests/e2e/models/configs",
    ]

    subprocess.run(
        [str(bash), "tests/e2e/nightly/scripts/aop_process.sh", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = [block.splitlines() for block in capture.read_text(encoding="utf-8").split("CALL\n") if block]
    assert len(calls) == 2
    assert [line.removeprefix("ARG=") for line in calls[1]] == [
        "-m",
        "tools.bisect.auto_bisect",
        "--scene",
        "single_node",
        "--bad-commit",
        "badbad1",
        "--good-table",
        shell_good_table,
        "--config-yaml",
        "case.yaml",
        "--name",
        "aop-case",
        "--soc",
        "a2",
        "--env-table",
        shell_env_table,
        "--good-commit",
        "good123",
        "--fail-confirm-retries",
        "3",
        "--trial-timeout-s",
        "120",
        "--barrier-timeout-s",
        "60",
        "--no-verify-good",
        "--no-verify-bad",
        "--force-initial-build",
        "--no-assume-built-head",
        "--native-check",
        "since-build",
        "--config-base-path",
        "tests/e2e/models/configs",
    ]
