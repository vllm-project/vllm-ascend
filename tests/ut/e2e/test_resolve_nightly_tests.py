# SPDX-License-Identifier: Apache-2.0
"""PR-derived nightly matrix validation in resolve_nightly_tests.py.

CPU-only: cmd_matrix() must validate any ./pr/-derived MATRIX_FILE against
the base-branch config (unknown runner labels and shell metacharacters are
dropped) and must leave the trusted base config untouched, so normal nightly
runs cannot silently lose coverage.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github/workflows/scripts/resolve_nightly_tests.py"
SPEC = importlib.util.spec_from_file_location("resolve_nightly_tests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)

BASE_CONFIG = """\
a2:
  single_node:
    test_config:
      - name: known-runner
        os: linux-aarch64-a2b3-8
        tests: tests/e2e/test_a.py
"""

PR_CONFIG = """\
a2:
  single_node:
    test_config:
      - name: known-runner
        os: linux-aarch64-a2b3-8
        tests: tests/e2e/test_a.py
      - name: unknown-runner
        os: linux-aarch64-not-in-base
        tests: tests/e2e/test_b.py
      - name: unsafe-value
        os: linux-aarch64-a2b3-8
        tests: 'tests/x.py"; rm -rf /'
"""


def _write_yaml(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _run_matrix(tmp_path: Path, monkeypatch, matrix_file: str, allowlist_file: str | None = None) -> dict[str, list]:
    """Run cmd_matrix in a fake repo layout and parse its GITHUB_OUTPUT."""
    output = tmp_path / "github_output.txt"
    monkeypatch.setenv("MATRIX_FILE", matrix_file)
    monkeypatch.setenv("MATRIX_OUTPUTS", json.dumps({"single_node": "a2.single_node.test_config"}))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    if allowlist_file is None:
        monkeypatch.delenv("MATRIX_ALLOWLIST_FILE", raising=False)
    else:
        monkeypatch.setenv("MATRIX_ALLOWLIST_FILE", allowlist_file)
    monkeypatch.chdir(tmp_path)

    resolver.cmd_matrix(None)

    parsed = {}
    for line in output.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        parsed[name] = json.loads(value)
    return parsed


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _write_yaml(tmp_path / ".github/workflows/configs/nightly_config.yaml", BASE_CONFIG)
    _write_yaml(tmp_path / "pr/.github/workflows/configs/nightly_config.yaml", PR_CONFIG)
    return tmp_path


def test_pr_derived_matrix_is_validated(repo: Path, monkeypatch, capsys):
    parsed = _run_matrix(repo, monkeypatch, "./pr/.github/workflows/configs/nightly_config.yaml")

    # The entry matching the base config survives; the unknown runner label
    # and the shell-metacharacter value are dropped.
    assert parsed["single_node"] == [
        {"name": "known-runner", "os": "linux-aarch64-a2b3-8", "tests": "tests/e2e/test_a.py"}
    ]
    notices = capsys.readouterr().out
    assert "unknown runner label" in notices
    assert "unsafe characters" in notices


def test_base_config_is_never_filtered(tmp_path: Path, monkeypatch):
    # A value outside _SAFE_VALUE_CHARS must survive on the trusted base
    # config instead of silently shrinking nightly coverage.
    _write_yaml(
        tmp_path / ".github/workflows/configs/nightly_config.yaml",
        BASE_CONFIG.replace("tests/e2e/test_a.py", "cases/Model(1).yaml"),
    )
    parsed = _run_matrix(tmp_path, monkeypatch, ".github/workflows/configs/nightly_config.yaml")

    assert parsed["single_node"] == [
        {"name": "known-runner", "os": "linux-aarch64-a2b3-8", "tests": "cases/Model(1).yaml"}
    ]


def test_explicit_allowlist_equal_to_matrix_file_skips_validation(tmp_path: Path, monkeypatch):
    _write_yaml(
        tmp_path / ".github/workflows/configs/nightly_config.yaml",
        BASE_CONFIG.replace("tests/e2e/test_a.py", "cases/Model(1).yaml"),
    )
    parsed = _run_matrix(
        tmp_path,
        monkeypatch,
        ".github/workflows/configs/nightly_config.yaml",
        allowlist_file=".github/workflows/configs/nightly_config.yaml",
    )

    assert parsed["single_node"][0]["tests"] == "cases/Model(1).yaml"


def test_explicit_allowlist_override_validates_pr_config(repo: Path, monkeypatch):
    parsed = _run_matrix(
        repo,
        monkeypatch,
        "./pr/.github/workflows/configs/nightly_config.yaml",
        allowlist_file=".github/workflows/configs/nightly_config.yaml",
    )

    assert [entry["name"] for entry in parsed["single_node"]] == ["known-runner"]


def test_resolve_allowlist_file_derivation(monkeypatch):
    monkeypatch.delenv("MATRIX_ALLOWLIST_FILE", raising=False)
    assert resolver._resolve_allowlist_file(".github/workflows/configs/nightly_config.yaml") == ""
    assert resolver._resolve_allowlist_file("./pr/.github/workflows/configs/nightly_config.yaml") == (
        ".github/workflows/configs/nightly_config.yaml"
    )

    monkeypatch.setenv("MATRIX_ALLOWLIST_FILE", "explicit.yaml")
    assert resolver._resolve_allowlist_file("./pr/.github/workflows/configs/nightly_config.yaml") == "explicit.yaml"


def test_pr_config_without_base_counterpart_fails(tmp_path: Path, monkeypatch):
    # A PR cannot invent a config the base branch does not have: the derived
    # allowlist is missing and the resolver must fail loudly, not pass the
    # unvalidated matrix through.
    _write_yaml(tmp_path / "pr/.github/workflows/configs/nightly_config.yaml", PR_CONFIG)
    with pytest.raises(FileNotFoundError):
        _run_matrix(tmp_path, monkeypatch, "./pr/.github/workflows/configs/nightly_config.yaml")
