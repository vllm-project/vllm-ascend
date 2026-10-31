# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

from .build_cache_test_utils import REPO_ROOT

HELPER = REPO_ROOT / ".github" / "workflows" / "scripts" / "prepare_csrc_l1_restore.py"


def _load_helper(name: str):
    spec = importlib.util.spec_from_file_location(name, HELPER)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _args(tmp_path: Path, source_root: Path, *, csrc_hash: str = "caller-hash") -> argparse.Namespace:
    return argparse.Namespace(
        cache_dir=str(tmp_path / "cache"),
        architecture="arm64",
        soc_version="a2",
        toolchain_image="image@sha256:digest",
        source_root=str(source_root),
        csrc_hash=csrc_hash,
    )


def _source_root(tmp_path: Path) -> Path:
    source_root = tmp_path / "source"
    engine = source_root / "csrc" / "scripts" / "build_cache.py"
    engine.parent.mkdir(parents=True)
    engine.write_text("# engine\n", encoding="utf-8")
    return source_root


def test_prepare_uses_engine_output_and_exports_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = _load_helper("prepare_csrc_l1_restore_valid")
    source_root = _source_root(tmp_path)
    output = tmp_path / "github-output"
    environment = tmp_path / "github-env"
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_ENV", str(environment))
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    monkeypatch.setenv("GITHUB_JOB", "selected/a2")
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="primary\nsame-csrc-\ncompat-\n",
            stderr="",
        ),
    )

    values = helper.prepare(_args(tmp_path, source_root))
    helper._append_values(output, values)

    assert values == {
        "supported": "true",
        "primary_key": "primary",
        "csrc_hash": "caller-hash",
        "same_csrc_prefix": "same-csrc-",
        "compat_prefix": "compat-",
    }
    assert "supported=true" in output.read_text(encoding="utf-8")
    exported = environment.read_text(encoding="utf-8")
    assert f"VLLM_ASCEND_BUILD_CACHE_DIR={tmp_path / 'cache'}" in exported
    assert f"VLLM_ASCEND_BUILD_CACHE_EVENT_LOG={runner_temp / 'csrc-l1-selected_a2.jsonl'}" in exported


def test_prepare_missing_engine_is_unsupported_but_exports_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = _load_helper("prepare_csrc_l1_restore_missing")
    source_root = tmp_path / "old-source"
    source_root.mkdir()
    environment = tmp_path / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(environment))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    assert helper.prepare(_args(tmp_path, source_root)) == {"supported": "false"}
    assert "VLLM_ASCEND_BUILD_CACHE_DIR=" in environment.read_text(encoding="utf-8")


@pytest.mark.parametrize("stdout", ["", "one\ntwo\n", "one\ntwo\nthree\nfour\n"])
def test_prepare_rejects_invalid_snapshot_key_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
):
    helper = _load_helper("prepare_csrc_l1_restore_invalid")
    source_root = _source_root(tmp_path)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_ENV", raising=False)
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=""),
    )

    assert helper.prepare(_args(tmp_path, source_root)) == {"supported": "false"}


def test_prepare_snapshot_key_failure_is_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = _load_helper("prepare_csrc_l1_restore_failure")
    source_root = _source_root(tmp_path)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_ENV", raising=False)

    def fail(command, **_kwargs):
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(helper.subprocess, "run", fail)
    assert helper.prepare(_args(tmp_path, source_root)) == {"supported": "false"}


def test_prepare_generates_tracked_csrc_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = _load_helper("prepare_csrc_l1_restore_generated_hash")
    source_root = _source_root(tmp_path)
    manifest = b"100644 blob 0\tcsrc/kernel.cpp\n"
    seen: list[list[str]] = []
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_ENV", raising=False)

    def fake_run(command, **_kwargs):
        seen.append(command)
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout=manifest, stderr=b"")
        return subprocess.CompletedProcess(command, 0, stdout="primary\nsame\ncompat\n", stderr="")

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    values = helper.prepare(_args(tmp_path, source_root, csrc_hash=""))

    expected_hash = hashlib.sha256(manifest).hexdigest()
    assert values["csrc_hash"] == expected_hash
    snapshot_command = seen[-1]
    assert snapshot_command[snapshot_command.index("--csrc-hash") + 1] == expected_hash
