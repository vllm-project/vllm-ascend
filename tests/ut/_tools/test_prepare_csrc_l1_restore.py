# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest
import yaml

from .build_cache_test_utils import ENGINE, REPO_ROOT, build_cache_command

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


def test_prepare_relative_cache_dir_across_workspaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = _load_helper("prepare_csrc_l1_restore_relative")
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout="primary\nsame\ncompat\n"),
    )
    for name in ("producer", "consumer"):
        workspace = tmp_path / name
        workspace.mkdir()
        source_root = _source_root(workspace)
        environment = workspace / "github-env"
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("GITHUB_ENV", str(environment))
        args = _args(workspace, source_root)
        args.cache_dir = "csrc/build_cache"

        assert helper.prepare(args)["supported"] == "true"
        assert (workspace / args.cache_dir).is_dir()
        assert f"VLLM_ASCEND_BUILD_CACHE_DIR={workspace / args.cache_dir}\n" in environment.read_text()


def test_l1_callers_share_relative_transport_path():
    # runs-on/cache versions the snapshot with the literal path input. Keep
    # this transport path stable even when runner/container workspaces differ.
    workflows = REPO_ROOT / ".github" / "workflows"
    callers = []
    for path in sorted(workflows.iterdir()):
        if path.suffix not in (".yaml", ".yml"):
            continue
        workflow = yaml.safe_load(path.read_text())
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", []):
                if "csrc-l1-restore" not in step.get("uses", "") and "csrc-l1-save" not in step.get("uses", ""):
                    continue
                cache_dir = step["with"]["cache-dir"]
                if cache_dir == "${{ env.INCREMENTAL_BUILD_CACHE_DIR }}":
                    cache_dir = workflow["env"]["INCREMENTAL_BUILD_CACHE_DIR"]
                assert cache_dir == "csrc/build_cache", f"{path.name}: {step['name']}"
                callers.append(path.name)

    assert "_build_csrc_cache.yaml" in callers
    assert "_selected_tests.yaml" in callers
    producer = yaml.safe_load((workflows / "_build_csrc_cache.yaml").read_text())
    install = next(step for step in producer["jobs"]["build"]["steps"] if step["name"] == "Install vllm-ascend")
    assert "VLLM_ASCEND_BUILD_CACHE_DIR" not in install.get("env", {})


@pytest.mark.parametrize("action,operation", [("csrc-l1-restore", "restore"), ("csrc-l1-save", "save")])
def test_l1_transport_failure_remains_nonfatal(action: str, operation: str):
    path = REPO_ROOT / ".github" / "actions" / action / "action.yaml"
    steps = yaml.safe_load(path.read_text())["runs"]["steps"]
    transport = next(step for step in steps if step.get("uses") == f"runs-on/cache/{operation}@v5")
    assert transport["continue-on-error"] is True


@pytest.mark.parametrize("entrypoint", ["engine", "restore-helper"])
def test_snapshot_key_before_build_dependencies(tmp_path: Path, entrypoint: str):
    # Restore runs before project dependencies are installed. Exercise the real
    # subprocess boundary with a Python environment that cannot import regex.
    python_root = tmp_path / "python"
    venv.EnvBuilder(with_pip=False).create(python_root)
    python_bin = python_root / "bin"
    python = python_bin / "python3"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PATH"] = f"{python_bin}{os.pathsep}{os.defpath}"
    output = tmp_path / "github-output"
    environment["GITHUB_OUTPUT"] = str(output)
    environment["GITHUB_ENV"] = str(tmp_path / "github-env")
    environment["RUNNER_TEMP"] = str(tmp_path)

    subprocess.run(
        [str(python), "-c", "import importlib.util; assert importlib.util.find_spec('regex') is None"],
        env=environment,
        check=True,
    )
    command = [
        str(python),
        str(ENGINE if entrypoint == "engine" else HELPER),
    ]
    if entrypoint == "engine":
        command += ["snapshot-key", "--unique-suffix", "bootstrap"]
    else:
        command += ["--cache-dir", str(tmp_path / "cache"), "--source-root", str(REPO_ROOT)]
    command += [
        "--architecture",
        "ARM64",
        "--soc-version",
        "a3",
        "--toolchain-image",
        "image@sha256:bootstrap",
        "--csrc-hash",
        "bootstrap-source",
    ]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    if entrypoint == "engine":
        assert len(result.stdout.splitlines()) == 3
        assert all(line.startswith("vllm-ascend-inc-v1-schema4-") for line in result.stdout.splitlines())
    else:
        values = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
        assert values["supported"] == "true", result.stdout
        assert values["primary_key"].startswith(values["same_csrc_prefix"])
        assert values["same_csrc_prefix"].startswith(values["compat_prefix"])


def test_cache_cold_cross_root_warm_and_mutation_without_build_dependencies(tmp_path: Path):
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", "import importlib.util; assert importlib.util.find_spec('regex') is None"],
        check=True,
    )
    builder = tmp_path / "builder.py"
    counter = tmp_path / "counter"
    builder.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "prepared, output, counter = map(Path, sys.argv[1:])\n"
        "output.mkdir(parents=True, exist_ok=True)\n"
        "(output / 'kernel.o').write_text(prepared.read_text().split('VALUE=')[1])\n"
        "count = int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(count + 1))\n",
        encoding="utf-8",
    )

    for root_name, value, status, build_count in [
        ("root-a", 1, "MISS", 1),
        ("root-b", 1, "HIT", 1),
        ("root-b", 2, "MISS", 2),
    ]:
        root = tmp_path / root_name
        root.mkdir(exist_ok=True)
        prepared = root / "input.txt"
        prepared.write_text(f'ROOT="{root}"\nVALUE={value}\n', encoding="utf-8")
        output = root / "output"
        command = build_cache_command(
            cache_root=tmp_path / "cache",
            domain="third_party",
            unit="standalone",
            output_dir=output,
            environment_profile="host-cxx",
            prepared_inputs=[prepared],
            recipe_values=[],
            environment_values=["toolchain=standalone-test"],
            environment_tools=[],
            build_command=[sys.executable, "-I", "-S", str(builder), str(prepared), str(output), str(counter)],
            normalize_paths=[root],
            artifact_includes=["kernel.o"],
        )
        command[1:1] = ["-I", "-S"]
        result = subprocess.run(command, capture_output=True, text=True)

        assert result.returncode == 0, result.stderr
        assert f"[build-cache] {status} " in result.stdout
        assert "BYPASS" not in result.stdout
        if status == "MISS":
            assert "[build-cache] SAVED " in result.stdout
        assert counter.read_text() == str(build_count)
        assert (output / "kernel.o").read_text() == f"{value}\n"


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
