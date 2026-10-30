# Copyright (c) 2026 Huawei Technologies Co., Ltd.
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path

import pytest
import yaml

from .build_cache_test_utils import REPO_ROOT

SCRIPTS = REPO_ROOT / ".github/workflows/scripts"


def load(name):
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(SCRIPTS))
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def test_snapshot_runs_in_exact_image_and_removes_container(monkeypatch):
    helper = load("image_csrc_cache")
    calls = []

    def docker(*args, **kwargs):
        calls.append(args)
        return {"create": "owned-container", "start": "primary\nsame\ncompat", "wait": "0"}.get(args[0], "")

    monkeypatch.setattr(helper, "docker", docker)
    assert (
        helper.snapshot_key("image@sha256:exact", Path("engine.py"), ["--soc-version", "a3"]) == "primary\nsame\ncompat"
    )
    assert calls[0][0] == "create"
    assert calls[0][1] == "--name"
    assert calls[0][2].startswith("va-cpu-csrc-l1-")
    assert calls[0][3:] == (
        "--network",
        "host",
        "--entrypoint",
        "python3",
        "image@sha256:exact",
        "/tmp/csrc-build-cache.py",
        "snapshot-key",
        "--soc-version",
        "a3",
    )
    assert calls[-1] == ("rm", "--force", "owned-container")
    assert not any("--env" in call or "--mount" in call or "--device" in call for call in calls)


def test_container_failure_is_not_success_and_is_cleaned(monkeypatch):
    helper = load("image_csrc_cache")
    calls = []

    def docker(*args, **kwargs):
        calls.append(args)
        return {"create": "owned-container", "wait": "17"}.get(args[0], "")

    monkeypatch.setattr(helper, "docker", docker)
    with pytest.raises(RuntimeError, match="status 17"):
        helper.snapshot_key("image", Path("engine.py"), [])
    assert calls[-1] == ("rm", "--force", "owned-container")


def test_archive_excludes_untracked_credentials_and_restored_cache(tmp_path, monkeypatch):
    helper = load("image_csrc_cache")
    (tmp_path / "setup.py").write_text("# tracked")
    (tmp_path / "credentials").write_text("not-for-docker")
    (tmp_path / "build_cache").mkdir()
    monkeypatch.setattr(
        helper.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=b"setup.py\0")
    )
    helper.source_archive(tmp_path, tmp_path / "source.tar")
    with tarfile.open(tmp_path / "source.tar") as archive:
        assert archive.getnames() == ["setup.py"]


@pytest.mark.parametrize("nightly", [False, True])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("soc", ["a3", "310p"])
def test_build_handoff_pruning_and_failure(tmp_path, monkeypatch, nightly, failure, soc):
    helper = load("image_csrc_cache")
    source = tmp_path / "source"
    source.mkdir()
    cache = source / "csrc/build_cache"
    cache.mkdir(parents=True)
    (cache / "pruned-old-hash").write_text("old")
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(helper, "source_archive", lambda *a: None)
    calls = []

    def docker(*args, **kwargs):
        calls.append(args)
        if args[0] == "create":
            return "owned-container"
        if args[0] == "wait":
            return "1" if failure else "0"
        if args[0] == "cp" and args[1] == f"owned-container:{helper.CACHE}":
            updated = Path(args[2])
            updated.mkdir()
            (updated / "retained-new-hash").write_text("new")
            (updated / ".updated").touch()
        if args[0] == "cp" and args[1].endswith("/vllm_ascend"):
            package = Path(args[2])
            package.mkdir()
            (package / "_C.so").write_text("native")
            (package / "unrelated.py").write_text("do not publish")
        return ""

    monkeypatch.setattr(helper, "docker", docker)
    (source / "vllm_ascend").mkdir()
    args = argparse.Namespace(
        source_root=str(source),
        cache_dir=str(cache),
        image="exact-image",
        soc_version=soc,
        nightly=nightly,
        artifacts=not nightly,
    )
    if failure:
        with pytest.raises(RuntimeError):
            helper.build(args)
        assert not output.exists()
        assert (cache / "pruned-old-hash").exists()
    else:
        helper.build(args)
        assert output.read_text() == "built=true\n"
        assert sorted(p.name for p in cache.iterdir()) == [".updated", "retained-new-hash"]
        assert (source / "vllm_ascend/_C.so").exists() is not nightly
        assert not (source / "vllm_ascend/unrelated.py").exists()
    command = calls[0][-1]
    assert ("uv pip install -e . --no-deps" in command) is nightly
    assert ("pip install uv uc-manager" in command) is nightly
    assert ("export UV_SYSTEM_PYTHON=1" in command) is nightly
    assert ("uv pip install -r /tmp/requirements-runtime.txt" in command) is nightly
    assert f"export SOC_VERSION={helper.SOCS[soc]}" in command
    assert f"export VLLM_BATCH_INVARIANT={0 if nightly or soc == '310p' else 1}" in command
    assert calls[-1] == ("rm", "--force", "owned-container")


def test_missing_native_artifacts_do_not_publish(tmp_path):
    helper = load("image_csrc_cache")
    package = tmp_path / "package"
    package.mkdir()
    with pytest.raises(ValueError, match="native shared libraries"):
        helper.publish_artifacts(package, tmp_path / "destination")
    assert not (tmp_path / "destination").exists()


def test_runtime_key_uses_nested_image_not_outer_environment(tmp_path, monkeypatch):
    helper = load("prepare_csrc_l1_restore")
    engine = tmp_path / "csrc/scripts/build_cache.py"
    engine.parent.mkdir(parents=True)
    engine.touch()
    observed = []

    def key(image, candidate, arguments):
        observed.append((image, candidate, arguments))
        return "primary\nsame\ncompat"

    monkeypatch.setattr(helper, "snapshot_key", key)
    args = argparse.Namespace(
        source_root=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        architecture="arm64",
        soc_version="a3",
        toolchain_image="",
        runtime_image="built-nightly@sha256:exact",
        csrc_hash="source",
    )
    assert helper.prepare(args)["primary_key"] == "primary"
    assert observed[0][0] == args.runtime_image
    assert observed[0][2][observed[0][2].index("--toolchain-image") + 1] == ""


def test_runtime_key_failure_is_fail_open(tmp_path, monkeypatch):
    helper = load("prepare_csrc_l1_restore")
    engine = tmp_path / "csrc/scripts/build_cache.py"
    engine.parent.mkdir(parents=True)
    engine.touch()
    monkeypatch.setattr(helper, "snapshot_key", lambda *a: (_ for _ in ()).throw(RuntimeError("docker unavailable")))
    args = argparse.Namespace(
        source_root=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        architecture="arm64",
        soc_version="a3",
        toolchain_image="",
        runtime_image="nightly",
        csrc_hash="source",
    )
    assert helper.prepare(args) == {"supported": "false"}


def workflow(name):
    return yaml.safe_load((REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def test_image_l0_priority_and_l1_failure_fallback():
    steps = workflow("_schedule_image_build.yaml")["jobs"]["build-push-digest"]["steps"]
    compiler = next(s for s in steps if s.get("id") == "compiler-image")
    assert compiler["if"] == "steps.cache-csrc.outputs.cache-hit != 'true'"
    assert compiler["continue-on-error"] is True
    l1 = next(s for s in steps if s.get("id") == "image-l1")
    assert l1["continue-on-error"] is True
    final = next(s for s in steps if s.get("id") == "build")
    assert "steps.image-l1.outputs.built == 'true'" in final["with"]["build-args"]
    assert "&& '0' || '1'" in final["with"]["build-args"]


def test_all_image_variants_expose_base_without_copying_l1():
    names = ["Dockerfile", "Dockerfile.a3", "Dockerfile.a5", "Dockerfile.310p"]
    names += [f"{name}.openEuler" if name != "Dockerfile" else "Dockerfile.openEuler" for name in list(names)]
    for name in names:
        text = (REPO_ROOT / name).read_text()
        assert " AS csrc-build-base" in text
        assert "install -y zstd " in text
        assert "export VLLM_ASCEND_BUILD_CACHE_DIR=/vllm-workspace/vllm-ascend/csrc/build_cache" in text
        assert text.index("FROM csrc-build-base AS runtime") < text.index("COPY . /vllm-workspace/vllm-ascend/")
        native_install = text[text.index("python3 -m pip install -e /vllm-workspace/vllm-ascend/") :].split("\n\n", 1)[
            0
        ]
        assert "rm -rf /vllm-workspace/vllm-ascend/csrc/build_cache" in native_install
    assert "csrc/build_cache" in (REPO_ROOT / ".dockerignore").read_text().splitlines()


def test_nightly_produces_from_loaded_image_and_writes_only_trusted_sources():
    steps = workflow("_nightly_image_build.yaml")["jobs"]["build"]["steps"]
    assert any("apt-get install -y zstd" in step.get("run", "") for step in steps)
    build = next(s for s in steps if s.get("id") == "build")
    assert '--load -t "$LOCAL_IMAGE"' in build["run"]
    assert "local_image=${IMAGE_ID}" in build["run"]
    l1 = next(s for s in steps if s.get("id") == "nightly-l1")
    assert l1["with"]["image"] == "${{ steps.build.outputs.local_image }}"
    assert l1["with"]["nightly"] == "true"
    for key in ("access-key-id", "secret-access-key"):
        gate = l1["with"][key]
        assert "github.repository == 'vllm-project/vllm-ascend'" in gate
        assert "github.event_name != 'pull_request'" in gate
        assert "github.event_name != 'pull_request_target'" in gate
        assert "inputs.should_push" in gate
        assert "inputs.vllm_ascend_branch == 'main'" in gate
    push = next(s for s in steps if s.get("name") == "Push the verified nightly image")
    assert steps.index(build) < steps.index(l1) < steps.index(push)


def test_writer_secrets_are_forwarded_through_every_image_caller():
    targets = {
        f"./.github/workflows/{name}"
        for name in (
            "_schedule_image_build.yaml",
            "_nightly_image_build.yaml",
            "_schedule_nightly_pair_image_build.yaml",
        )
    }
    for path in (REPO_ROOT / ".github/workflows").glob("*.yaml"):
        for job in yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs", {}).values():
            if job.get("uses") in targets:
                assert job["secrets"]["HW_OBS_AK"] == "${{ secrets.HW_OBS_AK }}", path
                assert job["secrets"]["HW_OBS_SK"] == "${{ secrets.HW_OBS_SK }}", path


def test_reject_cache_directory_outside_transport_scope(tmp_path):
    helper = load("image_csrc_cache")
    args = argparse.Namespace(source_root=str(tmp_path), cache_dir=str(tmp_path.parent))
    with pytest.raises(ValueError, match="transport"):
        helper.build(args)


def test_producer_pins_image_and_keeps_credentials_outside_build():
    action = yaml.safe_load((REPO_ROOT / ".github/actions/csrc-l1-produce/action.yaml").read_text(encoding="utf-8"))
    steps = action["runs"]["steps"]
    image = next(step for step in steps if step.get("id") == "image")
    restore = next(step for step in steps if step.get("id") == "restore")
    build = next(step for step in steps if step.get("id") == "build")
    save = steps[-1]
    assert "docker image inspect" in image["run"]
    assert restore["with"]["runtime-image"] == "${{ steps.image.outputs.identity }}"
    assert build["env"]["COMPILER_IMAGE"] == "${{ steps.image.outputs.identity }}"
    assert "secret" not in str(build).lower()
    assert save["with"]["cache-dir"] == restore["with"]["cache-dir"] == "csrc/build_cache"
    assert save["with"]["primary-key"] == "${{ steps.restore.outputs.primary-key }}"


@pytest.mark.skipif(not os.environ.get("VLLM_ASCEND_TEST_COMPILER_IMAGE"), reason="opt-in CPU Docker transport test")
def test_real_docker_cold_warm_and_source_mutation(tmp_path, monkeypatch):
    # This proves transport and native artifact handoff, not Ascend correctness
    # or production OBS authorization. The fixture uses synthetic CANN metadata.
    helper = load("image_csrc_cache")
    image = os.environ["VLLM_ASCEND_TEST_COMPILER_IMAGE"]
    fixture = Path(__file__).parent / "fixtures/image_csrc_cache"
    source = tmp_path / "source"
    shutil.copytree(fixture, source)
    engine = source / "csrc/scripts/build_cache.py"
    engine.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "csrc/scripts/build_cache.py", engine)
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    cache = source / "csrc/build_cache"
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("VLLM_ASCEND_BUILD_CACHE_EVENT_LOG", str(events))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    args = argparse.Namespace(
        source_root=str(source), cache_dir=str(cache), image=image, soc_version="a3", nightly=False, artifacts=True
    )
    artifacts = []
    for iteration in range(3):
        if iteration == 2:
            (source / "hello.cpp").write_text('extern "C" int cache_fixture_value() { return 43; }\n')
        helper.build(args)
        records = [json.loads(line) for line in events.read_text().splitlines()]
        statuses = [record.get("status") for record in records if record.get("event") == "cache_result"]
        assert ("HIT" in statuses) is (iteration == 1), records
        assert (cache / ".updated").exists() is (iteration != 1)
        artifacts.append(hashlib.sha256((source / "vllm_ascend/_C.so").read_bytes()).hexdigest())
    assert artifacts[0] == artifacts[1]
    assert artifacts[1] != artifacts[2]
    keys = helper.snapshot_key(
        image,
        engine,
        ["--architecture", "x64", "--soc-version", "a3", "--csrc-hash", "fixture", "--unique-suffix", "fixture"],
    )
    assert len(keys.splitlines()) == 3
    assert all(key.startswith("vllm-ascend-inc-v1-schema4-") for key in keys.splitlines())
    # Exercise the final image boundary as well: native L0 handoff is copied,
    # but the restored/updated L1 directory never enters COPY's layer.
    shutil.copy2(REPO_ROOT / ".dockerignore", source / ".dockerignore")
    (source / "Dockerfile.runtime").write_text(
        "ARG BASE_IMAGE\nFROM ${BASE_IMAGE} AS csrc-build-base\n"
        "FROM csrc-build-base AS runtime\nCOPY . /vllm-workspace/vllm-ascend/\n"
        "ARG COMPILE_CUSTOM_KERNELS=1\n"
        'RUN ["/bin/bash", "-c", "test \\"$COMPILE_CUSTOM_KERNELS\\" = 0 && '
        "test -f /vllm-workspace/vllm-ascend/vllm_ascend/_C.so && "
        '! test -d /vllm-workspace/vllm-ascend/csrc/build_cache"]\n'
    )
    final = f"va-local-cpu-csrc-runtime-{uuid.uuid4().hex[:12]}"
    try:
        subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                f"BASE_IMAGE={image}",
                "--build-arg",
                "COMPILE_CUSTOM_KERNELS=0",
                "-f",
                str(source / "Dockerfile.runtime"),
                "-t",
                final,
                str(source),
            ],
            check=True,
        )
    finally:
        subprocess.run(["docker", "image", "rm", final], check=False)
