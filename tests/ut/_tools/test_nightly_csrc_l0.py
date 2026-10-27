# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import ctypes
import importlib.util
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from .build_cache_test_utils import REPO_ROOT, build_cache_command

SCRIPTS = REPO_ROOT / ".github/workflows/scripts"


def load_script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifacts(root):
    package = root / "vllm_ascend"
    package.mkdir(parents=True)
    (package / "vllm_ascend_C.so").write_bytes(b"native extension")
    (package / "lib").mkdir()
    (package / "lib/libcustom.so").write_bytes(b"native dependency")
    return package


def test_restore_validates_all_bytes_before_copying_into_pr(tmp_path, monkeypatch):
    helper = load_script("nightly_csrc_l0", monkeypatch)
    cache = tmp_path / "cache"
    package = artifacts(cache)
    source = tmp_path / "pr"
    (source / "vllm_ascend").mkdir(parents=True)
    python_file = source / "vllm_ascend/__init__.py"
    python_file.write_text("# PR source must survive", encoding="utf-8")
    helper.write_manifest(cache, "key")
    assert helper.validate_and_copy(cache, source, "key")
    assert (source / "vllm_ascend/lib/libcustom.so").read_bytes() == b"native dependency"
    assert python_file.read_text() == "# PR source must survive"
    (package / "lib/libcustom.so").write_bytes(b"corrupt")
    fresh = tmp_path / "fresh"
    assert not helper.validate_and_copy(cache, fresh, "key")
    assert not fresh.exists()


@pytest.mark.parametrize(
    "damage", ["missing-manifest", "wrong-key", "missing-file", "extra-file", "empty-extension", "escape"]
)
def test_invalid_bundle_is_a_miss(tmp_path, monkeypatch, damage):
    helper = load_script("nightly_csrc_l0", monkeypatch)
    package = artifacts(tmp_path)
    helper.write_manifest(tmp_path, "key")
    manifest = tmp_path / helper.MANIFEST
    if damage == "missing-manifest":
        manifest.unlink()
    elif damage == "wrong-key":
        data = json.loads(manifest.read_text())
        data["key"] = "different-runtime"
        manifest.write_text(json.dumps(data))
    elif damage == "missing-file":
        (package / "lib/libcustom.so").unlink()
    elif damage == "extra-file":
        (package / "stale.so").write_bytes(b"stale")
    elif damage == "empty-extension":
        (package / "vllm_ascend_C.so").write_bytes(b"")
    else:
        data = json.loads(manifest.read_text())
        data["files"]["../outside"] = "forged"
        manifest.write_text(json.dumps(data))
    assert not helper.validate_and_copy(tmp_path, tmp_path / "source", "key")


def test_refuse_to_publish_an_empty_extension(tmp_path, monkeypatch):
    helper = load_script("nightly_csrc_l0", monkeypatch)
    package = artifacts(tmp_path)
    (package / "vllm_ascend_C.so").write_bytes(b"")
    with pytest.raises(ValueError, match="Missing native extension"):
        helper.write_manifest(tmp_path, "key")
    assert not (tmp_path / helper.MANIFEST).exists()


@pytest.mark.parametrize(
    "dimension",
    [
        "runtime",
        "source",
        "python",
        "torch",
        "torch-npu",
        "compiler",
        "flags",
        "build_config",
        "torch-record",
        "torch-npu-record",
    ],
)
def test_nightly_key_tracks_runtime_source_and_abi(tmp_path, monkeypatch, dimension):
    helper = load_script("nightly_csrc_l0", monkeypatch)
    state = {"runtime": "CANN-OS-libc", "source": "source-hash", "torch": "2.10", "torch-npu": "2.10"}
    state["compiler"] = "toolchain-version"
    state["torch-record"] = "torch wheel contents"
    state["torch-npu-record"] = "torch-npu wheel contents"
    monkeypatch.setattr(helper, "_tracked_csrc_hash", lambda _: state["source"])
    monkeypatch.setattr(
        helper.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            version=state[name],
            read_text=lambda _: state[name + "-record"],
        ),
    )

    def snapshot(command, **kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, state["compiler"])
        source_hash = command[command.index("--csrc-hash") + 1]
        return subprocess.CompletedProcess(command, 0, f"unique\n{state['runtime']}-{source_hash}-\ncompat-\n")

    monkeypatch.setattr(helper.subprocess, "run", snapshot)
    first = helper.cache_key(tmp_path, "ARM64", "a2")
    if dimension == "python":
        monkeypatch.setattr(helper.sys.implementation, "cache_tag", "another-abi")
    elif dimension == "flags":
        monkeypatch.setenv("CXXFLAGS", "-DCHANGED_BUILD_OPTION")
    elif dimension == "build_config":
        (tmp_path / "pyproject.toml").write_text("# changed build requirements", encoding="utf-8")
    else:
        state[dimension] += "-changed"
    assert first != helper.cache_key(tmp_path, "ARM64", "a2")


def test_missing_package_record_is_not_assumed_compatible(monkeypatch):
    helper = load_script("nightly_csrc_l0", monkeypatch)
    monkeypatch.setattr(
        helper.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(
            version="same-version",
            read_text=lambda _: None,
        ),
    )
    with pytest.raises(ValueError, match="No installed artifact record"):
        helper.package_identity("torch-npu")


@pytest.mark.parametrize(
    "filename",
    [
        "Dockerfile",
        "Dockerfile.a3",
        "Dockerfile.a5",
        "Dockerfile.310p",
        "Dockerfile.openEuler",
        "Dockerfile.a3.openEuler",
        "Dockerfile.a5.openEuler",
        "Dockerfile.310p.openEuler",
    ],
)
def test_image_targets_follow_actual_dockerfile(tmp_path, monkeypatch, filename):
    resolver = load_script("resolve_csrc_cache_targets", monkeypatch)
    path = REPO_ROOT / filename
    targets = resolver.resolve_build_targets(dockerfile=str(path))
    assert {target["cache_arch"] for target in targets} == {"ARM64", "X64"}
    image = next(line.split()[1] for line in path.read_text().splitlines() if line.startswith("FROM "))
    assert all(target["image"] == image and not target["nightly"] for target in targets)
    assert all(target["soc_version"].startswith("ascend") for target in targets)


@pytest.mark.parametrize("soc", ["a2", "a3", "a5", "310P"])
def test_nightly_targets_use_runtime_without_changing_configured_targets(monkeypatch, soc):
    resolver = load_script("resolve_csrc_cache_targets", monkeypatch)
    previous = resolver.resolve_targets(None)
    targets = resolver.resolve_build_targets(nightly_image="nightly@sha256:exact", soc_version=soc)
    assert all(target["nightly"] and target["image"] == "nightly@sha256:exact" for target in targets)
    assert resolver.resolve_targets(None) == previous


def workflow(name):
    return yaml.load((REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_producer_cache_contract_matches_consumer_and_backend_is_explicit():
    producer_steps = workflow("_build_csrc_cache.yaml")["jobs"]["build"]["steps"]
    saves = [step for step in producer_steps if step.get("uses") == "actions/cache/save@v4"]
    assert len(saves) == 2
    nightly_save = next(step for step in saves if ".nightly-csrc-l0.json" in step["with"]["path"])
    action = yaml.load(
        (REPO_ROOT / ".github/actions/nightly-csrc-l0-restore/action.yaml").read_text(), Loader=yaml.BaseLoader
    )
    restore = next(step for step in action["runs"]["steps"] if step.get("uses") == "actions/cache/restore@v4")
    assert nightly_save["with"]["path"] == restore["with"]["path"]
    for path in (REPO_ROOT / ".github/workflows").glob("*.y*ml"):
        for job in workflow(path.name).get("jobs", {}).values():
            for step in job.get("steps", []):
                if "vllm-ascend-build-v2-" in step.get("with", {}).get("key", ""):
                    assert step["uses"].startswith("actions/cache/"), (path.name, step)


def test_all_pr_nightly_consumers_wait_for_production_but_keep_fallback():
    found = 0
    for path in (REPO_ROOT / ".github/workflows").glob("schedule_*test_*.yaml"):
        jobs = workflow(path.name)["jobs"]
        for job in jobs.values():
            if job.get("uses", "").endswith(
                (
                    "_e2e_nightly_single_node.yaml",
                    "_e2e_nightly_single_node_560t.yaml",
                    "_e2e_nightly_single_node_models.yaml",
                )
            ):
                found += 1
                assert "ensure-pr-csrc-cache" in job["needs"], path.name
                assert "always()" in job["if"]
                assert "secrets" not in jobs["ensure-pr-csrc-cache"]
    assert found >= 9


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("cc") is None, reason="Linux C compiler required")
def test_real_cpu_l1_to_l0_handoff_without_consumer_compilation(tmp_path, monkeypatch):
    helper = load_script("nightly_csrc_l0", monkeypatch)
    builder = tmp_path / "builder.py"
    counter = tmp_path / "compile-count"
    builder.write_text(
        "import subprocess,sys\nfrom pathlib import Path\n"
        "source,output,counter=map(Path,sys.argv[1:])\n"
        "output.mkdir(parents=True,exist_ok=True)\n"
        "subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(output/'vllm_ascend_C.so')],check=True)\n"
        "counter.write_text(str(int(counter.read_text())+1) if counter.exists() else '1')\n",
        encoding="utf-8",
    )
    for name, expected in (("cold-producer", "MISS"), ("warm-producer", "HIT")):
        producer = tmp_path / name
        producer.mkdir()
        source = producer / "native.c"
        source.write_text("int answer(void) { return 42; }\n", encoding="utf-8")
        output = producer / "vllm_ascend"
        command = build_cache_command(
            cache_root=tmp_path / "l1",
            domain="third_party",
            unit="cpu-handoff",
            output_dir=output,
            environment_profile="host-cxx",
            prepared_inputs=[source],
            recipe_values=[],
            environment_values=[],
            environment_tools=["cc"],
            build_command=[sys.executable, str(builder), str(source), str(output), str(counter)],
            normalize_paths=[producer],
            artifact_includes=["*.so"],
        )
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        assert f"[build-cache] {expected} " in result.stdout
        assert counter.read_text() == "1"
        helper.write_manifest(producer, "runtime-key")
        consumer = tmp_path / f"consumer-{name}"
        assert helper.validate_and_copy(producer, consumer, "runtime-key")
        library = ctypes.CDLL(str(consumer / "vllm_ascend/vllm_ascend_C.so"))
        assert library.answer() == 42
        assert counter.read_text() == "1"
