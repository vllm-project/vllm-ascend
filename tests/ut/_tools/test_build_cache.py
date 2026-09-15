# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE = REPO_ROOT / "csrc" / "scripts" / "build_cache.py"
_KEY_RE = re.compile(r"\bkey=([0-9a-f]{64})\b")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_builder(tmp_path: Path) -> Path:
    builder = tmp_path / "fake_builder.py"
    builder.write_text(
        '''from pathlib import Path
import sys
import time

output_dir = Path(sys.argv[1])
counter = Path(sys.argv[2])
mode = sys.argv[3]
artifact_name = sys.argv[4]

count = int(counter.read_text()) if counter.exists() else 0
count += 1
counter.write_text(str(count))

output_dir.mkdir(parents=True, exist_ok=True)

if mode == "sleep":
    time.sleep(0.35)

if mode == "symlink":
    target = output_dir / "ascend_protoc"
    target.write_text("#!/bin/sh\\necho fake-protoc\\n", encoding="utf-8")
    target.chmod(0o755)

    link = output_dir / "protoc"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to("ascend_protoc")
else:
    artifact = output_dir / artifact_name
    artifact.parent.mkdir(parents=True, exist_ok=True)
    content = "fixed-artifact" if mode == "fixed" else f"artifact-{count}"
    artifact.write_text(content, encoding="utf-8")
''',
        encoding="utf-8",
    )
    return builder


def _run_cache(
    *,
    cache_root: Path,
    prepared_inputs: list[Path],
    output_dir: Path,
    builder: Path,
    counter: Path,
    recipe_values: list[str] | None = None,
    environment_values: list[str] | None = None,
    domain: str = "custom_operator",
    operator_source: Path | None = None,
    artifact_includes: list[str] | None = None,
    builder_mode: str = "counted",
    artifact_name: str = "kernel.o",
    action: str = "TestOperator-0",
    stage_dir: Path | None = None,
    publish_state_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    actual_output = output_dir
    if domain == "custom_operator":
        safe_action = re.sub(r"[^A-Za-z0-9_.-]", "_", action)
        stage_dir = stage_dir or (
            output_dir.parent / "private-stages" / safe_action
        )
        publish_state_dir = publish_state_dir or (
            output_dir.parent / "publish-state"
        )
        actual_output = stage_dir

    command = [
        sys.executable,
        str(ENGINE),
        "run",
        "--cache-root",
        str(cache_root),
        "--domain",
        domain,
        "--unit",
        "test_unit",
        "--output-dir",
        str(actual_output),
        "--environment-profile",
        "ascendc" if domain == "custom_operator" else "host-cxx",
        "--environment-tool",
        sys.executable,
    ]

    for path in prepared_inputs:
        command.extend(["--prepared-input", str(path)])

    for value in recipe_values or ["recipe=stable"]:
        command.extend(["--recipe-value", value])

    for value in environment_values or ["abi=test"]:
        command.extend(["--environment-value", value])

    for pattern in artifact_includes or []:
        command.extend(["--artifact-include", pattern])

    if domain == "custom_operator":
        if operator_source is None:
            operator_source = prepared_inputs[0]
        assert stage_dir is not None
        assert publish_state_dir is not None
        command.extend(
            [
                "--soc",
                "ascend910b",
                "--operator",
                "test_operator",
                "--action",
                action,
                "--operator-source",
                str(operator_source),
                "--publish-dir",
                str(output_dir),
                "--publish-state-dir",
                str(publish_state_dir),
            ]
        )

    command.extend(
        [
            "--",
            sys.executable,
            str(builder),
            str(actual_output),
            str(counter),
            builder_mode,
            artifact_name,
        ]
    )

    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

def _assert_success(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, (
        f"returncode={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )


def _extract_key(proc: subprocess.CompletedProcess[str]) -> str:
    matches = _KEY_RE.findall(proc.stdout)
    assert matches, f"no cache key in stdout:\n{proc.stdout}"
    assert len(set(matches)) == 1, f"multiple cache keys in stdout: {matches}"
    return matches[-1]


def _find_entries(cache_root: Path, domain: str, final_key: str) -> list[Path]:
    # Deliberately locate entries from manifest contents rather than assuming
    # the cache directory layout. This is what the old T6 integration check got wrong.
    domain_root = cache_root / domain
    entries: list[Path] = []
    if not domain_root.exists():
        return entries

    for manifest_path in domain_root.rglob("manifest.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("domain") == domain and payload.get("final_key") == final_key:
            entries.append(manifest_path.parent)

    return sorted(entries)


def _only_entry(cache_root: Path, domain: str, final_key: str) -> Path:
    entries = _find_entries(cache_root, domain, final_key)
    assert len(entries) == 1, (
        f"expected exactly one entry for domain={domain} key={final_key}, "
        f"got {len(entries)}: {entries}"
    )
    return entries[0]


def _manifest(entry: Path) -> dict:
    return json.loads((entry / "manifest.json").read_text(encoding="utf-8"))


def _artifact_kind(artifact: dict) -> str:
    return artifact.get("kind", "file")


def _first_file_artifact(entry: Path) -> tuple[dict, Path]:
    manifest = _manifest(entry)
    files = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if _artifact_kind(artifact) == "file"
    ]
    assert files, f"entry has no regular-file artifacts: {entry}"
    artifact = files[0]
    return artifact, entry / "artifacts" / artifact["path"]


def _fresh_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _make_operator_inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.cpp").write_text("int source = 1;\n", encoding="utf-8")

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "kernel.cpp").write_text("int prepared = 1;\n", encoding="utf-8")
    return source, prepared


def test_custom_operator_miss_then_hit_and_restore(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(first)
    assert "[build-cache] MISS" in first.stdout
    assert "[build-cache] SAVED" in first.stdout
    assert counter.read_text() == "1"

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(second)
    assert "[build-cache] HIT" in second.stdout
    assert counter.read_text() == "1"
    assert (output / "kernel.o").read_text(encoding="utf-8") == "artifact-1"


def test_prepared_input_change_invalidates_and_revert_hits_history(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    original = (prepared / "kernel.cpp").read_text(encoding="utf-8")

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(first)
    key_original = _extract_key(first)

    (prepared / "kernel.cpp").write_text("int prepared = 2;\n", encoding="utf-8")
    _fresh_dir(output)

    changed = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(changed)
    key_changed = _extract_key(changed)
    assert "[build-cache] MISS" in changed.stdout
    assert key_changed != key_original
    assert counter.read_text() == "2"

    (prepared / "kernel.cpp").write_text(original, encoding="utf-8")
    _fresh_dir(output)

    reverted = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(reverted)
    assert "[build-cache] HIT" in reverted.stdout
    assert _extract_key(reverted) == key_original
    assert counter.read_text() == "2"


def test_recipe_normalizes_ephemeral_cmake_path_without_hiding_semantic_changes(
    tmp_path: Path,
):
    spec = importlib.util.spec_from_file_location(
        "build_cache_engine_recipe_path_test",
        ENGINE,
    )
    assert spec is not None and spec.loader is not None
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    source = tmp_path / "source"
    source.mkdir()
    binary_a = tmp_path / "build-a"
    binary_b = tmp_path / "build-b"
    binary_a.mkdir()
    binary_b.mkdir()

    tool_a = tmp_path / "pep517-a" / "bin" / "cmake"
    tool_b = tmp_path / "pep517-b" / "bin" / "cmake"

    def write_fake_cmake(path: Path, version: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#!/bin/sh\n"
            f'echo "cmake version {version}"\n',
            encoding="utf-8",
        )
        path.chmod(0o755)

    write_fake_cmake(tool_a, "4.4.3")
    write_fake_cmake(tool_b, "4.4.3")

    env_hash_a, _ = engine._hash_compiler_environment(
        "host-cxx",
        [],
        [],
        [str(tool_a)],
    )
    env_hash_b, _ = engine._hash_compiler_environment(
        "host-cxx",
        [],
        [],
        [str(tool_b)],
    )
    assert env_hash_a == env_hash_b

    recipe_values = ["protobuf_BUILD_TESTS=OFF"]
    command_a = [str(tool_a), "--build", "."]
    command_b = [str(tool_b), "--build", "."]

    raw_hash_a, _ = engine._hash_recipe(
        [],
        recipe_values,
        command_a,
        [source, binary_a],
    )
    raw_hash_b, _ = engine._hash_recipe(
        [],
        recipe_values,
        command_b,
        [source, binary_b],
    )
    assert raw_hash_a != raw_hash_b

    stable_hash_a, manifest_a = engine._hash_recipe(
        [],
        recipe_values,
        command_a,
        [source, binary_a, tool_a],
    )
    stable_hash_b, manifest_b = engine._hash_recipe(
        [],
        recipe_values,
        command_b,
        [source, binary_b, tool_b],
    )
    assert stable_hash_a == stable_hash_b
    assert manifest_a[-1]["argv"] == ["<PATH_2>", "--build", "."]
    assert manifest_b[-1]["argv"] == ["<PATH_2>", "--build", "."]

    write_fake_cmake(tool_b, "4.4.4")
    changed_env_hash, _ = engine._hash_compiler_environment(
        "host-cxx",
        [],
        [],
        [str(tool_b)],
    )
    assert changed_env_hash != env_hash_a

    changed_recipe_hash, _ = engine._hash_recipe(
        [],
        ["protobuf_BUILD_TESTS=ON"],
        command_b,
        [source, binary_b, tool_b],
    )
    assert changed_recipe_hash != stable_hash_a


def test_recipe_change_invalidates_cache(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_values=["optimization=-O2"],
    )
    _assert_success(first)
    key_a = _extract_key(first)

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_values=["optimization=-O0"],
    )
    _assert_success(second)
    assert "[build-cache] MISS" in second.stdout
    assert _extract_key(second) != key_a
    assert counter.read_text() == "2"


def test_compiler_environment_change_invalidates_cache(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        environment_values=["toolkit=9.1.0"],
    )
    _assert_success(first)
    key_a = _extract_key(first)

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        environment_values=["toolkit=9.2.0"],
    )
    _assert_success(second)
    assert "[build-cache] MISS" in second.stdout
    assert _extract_key(second) != key_a
    assert counter.read_text() == "2"


def test_operator_text_hash_is_identity_namespace_not_action_key(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(first)
    key = _extract_key(first)
    assert len(_find_entries(cache_root, "custom_operator", key)) == 1

    (source / "kernel.cpp").write_text("int source = 2;\n", encoding="utf-8")
    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(second)
    assert "[build-cache] MISS" in second.stdout
    assert _extract_key(second) == key
    assert counter.read_text() == "2"
    assert len(_find_entries(cache_root, "custom_operator", key)) == 2


def test_active_corrupted_artifact_rebuilds_repairs_then_hits(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(first)
    key = _extract_key(first)
    entry = _only_entry(cache_root, "custom_operator", key)

    artifact_meta, cached_artifact = _first_file_artifact(entry)
    expected_sha = artifact_meta["sha256"]
    assert _sha256_file(cached_artifact) == expected_sha

    cached_artifact.write_bytes(cached_artifact.read_bytes() + b"\nCORRUPTED\n")
    assert _sha256_file(cached_artifact) != expected_sha

    _fresh_dir(output)

    rebuilt = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(rebuilt)
    assert _extract_key(rebuilt) == key
    assert "[build-cache] MISS" in rebuilt.stdout
    assert "[build-cache] SAVED" in rebuilt.stdout
    assert counter.read_text() == "2"

    repaired_entry = _only_entry(cache_root, "custom_operator", key)
    repaired_meta, repaired_artifact = _first_file_artifact(repaired_entry)
    assert _sha256_file(repaired_artifact) == repaired_meta["sha256"]

    _fresh_dir(output)

    warm = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(warm)
    assert "[build-cache] HIT" in warm.stdout
    assert _extract_key(warm) == key
    assert counter.read_text() == "2"


def test_identical_rebuild_output_is_still_cacheable(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_values=["recipe=a"],
        builder_mode="fixed",
    )
    _assert_success(first)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
        recipe_values=["recipe=b"],
        builder_mode="fixed",
    )
    _assert_success(second)
    assert "[build-cache] MISS" in second.stdout
    assert "[build-cache] SAVED" in second.stdout
    assert counter.read_text() == "2"


def test_third_party_whole_unit_restores_regular_product(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "third_party.cc").write_text("source\n", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["*.a"],
        artifact_name="libtest.a",
    )
    _assert_success(first)
    assert "[build-cache] MISS" in first.stdout
    assert counter.read_text() == "1"

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["*.a"],
        artifact_name="libtest.a",
    )
    _assert_success(second)
    assert "[build-cache] HIT" in second.stdout
    assert counter.read_text() == "1"
    assert (output / "libtest.a").read_text(encoding="utf-8") == "artifact-1"


def test_third_party_symlink_restores_link_and_internal_target(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "protobuf.cc").write_text("source\n", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["protoc"],
        builder_mode="symlink",
    )
    _assert_success(first)
    assert "[build-cache] MISS" in first.stdout

    key = _extract_key(first)
    entry = _only_entry(cache_root, "third_party", key)
    manifest = _manifest(entry)
    artifacts = {artifact["path"]: artifact for artifact in manifest["artifacts"]}

    assert artifacts["protoc"]["kind"] == "symlink"
    assert artifacts["protoc"]["target"] == "ascend_protoc"
    assert artifacts["ascend_protoc"]["kind"] == "file"

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["protoc"],
        builder_mode="symlink",
    )
    _assert_success(second)
    assert "[build-cache] HIT" in second.stdout
    assert counter.read_text() == "1"

    protoc = output / "protoc"
    target = output / "ascend_protoc"
    assert protoc.is_symlink()
    assert os.readlink(protoc) == "ascend_protoc"
    assert target.is_file()
    assert os.access(protoc, os.X_OK)


def test_corrupted_cached_symlink_rebuilds(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "protobuf.cc").write_text("source\n", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["protoc"],
        builder_mode="symlink",
    )
    _assert_success(first)

    key = _extract_key(first)
    entry = _only_entry(cache_root, "third_party", key)
    cached_link = entry / "artifacts" / "protoc"
    assert cached_link.is_symlink()
    cached_link.unlink()
    cached_link.symlink_to("wrong_target")

    _fresh_dir(output)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["protoc"],
        builder_mode="symlink",
    )
    _assert_success(second)
    assert "[build-cache] MISS" in second.stdout
    assert "[build-cache] SAVED" in second.stdout
    assert _extract_key(second) == key
    assert counter.read_text() == "2"


def _spawn_cache(**kwargs) -> subprocess.Popen[str]:
    # Build the exact command through a lightweight recorder by duplicating the
    # helper's argument assembly. The returned process lets tests overlap two
    # real cache-engine invocations.
    cache_root = kwargs["cache_root"]
    prepared_inputs = kwargs["prepared_inputs"]
    output_dir = kwargs["output_dir"]
    builder = kwargs["builder"]
    counter = kwargs["counter"]
    operator_source = kwargs.get("operator_source") or prepared_inputs[0]
    action = kwargs.get("action", "TestOperator-0")
    artifact_name = kwargs.get("artifact_name", "kernel.o")
    builder_mode = kwargs.get("builder_mode", "sleep")
    recipe_values = kwargs.get("recipe_values") or ["recipe=stable"]
    publish_state_dir = kwargs.get("publish_state_dir") or (
        output_dir.parent / "publish-state"
    )
    safe_action = re.sub(r"[^A-Za-z0-9_.-]", "_", action)
    stage_dir = kwargs.get("stage_dir") or (
        output_dir.parent / "private-stages" / safe_action
    )

    command = [
        sys.executable,
        str(ENGINE),
        "run",
        "--cache-root",
        str(cache_root),
        "--domain",
        "custom_operator",
        "--unit",
        "test_unit",
        "--output-dir",
        str(stage_dir),
        "--publish-dir",
        str(output_dir),
        "--publish-state-dir",
        str(publish_state_dir),
        "--prepared-input",
        str(prepared_inputs[0]),
        "--recipe-value",
        recipe_values[0],
        "--environment-value",
        "abi=test",
        "--environment-profile",
        "ascendc",
        "--environment-tool",
        sys.executable,
        "--soc",
        "ascend910b",
        "--operator",
        "test_operator",
        "--action",
        action,
        "--operator-source",
        str(operator_source),
        "--",
        sys.executable,
        str(builder),
        str(stage_dir),
        str(counter),
        builder_mode,
        artifact_name,
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish_process(proc: subprocess.Popen[str]) -> tuple[str, str]:
    stdout, stderr = proc.communicate(timeout=10)
    assert proc.returncode == 0, (
        f"returncode={proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    return stdout, stderr


def test_parallel_actions_have_exact_artifact_ownership(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "shared-output"
    output.mkdir()
    state = tmp_path / "publish-state"
    cache_root = tmp_path / "cache"
    builder = _write_builder(tmp_path)

    first = _spawn_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        publish_state_dir=state,
        builder=builder,
        counter=tmp_path / "counter-a",
        action="TestOperator-0",
        artifact_name="A.o",
        recipe_values=["recipe=A"],
    )
    second = _spawn_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        publish_state_dir=state,
        builder=builder,
        counter=tmp_path / "counter-b",
        action="TestOperator-1",
        artifact_name="B.o",
        recipe_values=["recipe=B"],
    )
    out_a, _ = _finish_process(first)
    out_b, _ = _finish_process(second)
    assert "[build-cache] SAVED" in out_a
    assert "[build-cache] SAVED" in out_b
    assert (output / "A.o").is_file()
    assert (output / "B.o").is_file()

    manifests = []
    for manifest_path in (cache_root / "custom_operator").rglob("manifest.json"):
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    assert len(manifests) == 2
    by_action = {manifest["action"]: manifest for manifest in manifests}
    assert {item["path"] for item in by_action["TestOperator-0"]["artifacts"]} == {"A.o"}
    assert {item["path"] for item in by_action["TestOperator-1"]["artifacts"]} == {"B.o"}
    assert all(manifest["artifact_model"] == 2 for manifest in manifests)


def test_parallel_cache_hits_publish_without_shared_output_race(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "shared-output"
    output.mkdir()
    state = tmp_path / "publish-state"
    cache_root = tmp_path / "cache"
    builder = _write_builder(tmp_path)

    for action, artifact, recipe, counter_name in [
        ("TestOperator-0", "A.o", "recipe=A", "counter-a"),
        ("TestOperator-1", "B.o", "recipe=B", "counter-b"),
    ]:
        proc = _run_cache(
            cache_root=cache_root,
            prepared_inputs=[prepared],
            operator_source=source,
            output_dir=output,
            publish_state_dir=state,
            builder=builder,
            counter=tmp_path / counter_name,
            action=action,
            artifact_name=artifact,
            recipe_values=[recipe],
        )
        _assert_success(proc)

    _fresh_dir(output)
    shutil.rmtree(tmp_path / "private-stages", ignore_errors=True)

    first = _spawn_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        publish_state_dir=state,
        builder=builder,
        counter=tmp_path / "counter-a",
        action="TestOperator-0",
        artifact_name="A.o",
        recipe_values=["recipe=A"],
        builder_mode="counted",
    )
    second = _spawn_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        publish_state_dir=state,
        builder=builder,
        counter=tmp_path / "counter-b",
        action="TestOperator-1",
        artifact_name="B.o",
        recipe_values=["recipe=B"],
        builder_mode="counted",
    )
    out_a, _ = _finish_process(first)
    out_b, _ = _finish_process(second)
    assert "[build-cache] HIT" in out_a
    assert "[build-cache] HIT" in out_b
    assert (output / "A.o").is_file()
    assert (output / "B.o").is_file()
    assert (tmp_path / "counter-a").read_text() == "1"
    assert (tmp_path / "counter-b").read_text() == "1"


def test_same_key_parallel_requests_compile_once(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "shared-output"
    output.mkdir()
    state = tmp_path / "publish-state"
    cache_root = tmp_path / "cache"
    builder = _write_builder(tmp_path)
    counter = tmp_path / "counter"

    first = _spawn_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        publish_state_dir=state,
        builder=builder,
        counter=counter,
        action="TestOperator-0",
        artifact_name="A.o",
        recipe_values=["recipe=A"],
    )
    time.sleep(0.05)
    second = _spawn_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        publish_state_dir=state,
        builder=builder,
        counter=counter,
        action="TestOperator-0",
        artifact_name="A.o",
        recipe_values=["recipe=A"],
    )
    out_a, _ = _finish_process(first)
    out_b, _ = _finish_process(second)
    combined = out_a + out_b
    assert combined.count("[build-cache] MISS") == 1
    assert combined.count("[build-cache] HIT") == 1
    assert counter.read_text() == "1"


def test_same_action_removes_stale_published_artifact(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "shared-output"
    output.mkdir()
    state = tmp_path / "publish-state"
    cache_root = tmp_path / "cache"
    builder = _write_builder(tmp_path)
    counter = tmp_path / "counter"

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        publish_state_dir=state,
        builder=builder,
        counter=counter,
        action="TestOperator-0",
        artifact_name="old.o",
        recipe_values=["recipe=old"],
    )
    _assert_success(first)
    assert (output / "old.o").is_file()

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        publish_state_dir=state,
        builder=builder,
        counter=counter,
        action="TestOperator-0",
        artifact_name="new.o",
        recipe_values=["recipe=new"],
    )
    _assert_success(second)
    assert not (output / "old.o").exists()
    assert (output / "new.o").is_file()


def test_legacy_custom_operator_model_is_rebuilt(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(first)
    key = _extract_key(first)
    entry = _only_entry(cache_root, "custom_operator", key)
    manifest_path = entry / "manifest.json"
    manifest = _manifest(entry)
    manifest.pop("artifact_model")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _fresh_dir(output)
    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(second)
    assert "[build-cache] MISS" in second.stdout
    assert counter.read_text() == "2"


def test_schema3_third_party_without_artifact_model_still_hits(tmp_path: Path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "third_party.cc").write_text("source\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    cache_root = tmp_path / "cache"
    counter = tmp_path / "counter"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["*.a"],
        artifact_name="libtest.a",
    )
    _assert_success(first)
    key = _extract_key(first)
    entry = _only_entry(cache_root, "third_party", key)
    manifest_path = entry / "manifest.json"
    manifest = _manifest(entry)
    manifest.pop("artifact_model")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _fresh_dir(output)
    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        output_dir=output,
        builder=builder,
        counter=counter,
        domain="third_party",
        artifact_includes=["*.a"],
        artifact_name="libtest.a",
    )
    _assert_success(second)
    assert "[build-cache] HIT" in second.stdout
    assert counter.read_text() == "1"



def test_cache_lock_failure_is_fail_open_for_custom_operator(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    builder = _write_builder(tmp_path)
    counter = tmp_path / "counter"

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block cache mkdir", encoding="utf-8")
    unusable_cache = blocker / "cache"

    proc = _run_cache(
        cache_root=unusable_cache,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        builder=builder,
        counter=counter,
    )
    _assert_success(proc)
    assert "[build-cache] BYPASS" in proc.stdout
    assert (output / "kernel.o").is_file()
    assert counter.read_text() == "1"


def test_different_actions_cannot_publish_same_relative_path(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "shared-output"
    output.mkdir()
    state = tmp_path / "publish-state"
    cache_root = tmp_path / "cache"
    builder = _write_builder(tmp_path)

    first = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        publish_state_dir=state,
        builder=builder,
        counter=tmp_path / "counter-a",
        action="TestOperator-0",
        artifact_name="same.o",
        recipe_values=["recipe=A"],
    )
    _assert_success(first)

    second = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        publish_state_dir=state,
        builder=builder,
        counter=tmp_path / "counter-b",
        action="TestOperator-1",
        artifact_name="same.o",
        recipe_values=["recipe=B"],
    )
    assert second.returncode != 0
    assert "artifact ownership collision" in second.stderr


def test_first_isolated_publish_cleans_legacy_shared_output(tmp_path: Path):
    source, prepared = _make_operator_inputs(tmp_path)
    output = tmp_path / "shared-output"
    output.mkdir()
    (output / "legacy-stale.o").write_text("stale", encoding="utf-8")
    cache_root = tmp_path / "cache"
    builder = _write_builder(tmp_path)

    proc = _run_cache(
        cache_root=cache_root,
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=output,
        publish_state_dir=tmp_path / "publish-state",
        builder=builder,
        counter=tmp_path / "counter",
        artifact_name="current.o",
    )
    _assert_success(proc)
    assert not (output / "legacy-stale.o").exists()
    assert (output / "current.o").is_file()
