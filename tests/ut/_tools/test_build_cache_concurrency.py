# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE = REPO_ROOT / "csrc" / "scripts" / "build_cache.py"


def _builder(tmp_path: Path) -> Path:
    path = tmp_path / "builder.py"
    path.write_text(
        """from pathlib import Path
import sys

output = Path(sys.argv[1])
counter = Path(sys.argv[2])
artifact = sys.argv[3]
output.mkdir(parents=True, exist_ok=True)
count = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(count + 1))
(output / artifact).write_text("artifact")
""",
        encoding="utf-8",
    )
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.cpp").write_text("int x = 1;\n", encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "kernel.cpp").write_text("int y = 1;\n", encoding="utf-8")
    return source, prepared


def _command(
    *,
    cache: Path,
    source: Path,
    prepared: Path,
    publish: Path,
    state: Path,
    stage: Path,
    builder: Path,
    counter: Path,
    action: str,
    artifact: str,
    recipe: str,
) -> list[str]:
    return [
        sys.executable,
        str(ENGINE),
        "run",
        "--cache-root",
        str(cache),
        "--domain",
        "custom_operator",
        "--unit",
        "test_unit",
        "--output-dir",
        str(stage),
        "--publish-dir",
        str(publish),
        "--publish-state-dir",
        str(state),
        "--prepared-input",
        str(prepared),
        "--recipe-value",
        recipe,
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
        str(source),
        "--",
        sys.executable,
        str(builder),
        str(stage),
        str(counter),
        artifact,
    ]


def _run(command: list[str], env: dict[str, str] | None = None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=merged,
        check=False,
    )


def _seed_one(tmp_path: Path):
    source, prepared = _inputs(tmp_path)
    builder = _builder(tmp_path)
    cache = tmp_path / "cache"
    publish = tmp_path / "publish"
    state = tmp_path / "state"
    stage = tmp_path / "stage"
    counter = tmp_path / "counter"
    command = _command(
        cache=cache,
        source=source,
        prepared=prepared,
        publish=publish,
        state=state,
        stage=stage,
        builder=builder,
        counter=counter,
        action="TestOperator-0",
        artifact="A.o",
        recipe="recipe=A",
    )
    first = _run(command)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "[build-cache] MISS" in first.stdout
    assert counter.read_text() == "1"
    return source, prepared, builder, cache, publish, state, stage, counter, command


def _hold_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    return stream


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_index_lock_is_nonblocking_on_hit(tmp_path: Path):
    _, _, _, cache, _, _, _, counter, command = _seed_one(tmp_path)
    event_log = tmp_path / "events.jsonl"
    lock = _hold_lock(cache / "custom_operator" / "cache_index.json.lock")
    try:
        start = time.monotonic()
        hit = _run(
            command,
            {"VLLM_ASCEND_BUILD_CACHE_EVENT_LOG": str(event_log)},
        )
        elapsed = time.monotonic() - start
    finally:
        lock.close()

    assert hit.returncode == 0, hit.stdout + hit.stderr
    assert "[build-cache] HIT" in hit.stdout
    assert elapsed < 2.0
    assert counter.read_text() == "1"
    assert any(e["event"] == "index_skipped" for e in _read_events(event_log))


def test_entry_lock_timeout_bypasses_cache(tmp_path: Path):
    _, _, _, cache, _, _, _, counter, command = _seed_one(tmp_path)
    manifest = next((cache / "custom_operator").rglob("manifest.json"))
    entry = manifest.parent
    lock_name = hashlib.sha256(str(entry).encode("utf-8")).hexdigest()
    lock_path = cache / "custom_operator" / ".locks" / f"{lock_name}.lock"
    event_log = tmp_path / "events.jsonl"
    lock = _hold_lock(lock_path)
    try:
        start = time.monotonic()
        bypass = _run(
            command,
            {
                "VLLM_ASCEND_BUILD_CACHE_EVENT_LOG": str(event_log),
                "VLLM_ASCEND_BUILD_CACHE_ENTRY_LOCK_TIMEOUT_SECONDS": "0.2",
            },
        )
        elapsed = time.monotonic() - start
    finally:
        lock.close()

    assert bypass.returncode == 0, bypass.stdout + bypass.stderr
    assert "[build-cache] BYPASS" in bypass.stdout
    assert elapsed < 2.0
    assert counter.read_text() == "2"
    events = _read_events(event_log)
    assert any(e["event"] == "lock_timeout" and e["lock"] == "entry" for e in events)
    assert any(e["event"] == "cache_result" and e["status"] == "BYPASS" for e in events)


def test_publish_lock_timeout_fails_fast(tmp_path: Path):
    _, _, _, _, _, state, _, counter, command = _seed_one(tmp_path)
    event_log = tmp_path / "events.jsonl"
    lock = _hold_lock(state / ".publish.lock")
    try:
        start = time.monotonic()
        failed = _run(
            command,
            {
                "VLLM_ASCEND_BUILD_CACHE_EVENT_LOG": str(event_log),
                "VLLM_ASCEND_BUILD_CACHE_PUBLISH_LOCK_TIMEOUT_SECONDS": "0.2",
            },
        )
        elapsed = time.monotonic() - start
    finally:
        lock.close()

    assert failed.returncode == 75, failed.stdout + failed.stderr
    assert "[build-cache] ERROR lock timeout kind=publish" in failed.stdout
    assert elapsed < 2.0
    assert counter.read_text() == "1"


def test_action_lock_timeout_fails_fast(tmp_path: Path):
    _, _, _, _, _, state, _, counter, command = _seed_one(tmp_path)
    action_identity = "test_unit/TestOperator-0"
    action_hash = hashlib.sha256(action_identity.encode("utf-8")).hexdigest()
    action_lock = state / ".action_locks" / f"{action_hash}.lock"
    event_log = tmp_path / "events.jsonl"
    lock = _hold_lock(action_lock)
    try:
        start = time.monotonic()
        failed = _run(
            command,
            {
                "VLLM_ASCEND_BUILD_CACHE_EVENT_LOG": str(event_log),
                "VLLM_ASCEND_BUILD_CACHE_ACTION_LOCK_TIMEOUT_SECONDS": "0.2",
            },
        )
        elapsed = time.monotonic() - start
    finally:
        lock.close()

    assert failed.returncode == 75, failed.stdout + failed.stderr
    assert "[build-cache] ERROR lock timeout kind=action" in failed.stdout
    assert elapsed < 2.0
    assert counter.read_text() == "1"


def test_many_parallel_hits_complete_without_hang(tmp_path: Path):
    source, prepared = _inputs(tmp_path)
    builder = _builder(tmp_path)
    cache = tmp_path / "cache"
    publish = tmp_path / "publish"
    state = tmp_path / "state"
    commands: list[tuple[list[str], Path]] = []

    action_count = 32
    for index in range(action_count):
        counter = tmp_path / f"counter-{index}"
        command = _command(
            cache=cache,
            source=source,
            prepared=prepared,
            publish=publish,
            state=state,
            stage=tmp_path / "seed-stages" / str(index),
            builder=builder,
            counter=counter,
            action=f"TestOperator-{index}",
            artifact=f"{index}.o",
            recipe=f"recipe={index}",
        )
        seeded = _run(command)
        assert seeded.returncode == 0, seeded.stdout + seeded.stderr
        commands.append((command, counter))

    # Recreate the disposable build-tree side of publication state. Only the
    # persistent content-addressed cache survives, matching a fresh CI runner.
    #
    # Keep the same logical stage paths for the HIT wave. The stage path is part
    # of the wrapped command and therefore participates in recipe_hash unless it
    # is normalized. The real CMake integration recreates the same relative
    # action-stage path in a fresh build tree, so changing "seed-stages" to
    # "hit-stages" here would incorrectly manufacture a cache MISS.
    import shutil
    shutil.rmtree(publish, ignore_errors=True)
    shutil.rmtree(state, ignore_errors=True)
    shutil.rmtree(tmp_path / "seed-stages", ignore_errors=True)

    env = os.environ.copy()
    env.update(
        {
            "VLLM_ASCEND_BUILD_CACHE_ACTION_LOCK_TIMEOUT_SECONDS": "5",
            "VLLM_ASCEND_BUILD_CACHE_PUBLISH_LOCK_TIMEOUT_SECONDS": "5",
        }
    )

    processes = []
    for command, _ in commands:
        processes.append(
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        )

    outputs = []
    for proc in processes:
        stdout, stderr = proc.communicate(timeout=20)
        outputs.append((proc.returncode, stdout, stderr))

    assert all(code == 0 for code, _, _ in outputs), outputs
    assert all("[build-cache] HIT" in stdout for _, stdout, _ in outputs)
    assert all(counter.read_text() == "1" for _, counter in commands)
    assert len(list(publish.glob("*.o"))) == action_count
