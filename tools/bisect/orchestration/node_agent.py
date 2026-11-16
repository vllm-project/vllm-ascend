# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""One node-side request, using the controller's frozen tool snapshot."""

import contextlib
import hashlib
import json
import os
import shutil
import socket
import sys
from pathlib import Path

from tools.bisect.deployment import git_ops
from tools.bisect.deployment.manager import BuildManager
from tools.bisect.orchestration.config import BisectOptions
from tools.bisect.testing.definition import TestEntry
from tools.bisect.testing.executor import evaluate_test, execute_test


def _write_state(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".pending-write")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(data, stream)
    temporary.replace(path)


@contextlib.contextmanager
def _claim(path: Path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+b") as stream:
        if os.name == "posix":
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        else:
            import msvcrt

            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def handle(request: dict) -> dict:
    try:
        session = Path.home() / ".cache/nightly_bisect/execution" / request["session_key"] / request["target_name"]
        session.mkdir(parents=True, exist_ok=True, mode=0o700)
        with _claim(session / "operation.lock"):
            return _handle(request)
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "action": request.get("action")}


def _handle(request: dict) -> dict:
    repository = Path(request["repository"])
    session = Path.home() / ".cache/nightly_bisect/execution" / request["session_key"] / request["target_name"]
    if not repository.is_absolute() or not session.is_absolute():
        raise ValueError("Node paths must be absolute")
    if session.is_symlink():
        raise RuntimeError("Session path cannot be a symlink")
    session.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = session / "environment.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    action = request["action"]
    if action == "probe":
        identity = [socket.gethostname(), str(repository.resolve())]
        boot_id = Path("/proc/sys/kernel/random/boot_id")
        namespace = Path("/proc/self/ns/mnt")
        if boot_id.is_file() and namespace.exists():
            identity.extend((boot_id.read_text().strip(), str(namespace.stat().st_ino)))
        return {
            "status": "ok",
            "head": git_ops.current_commit(repository),
            "python": sys.executable,
            "execution_identity": hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
        }
    test_marker = session / "active-test.pending"
    if test_marker.exists():
        raise RuntimeError(
            "A prior test did not finish; confirm its processes have stopped before clearing active-test.pending"
        )
    target = request["commit"]
    if action == "prepare":
        options = BisectOptions(
            repo_dir=repository,
            work_dir=str(session),
            soc=request["soc"],
            force_initial_build=request["force_initial_build"],
            assume_built_head=request["assume_built_head"],
            native_check=request["native_check"],
        )
        builder = BuildManager(options)
        if state is not None and not request["ignore_baseline"]:
            builder.restore_state(state)
        if request["channel"] == "ray":
            plan = builder.transition_plan(target)
            ray_prefix = Path(request["ray_runtime_prefix"]).resolve()
            if (
                plan is not None
                and plan.actions
                and (
                    Path(sys.prefix).resolve() == ray_prefix
                    or Path(shutil.which("python3") or sys.executable).resolve().is_relative_to(ray_prefix)
                )
            ):
                raise RuntimeError(
                    "Ray and target executor share a Python prefix; use an independent execution channel for mutations"
                )
            if plan is not None and any(item.kind.startswith("image-") for item in plan.actions):
                raise RuntimeError(
                    "Ray cannot control image/Python replacement; select an independent execution channel"
                )
        decision = builder.prepare(target, session / (request["trial"] + "-prepare.log"))
        _write_state(state_path, builder.export_state())
        return {"status": "ok", "commit": builder.installed_commit, "rebuilt": decision.rebuild}
    if action != "test" or state is None or state["installed_commit"] != target:
        raise RuntimeError("Test requested without a matching prepared environment")
    if git_ops.current_commit(repository) != target:
        raise RuntimeError("Target checkout changed after preparation")
    if (session / "base-transition-node0.pending").exists():
        raise RuntimeError("Target has an incomplete environment transition")
    data = request["test"]
    entry = TestEntry(
        tuple(data["command"]),
        data["working_directory"],
        tuple(tuple(item) for item in data["environment"]),
        data["timeout_seconds"],
    )
    test_marker.write_text(request["trial"], encoding="utf-8")
    outcome = execute_test(entry, repository, state["test_environment"], session / (request["trial"] + "-test.log"))
    test_marker.unlink()
    verdict, note = evaluate_test(outcome)
    return {"status": "ok", "verdict": verdict, "note": note, "exit_code": outcome.exit_code}
