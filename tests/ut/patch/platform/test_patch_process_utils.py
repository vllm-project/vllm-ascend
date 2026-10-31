# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call

import psutil
import pytest

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_process_utils.py"


@pytest.fixture
def isolated_patch(monkeypatch):
    """Load the CPU-only backport without unrelated NPU platform patches."""
    vllm = ModuleType("vllm")
    utils = ModuleType("vllm.utils")
    upstream = ModuleType("vllm.utils.system_utils")
    original = Mock()
    monkeypatch.setattr(upstream, "logger", Mock(), raising=False)
    monkeypatch.setattr(utils, "system_utils", upstream, raising=False)
    monkeypatch.setattr(vllm, "utils", utils, raising=False)
    consumers = [ModuleType(name) for name in ("vllm.v1.utils", "vllm.entrypoints.launchers.dp_supervisor")]
    for module in [vllm, utils, upstream, *consumers]:
        if module is not vllm:
            monkeypatch.setattr(module, "kill_process_tree", original, raising=False)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location("_test_process_utils_patch", PATCH_PATH)
    assert spec is not None and spec.loader is not None
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    return patch, upstream, consumers


@pytest.mark.parametrize("exit_during", ["name", "kill"])
def test_shutdown_continues_after_child_exits(isolated_patch, monkeypatch, exit_during):
    patch, _, _ = isolated_patch
    first, exited, last = [Mock(pid=pid) for pid in (101, 102, 103)]
    parent = Mock()
    parent.children.return_value = [first, exited, last]
    monkeypatch.setattr(patch.psutil, "Process", Mock(return_value=parent))
    if exit_during == "name":
        exited.name.side_effect = psutil.NoSuchProcess(exited.pid)

    def signal_process(pid, sig):
        if pid == exited.pid:
            raise ProcessLookupError

    kill = Mock(side_effect=signal_process)
    monkeypatch.setattr(patch.os, "kill", kill)
    patch._kill_process_tree(100)

    expected = [101, 103, 100] if exit_during == "name" else [101, 102, 103, 100]
    assert kill.call_args_list == [call(pid, signal.SIGKILL) for pid in expected]


def test_missing_parent_needs_no_signals(isolated_patch, monkeypatch):
    patch, _, _ = isolated_patch
    monkeypatch.setattr(patch.psutil, "Process", Mock(side_effect=psutil.NoSuchProcess(100)))
    kill = Mock()
    monkeypatch.setattr(patch.os, "kill", kill)
    patch._kill_process_tree(100)
    kill.assert_not_called()


def test_permission_error_is_not_hidden(isolated_patch, monkeypatch):
    patch, _, _ = isolated_patch
    parent = Mock()
    parent.children.return_value = [Mock(pid=101)]
    monkeypatch.setattr(patch.psutil, "Process", Mock(return_value=parent))
    monkeypatch.setattr(patch.os, "kill", Mock(side_effect=PermissionError))
    with pytest.raises(PermissionError):
        patch._kill_process_tree(100)


def test_preimported_shutdown_consumers_use_the_backport(isolated_patch, monkeypatch):
    patch, upstream, consumers = isolated_patch
    parent = Mock()
    parent.children.return_value = [Mock(pid=101)]
    parent.children.return_value[0].name.side_effect = psutil.NoSuchProcess(101)
    monkeypatch.setattr(patch.psutil, "Process", Mock(return_value=parent))
    kill = Mock()
    monkeypatch.setattr(patch.os, "kill", kill)

    for consumer in [upstream, *consumers]:
        consumer.kill_process_tree(100)
    assert kill.call_args_list == [call(100, signal.SIGKILL)] * 3


@pytest.mark.skipif(sys.platform != "linux", reason="Checks Linux SIGTERM/SIGKILL exit status.")
def test_reaped_child_does_not_leave_other_owned_processes_alive(isolated_patch, monkeypatch):
    patch, _, _ = isolated_patch
    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    target, *children = processes = [subprocess.Popen(sleep) for _ in range(4)]
    try:
        listed = [psutil.Process(child.pid) for child in children]
        children[1].terminate()
        children[1].wait(timeout=10)
        monkeypatch.setattr(psutil.Process, "children", lambda self, recursive=False: listed)

        patch._kill_process_tree(target.pid)

        assert children[1].returncode == -signal.SIGTERM
        for process in (children[0], children[2], target):
            assert process.wait(timeout=10) == -signal.SIGKILL
    finally:
        for process in processes:
            process.kill()
            process.wait(timeout=10)
