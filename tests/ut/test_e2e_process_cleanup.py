# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest
import vllm.v1.utils as vllm_process_utils

from tests.e2e.process_cleanup import shutdown_engine


@pytest.fixture
def process_tree(monkeypatch):
    parent = Mock(pid=100)
    exited_child = Mock(pid=101)
    surviving_child = Mock(pid=102)
    parent.children.return_value = [exited_child, surviving_child]
    exited_child.name.side_effect = psutil.NoSuchProcess(exited_child.pid)
    exited_child.kill.side_effect = psutil.NoSuchProcess(exited_child.pid)
    process_factory = Mock(return_value=parent)
    monkeypatch.setattr(psutil, "Process", process_factory)
    monkeypatch.setattr(
        "vllm.utils.system_utils.os.kill",
        Mock(side_effect=AssertionError("Tests must not signal real processes")),
    )
    return SimpleNamespace(parent=parent, exited=exited_child, surviving=surviving_child, factory=process_factory)


def test_shutdown_continues_after_a_child_exits(process_tree):
    parent, surviving_child = process_tree.parent, process_tree.surviving
    process = Mock(pid=parent.pid)
    process.name = "EngineCore"
    process.is_alive.return_value = True
    engine_core = SimpleNamespace(shutdown=lambda: vllm_process_utils.shutdown([process], timeout=0))
    original_kill = vllm_process_utils.kill_process_tree

    shutdown_engine(engine_core)

    surviving_child.kill.assert_called_once_with()
    parent.kill.assert_called_once_with()
    process.terminate.assert_called_once_with()
    assert vllm_process_utils.kill_process_tree is original_kill


@pytest.mark.parametrize("stage", ["lookup", "children", "parent_kill"])
def test_shutdown_tolerates_a_parent_that_already_exited(process_tree, stage):
    parent = process_tree.parent
    operation = {
        "lookup": process_tree.factory,
        "children": parent.children,
        "parent_kill": parent.kill,
    }[stage]
    operation.side_effect = psutil.NoSuchProcess(parent.pid)
    engine_core = SimpleNamespace(shutdown=lambda: vllm_process_utils.kill_process_tree(parent.pid))

    shutdown_engine(engine_core)

    if stage == "parent_kill":
        process_tree.surviving.kill.assert_called_once_with()
    else:
        process_tree.surviving.kill.assert_not_called()


@pytest.mark.parametrize("stage", ["lookup", "children", "child_kill", "parent_kill"])
def test_shutdown_propagates_cleanup_permission_errors(process_tree, stage):
    parent = process_tree.parent
    operation = {
        "lookup": process_tree.factory,
        "children": parent.children,
        "child_kill": process_tree.surviving.kill,
        "parent_kill": parent.kill,
    }[stage]
    operation.side_effect = psutil.AccessDenied(parent.pid)
    engine_core = SimpleNamespace(shutdown=lambda: vllm_process_utils.kill_process_tree(parent.pid))
    original_kill = vllm_process_utils.kill_process_tree

    with pytest.raises(psutil.AccessDenied):
        shutdown_engine(engine_core)

    assert vllm_process_utils.kill_process_tree is original_kill


@pytest.mark.parametrize("error", [RuntimeError("engine shutdown failed"), psutil.NoSuchProcess(100)])
def test_shutdown_propagates_other_engine_errors(error):
    engine_core = SimpleNamespace(shutdown=Mock(side_effect=error))
    original_kill = vllm_process_utils.kill_process_tree

    with pytest.raises(type(error)) as caught:
        shutdown_engine(engine_core)

    assert caught.value is error
    assert vllm_process_utils.kill_process_tree is original_kill
