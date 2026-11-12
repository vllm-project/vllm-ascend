# SPDX-License-Identifier: Apache-2.0
"""A missing consumer must not cause the owner to release its backing."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


def load_module(monkeypatch):
    if importlib.util.find_spec("vllm") is None:
        vllm = types.ModuleType("vllm")
        vllm.__path__ = []
        logger = types.ModuleType("vllm.logger")
        logger.logger = Mock()
        monkeypatch.setitem(sys.modules, "vllm", vllm)
        monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram"
    package = types.ModuleType("local_vmm_test_package")
    package.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in ("shared_host", "local_shared_vmm"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", path / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module


def test_owner_is_retained_if_consumer_does_not_reach_barrier(monkeypatch):
    module = load_module(monkeypatch)
    obj = module.LocalSharedVmmBuffer.__new__(module.LocalSharedVmmBuffer)
    obj.owner = True
    obj.group = types.SimpleNamespace(cpu_group=object(), world_size=2)
    obj.close = Mock()
    monkeypatch.setattr(module.dist, "monitored_barrier", Mock(side_effect=RuntimeError("peer unavailable")))
    with pytest.raises(RuntimeError, match="peer unavailable"):
        obj.close_collective()
    obj.close.assert_not_called()


def test_owner_is_retained_if_consumer_release_fails(monkeypatch):
    module = load_module(monkeypatch)
    obj = module.LocalSharedVmmBuffer.__new__(module.LocalSharedVmmBuffer)
    obj.owner = True
    obj.group = types.SimpleNamespace(cpu_group=object(), world_size=2)
    obj.close = Mock()
    monkeypatch.setattr(module.dist, "monitored_barrier", Mock())

    def gather(output, value, group):
        output[:] = [None, "consumer release error"]

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="consumer release error"):
        obj.close_collective()
    obj.close.assert_not_called()


def test_synchronized_constructor_failure_has_distinct_type_and_cleans_owner(monkeypatch, tmp_path):
    module = load_module(monkeypatch)
    directory = tmp_path / "capability"
    directory.mkdir()
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **_kwargs: str(directory))
    monkeypatch.setattr(module, "MappedHostBuffer", Mock(side_effect=RuntimeError("allocation failed")))
    monkeypatch.setattr(module, "retry_pending_rollbacks", Mock())
    monkeypatch.setattr(module.dist, "get_global_rank", lambda _group, rank: rank)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.dist, "monitored_barrier", Mock())
    monkeypatch.setattr(module.dist, "all_gather_object", lambda out, value, group: out.__setitem__(0, value))
    group = types.SimpleNamespace(rank_in_group=0, cpu_group=object(), world_size=2)
    with pytest.raises(module.EngramSharedInitError, match="rank 0: ERROR: RuntimeError: allocation failed"):
        module.LocalSharedVmmBuffer((8, 256), None, "npu:0", group)
    assert not directory.exists()
