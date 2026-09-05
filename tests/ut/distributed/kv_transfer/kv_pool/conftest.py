# SPDX-License-Identifier: Apache-2.0
"""CPU-only dependency boundaries for the kv_pool unit tests."""

import importlib.util
import socket
import subprocess
import sys
import threading
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import zmq


def _sdk_module(name):
    module = types.ModuleType(name)
    module.__path__ = []
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)

    def get_attribute(attribute):
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        value = MagicMock(name=f"{name}.{attribute}")
        setattr(module, attribute, value)
        return value

    module.__getattr__ = get_attribute
    return module


@contextmanager
def _external_sdks():
    names = (
        "mooncake",
        "mooncake.engine",
        "mooncake.store",
        "memcache_hybrid",
        "yr",
        "yr.datasystem",
        "yr.datasystem.hetero_client",
        "yr.datasystem.kv_client",
        "yr.datasystem.object_client",
        "ucm",
        "ucm.integration",
        "ucm.integration.vllm",
        "ucm.integration.vllm.ucm_connector",
    )
    with pytest.MonkeyPatch.context() as mp:
        modules = {name: _sdk_module(name) for name in names}
        for name, module in modules.items():
            mp.setitem(sys.modules, name, module)
            parent, _, child = name.rpartition(".")
            if parent in modules:
                setattr(modules[parent], child, module)
        yield


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    # Patch only optional SDK imports while this subtree is collected. Preserve
    # real vLLM/Ascend modules and restore each SDK entry afterwards.
    if Path(collector.path).is_relative_to(Path(__file__).parent):
        with _external_sdks():
            return (yield)
    return (yield)


@pytest.fixture(autouse=True)
def _cpu_dependencies(monkeypatch):
    threads_before = set(threading.enumerate())
    with _external_sdks():

        def unexpected_io(*args, **kwargs):
            raise AssertionError("kv_pool UT must replace external network/process/device operations explicitly")

        npu = MagicMock(name="npu")
        npu.current_device.return_value = "cpu"
        npu.is_available.return_value = False
        monkeypatch.setattr(torch, "npu", npu, raising=False)
        monkeypatch.setattr(socket.socket, "connect", unexpected_io)
        monkeypatch.setattr(socket.socket, "connect_ex", unexpected_io)
        monkeypatch.setattr(subprocess, "Popen", unexpected_io)
        monkeypatch.setattr(zmq, "Context", unexpected_io)
        monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", unexpected_io, raising=False)
        yield
    leaked = [thread.name for thread in threading.enumerate() if thread not in threads_before]
    assert not leaked, f"kv_pool UT leaked threads: {leaked}"
