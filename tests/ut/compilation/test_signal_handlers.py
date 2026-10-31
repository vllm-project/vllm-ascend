# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import runpy
import signal
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend.utils import preserve_framework_signal_handlers


@pytest.mark.parametrize(
    "relative,module_name",
    [
        ("compiler_interface.py", "npugraph_ex._acl_concrete_graph.static_kernel"),
        ("passes/base_pattern.py", "npugraph_ex"),
        ("passes/base_pattern.py", "torchair"),
    ],
)
@pytest.mark.parametrize("fails", [False, True])
def test_graph_import_preserves_framework_shutdown(monkeypatch, relative, module_name, fails):
    """Exercise real import sites with graph packages replacing both handlers."""
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    received, imported = [], []
    original_import = builtins.__import__

    def framework_handler(signum, frame):
        received.append(signum)

    def import_graph(name, *args, **kwargs):
        if module_name == "torchair" and name == "npugraph_ex":
            raise ImportError("use legacy graph backend")
        if name != module_name:
            return original_import(name, *args, **kwargs)
        imported.append(name)
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if fails:
            raise RuntimeError("graph import failed")
        return SimpleNamespace(_set_static_kernel_blacklist=lambda *_: None)

    path = Path(__file__).parents[3] / "vllm_ascend" / "compilation" / relative
    try:
        for signum in previous:
            signal.signal(signum, framework_handler)
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", import_graph)
            if fails:
                with pytest.raises(RuntimeError, match="graph import failed"):
                    runpy.run_path(str(path))
            else:
                runpy.run_path(str(path))
        assert imported == [module_name]
        for signum in previous:
            signal.raise_signal(signum)
        assert received == list(previous)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def test_graph_import_context_can_run_on_background_thread():
    completed, errors = [], []

    def initialize():
        try:
            with preserve_framework_signal_handlers():
                completed.append(True)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=initialize)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive() and completed == [True] and not errors


@pytest.mark.parametrize("v2_runner", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_model_constructor_import_cannot_replace_worker_shutdown(monkeypatch, v2_runner, fails):
    from vllm_ascend.worker import worker as worker_module

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    received = []

    def framework_handler(signum, frame):
        received.append(signum)

    def load_model():
        # CANN operator registration can import a graph package without going
        # through Ascend's compiler interface, even in eager mode.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if fails:
            raise RuntimeError("model loading failed")

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(enable_sleep_mode=False), weight_transfer_config=None),
        model_runner=SimpleNamespace(load_model=load_model),
        use_v2_model_runner=v2_runner,
    )
    monkeypatch.setattr(worker_module, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(worker_module, "has_ec_transfer", lambda: False)
    try:
        for signum in previous:
            signal.signal(signum, framework_handler)
        if fails:
            with pytest.raises(RuntimeError, match="model loading failed"):
                worker_module.NPUWorker.load_model(worker)
        else:
            worker_module.NPUWorker.load_model(worker)
        for signum in previous:
            signal.raise_signal(signum)
        assert received == list(previous)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
