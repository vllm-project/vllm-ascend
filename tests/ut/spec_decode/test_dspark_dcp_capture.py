# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as graph


@pytest.mark.parametrize("dcp_size", [1, 2, 8])
@pytest.mark.parametrize("fail", [False, True])
def test_capture_keeps_communicator_and_model_contexts(monkeypatch, dcp_size, fail):
    manager = object.__new__(graph.DFlashAclGraphManager)
    parallel = SimpleNamespace(decode_context_parallel_size=dcp_size)
    manager.speculator = SimpleNamespace(attn_vllm_config=SimpleNamespace(parallel_config=parallel))
    events = []

    @contextmanager
    def recording_context(name):
        events.append(f"enter:{name}")
        try:
            yield
        finally:
            events.append(f"exit:{name}")

    monkeypatch.setattr(graph, "communicator_switch", lambda: recording_context("communicator"))

    def capture_context(speculator, is_prefill):
        assert speculator is manager.speculator
        assert is_prefill is False
        return recording_context("model")

    monkeypatch.setattr(graph, "model_capture_wrapper", capture_context)

    def capture(self, *args):
        assert self is manager
        assert args[5] == 70000
        assert events == ["enter:communicator", "enter:model"]
        if fail:
            raise RuntimeError("capture failed")

    monkeypatch.setattr(graph.DFlashCudaGraphManager, "capture", capture)
    with pytest.raises(RuntimeError, match="capture failed") if fail else nullcontext():
        manager.capture(None, None, None, [], None, 70000)
    assert events == ["enter:communicator", "enter:model", "exit:model", "exit:communicator"]
