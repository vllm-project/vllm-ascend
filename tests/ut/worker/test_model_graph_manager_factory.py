# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.worker.v2 import model_runner


@pytest.mark.parametrize("specialized", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_graph_manager_factory_uses_model_state_hook(monkeypatch, specialized, legacy):
    calls = []

    class DefaultManager:
        def __init__(self, *args, **kwargs):
            calls.append((type(self), args, kwargs))

    class SpecializedManager(DefaultManager):
        pass

    monkeypatch.setattr(model_runner, "ModelAclGraphManager", DefaultManager)
    monkeypatch.setattr(model_runner, "vllm_version_is", lambda version: legacy)
    state = SimpleNamespace()
    if specialized:
        state.get_graph_manager_cls = lambda: SpecializedManager
    runner = SimpleNamespace(model_state=state)
    original = model_runner.vllm_model_runner.ModelCudaGraphManager
    config, device, mode, ubatch_runner = object(), object(), object(), object()
    query_shapes = 6 if legacy else {1: 6, 2: 3}
    query_kwargs = {"decode_query_len": query_shapes} if legacy else {"decode_query_lens": query_shapes}
    with model_runner.graph_manager_wrapper(runner):
        manager = model_runner.vllm_model_runner.ModelCudaGraphManager(
            config,
            device,
            mode,
            **query_kwargs,
            lora_capture_cases=[0],
            varlen_decode=True,
            ubatch_runner=ubatch_runner,
        )
    expected = SpecializedManager if specialized else DefaultManager
    assert type(manager) is expected
    assert calls == [
        (
            expected,
            (config, device, mode, query_shapes, runner),
            {"lora_capture_cases": [0], "varlen_decode": True, "ubatch_runner": ubatch_runner},
        )
    ]
    assert model_runner.vllm_model_runner.ModelCudaGraphManager is original
