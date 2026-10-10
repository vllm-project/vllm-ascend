# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.compilation.updatable_graph import GraphUpdateTask, UpdatableGraph


def test_lazy_binding_releases_first_task_before_binding_the_next(monkeypatch):
    calls = []
    providers = [object(), object()]
    graph = SimpleNamespace(provider_sizes={provider: 1 for provider in providers})
    tasks = [GraphUpdateTask(lambda **kwargs: None, {}, provider, 0, object(), object()) for provider in providers]
    graph.tasks = tasks
    original_bind = GraphUpdateTask.bind

    def bind(task, params):
        calls.append(("bind", params["layer"]))
        return original_bind(task, params)

    monkeypatch.setattr(GraphUpdateTask, "bind", bind)
    monkeypatch.setattr(GraphUpdateTask, "apply", lambda task, stream: calls.append(("update", task.kwargs["layer"])))
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())
    source = SimpleNamespace(get=lambda provider: [{"layer": providers.index(provider)}])
    resolved = UpdatableGraph.iter_resolved_tasks(graph, source)
    assert calls == []
    UpdatableGraph.update(graph, object(), resolved)
    assert calls == [("bind", 0), ("update", 0), ("bind", 1), ("update", 1)]


def test_invalid_later_provider_is_rejected_before_any_task_is_bound(monkeypatch):
    first, second = object(), object()
    graph = SimpleNamespace(provider_sizes={first: 1, second: 2}, tasks=[])
    source = SimpleNamespace(get=lambda provider: [{}])
    bind = MagicMock()
    monkeypatch.setattr(GraphUpdateTask, "bind", bind)
    with pytest.raises(AssertionError):
        UpdatableGraph.iter_resolved_tasks(graph, source)
    bind.assert_not_called()


def test_shared_provider_is_resolved_once_with_its_original_task_indices():
    provider = object()
    graph = SimpleNamespace(provider_sizes={provider: 2})
    graph.tasks = [GraphUpdateTask(lambda: None, {}, provider, index, None, None) for index in (1, 0)]
    source = SimpleNamespace(get=MagicMock(return_value=[{"value": 10}, {"value": 20}]))
    result = list(UpdatableGraph.iter_resolved_tasks(graph, source))
    assert [task.kwargs["value"] for task in result] == [20, 10]
    source.get.assert_called_once_with(provider)
    assert all(task.kwargs == {} for task in graph.tasks)
