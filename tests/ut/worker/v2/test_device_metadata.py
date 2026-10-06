# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import vllm_ascend.worker.v2.device_metadata as module
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, DeviceMetadataTask


class Executor:
    submission_in_flight = False

    def __init__(self, **kwargs):
        self.calls = []

    def submit(self, tasks):
        self.calls.append("submit")
        self.submission_in_flight = True

    def wait(self, stage, group_id):
        self.calls.append("wait")

    def release(self):
        self.calls.append("release")
        self.submission_in_flight = False


class Builder:
    enabled = False
    tasks = ()

    @contextmanager
    def defer_device_metadata(self, *, in_graph=False):
        self.enabled = True
        self.in_graph = in_graph
        try:
            yield
        finally:
            self.enabled = False

    def take_device_metadata_tasks(self):
        tasks, self.tasks = self.tasks, ()
        return tasks


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(module, "DeviceMetadataExecutor", Executor)
    return module.TargetDeviceMetadata()


def prepare(state, builder, full=False, fail=False):
    groups = [[SimpleNamespace(get_metadata_builder=lambda _: builder)]]
    with state.build(groups, full):
        assert builder.enabled and builder.in_graph == full
        builder.tasks = (DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: None, 1),)
        if fail:
            raise ValueError("build failure")
    assert not builder.enabled


def test_eager_lifecycle(state):
    with state.activate():
        prepare(state, Builder())
        assert state.executor.calls == ["submit"]
    assert state.executor.calls == ["submit", "wait", "release"]


def test_graph_producers_only_run_inside_forward_not_runtime_prepare(state):
    with state.activate():
        prepare(state, Builder(), full=True)
        assert state.executor.calls == []
        state.begin_forward()  # ModelWithContext, inside capture
        state.finish()
        assert state.executor.calls == ["submit", "wait", "release"]
        prepare(state, Builder(), full=True)
        state.finish_replay()  # all device work comes from captured nodes
    assert state.executor.calls == ["submit", "wait", "release"]


def test_build_failure_clears_provider(state):
    builder = Builder()
    with pytest.raises(ValueError), state.activate():
        prepare(state, builder, fail=True)
    assert not builder.enabled and builder.tasks == ()
    assert state.executor.calls == []


@pytest.mark.parametrize("full", [False, True])
def test_unretired_inputs_cannot_be_rebuilt(state, full):
    with state.activate():
        prepare(state, Builder(), full=full)
        with pytest.raises(RuntimeError, match="not retired"):
            prepare(state, Builder(), full=full)


def test_failure_joins_only_submitted_stream(state, monkeypatch):
    waits: list[object] = []
    state.executor.stream = object()
    monkeypatch.setattr(module.torch.npu, "current_stream", lambda: SimpleNamespace(wait_stream=waits.append))

    def fail(tasks):
        state.executor.submission_in_flight = True
        raise ValueError("producer failure")

    state.executor.submit = fail
    with pytest.raises(ValueError), state.activate():
        prepare(state, Builder(), full=True)
        state.begin_forward()
    assert waits == [state.executor.stream]
    assert state.executor.calls == ["release"]
    with pytest.raises(RuntimeError, match="recreate"):
        prepare(state, Builder(), full=True)
