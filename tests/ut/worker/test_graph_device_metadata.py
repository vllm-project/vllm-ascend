# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import vllm_ascend.worker.device_metadata as module
from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor, DeviceMetadataStage, DeviceMetadataTask


@pytest.fixture
def executor(monkeypatch):
    calls = []

    class Stream:
        def __init__(self, name="metadata"):
            self.name = name

        def wait_event(self, event):
            calls.append((self.name, "wait", event))

    class Event:
        def record(self, stream):
            calls.append((stream.name, "record", self))

    main = Stream("main")
    monkeypatch.setattr(module.torch.npu, "Stream", Stream)
    monkeypatch.setattr(module.torch.npu, "Event", Event)
    monkeypatch.setattr(module.torch.npu, "current_stream", lambda: main)
    monkeypatch.setattr(module.torch.npu, "stream", lambda _: nullcontext())
    return DeviceMetadataExecutor(capture_producers=True), calls


@pytest.mark.parametrize("order", [(1, 2), (2, 1)])
def test_stage_order_first_consumption_and_reuse(executor, order):
    state, calls = executor
    stage = DeviceMetadataStage
    tasks = tuple(
        DeviceMetadataTask(s, lambda s=s, g=g: calls.append(("task", s, g)), g)
        for s, g in ((stage.INDEXER, 3), (stage.ATTENTION, 1), (stage.ATTENTION, 2))
    )
    state.submit(tasks)
    assert calls[0][0:2] == ("main", "record")
    assert calls[1] == ("metadata", "wait", calls[0][2])
    assert [c[2] for c in calls if c[0] == "task"] == [1, 2, 3]
    with pytest.raises(RuntimeError, match="not been released"):
        state.submit(tasks)
    for group in order:
        state.wait(stage.ATTENTION, group)
        state.wait(stage.ATTENTION, group)
    assert len([c for c in calls if c[:2] == ("main", "wait")]) == 2
    state.wait(stage.INDEXER, 3)
    state.release()
    calls.clear()
    state.submit(tasks)
    assert [c[2] for c in calls if c[0] == "task"] == [*order, 3]
    # No external-event reset or cross-iteration fence; input readiness is
    # recorded after the previous main-stream consumers, including the join.
    assert calls[1] == ("metadata", "wait", calls[0][2])


def test_empty_and_unsubmitted_rejected(executor):
    state, _ = executor
    with pytest.raises(ValueError):
        state.submit(())
    with pytest.raises(RuntimeError):
        state.wait(DeviceMetadataStage.ATTENTION, 1)
    with pytest.raises(RuntimeError):
        state.release()


def test_failure_retains_in_flight_state(executor):
    state, _ = executor

    def fail():
        raise ValueError("failed producer")

    with pytest.raises(ValueError, match="failed producer"):
        state.submit((DeviceMetadataTask(DeviceMetadataStage.ATTENTION, fail, 1),))
    assert state.submission_in_flight


def test_executor_context_is_scoped(monkeypatch):
    calls = []
    state = SimpleNamespace(wait=lambda *args: calls.append(args))
    monkeypatch.setattr(module, "is_forward_context_available", lambda: True)
    context = SimpleNamespace(attn_metadata={"cache": SimpleNamespace(device_metadata_executor=state)})
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    module.wait_for_device_metadata(DeviceMetadataStage.ATTENTION, 1)
    context.attn_metadata = {}
    module.wait_for_device_metadata(DeviceMetadataStage.ATTENTION, 2)
    assert calls == [(DeviceMetadataStage.ATTENTION, 1)]
