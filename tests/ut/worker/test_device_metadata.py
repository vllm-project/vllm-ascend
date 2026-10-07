#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.forward_context import BatchDescriptor

import vllm_ascend.worker.device_metadata as device_metadata
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataExecutor,
    DeviceMetadataStage,
    DeviceMetadataTask,
    wait_for_device_metadata,
)


class _FakeStream:
    def __init__(self, name: str, calls: list[tuple]) -> None:
        self.name = name
        self.calls = calls

    def wait_event(self, event: "_FakeEvent") -> None:
        self.calls.append((self.name, "wait", event.name))


class _FakeEvent:
    def __init__(self, name: str, calls: list[tuple]) -> None:
        self.name = name
        self.calls = calls

    def record(self, stream: _FakeStream) -> None:
        self.calls.append((stream.name, "record", self.name))


class _FakeExternalEvent(_FakeEvent):
    def wait(self, stream: _FakeStream) -> None:
        self.calls.append((stream.name, "external_wait", self.name))

    def reset(self, stream: _FakeStream) -> None:
        self.calls.append((stream.name, "reset", self.name))


@pytest.fixture
def executor_env(monkeypatch):
    calls: list[tuple] = []
    allocations: list[str] = []
    model_stream = _FakeStream("model", calls)
    metadata_stream = _FakeStream("metadata", calls)
    event_names = iter(("inputs", "reusable", "compressor", "attention", "indexer", "extra-0", "extra-1", "extra-2"))
    external_event_names = iter(f"external-{index}" for index in range(12))

    def make_stream():
        allocations.append("stream")
        return metadata_stream

    def make_event():
        name = next(event_names)
        allocations.append(name)
        return _FakeEvent(name, calls)

    def make_external_event():
        name = next(external_event_names)
        allocations.append(name)
        return _FakeExternalEvent(name, calls)

    monkeypatch.setattr(torch.npu, "Stream", make_stream)
    monkeypatch.setattr(torch.npu, "Event", make_event)
    monkeypatch.setattr(torch.npu, "ExternalEvent", make_external_event, raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: model_stream)
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())

    return DeviceMetadataExecutor(), calls, allocations


def _tasks(calls: list[tuple]) -> tuple[DeviceMetadataTask, ...]:
    return (
        DeviceMetadataTask(
            DeviceMetadataStage.COMPRESSOR,
            lambda: calls.append(("task", "compressor")),
            1,
        ),
        DeviceMetadataTask(
            DeviceMetadataStage.INDEXER,
            lambda: calls.append(("task", "indexer")),
            2,
        ),
        DeviceMetadataTask(
            DeviceMetadataStage.ATTENTION,
            lambda: calls.append(("task", "attention")),
            3,
        ),
    )


def test_stream_lifecycle_and_stage_frontiers(executor_env):
    executor, calls, allocations = executor_env
    assert allocations == [
        "stream",
        "inputs",
        "reusable",
    ]
    tasks = tuple(reversed(_tasks(calls)))
    executor.submit(tasks)
    assert executor.submission_in_flight
    assert allocations == [
        "stream",
        "inputs",
        "reusable",
        "compressor",
        "attention",
        "indexer",
    ]

    assert calls == [
        ("model", "record", "inputs"),
        ("metadata", "wait", "inputs"),
        ("task", "compressor"),
        ("metadata", "record", "compressor"),
        ("task", "attention"),
        ("metadata", "record", "attention"),
        ("task", "indexer"),
        ("metadata", "record", "indexer"),
    ]
    executor.wait(DeviceMetadataStage.INDEXER, 2)
    executor.wait(DeviceMetadataStage.INDEXER, 2)
    assert calls.count(("model", "wait", "indexer")) == 1
    with pytest.raises(RuntimeError, match="has not been released"):
        executor.submit(tasks)
    executor.release()
    assert not executor.submission_in_flight
    assert calls[-2:] == [
        ("model", "wait", "indexer"),
        ("model", "record", "reusable"),
    ]
    calls.clear()
    executor.submit(tasks)

    assert calls[:3] == [
        ("model", "record", "inputs"),
        ("metadata", "wait", "inputs"),
        ("metadata", "wait", "reusable"),
    ]


def test_external_events_are_reused_per_batch_descriptor(executor_env):
    executor, calls, allocations = executor_env
    descriptor = BatchDescriptor(num_tokens=4, num_reqs=4)

    executor.submit(_tasks(calls), descriptor)
    assert executor.uses_external_events
    assert allocations[-3:] == ["external-0", "external-1", "external-2"]
    executor.wait(DeviceMetadataStage.INDEXER, 2)
    executor.wait(DeviceMetadataStage.INDEXER, 2)
    assert calls[-2:] == [
        ("model", "external_wait", "external-2"),
        ("model", "reset", "external-2"),
    ]
    assert calls.count(("model", "external_wait", "external-2")) == 1
    assert calls.count(("model", "reset", "external-2")) == 1
    assert calls.index(("metadata", "record", "external-2")) < calls.index(("model", "external_wait", "external-2"))
    executor.release()
    assert not executor.uses_external_events
    executor.submit(_tasks(calls), descriptor)
    assert allocations.count("external-0") == 1
    executor.release()
    executor.submit(_tasks(calls), BatchDescriptor(num_tokens=8, num_reqs=4))
    assert allocations[-3:] == ["external-3", "external-4", "external-5"]


@pytest.mark.parametrize("changed_task", ["removed", "added", "replaced"])
def test_external_event_frontiers_must_remain_stable(executor_env, changed_task):
    executor, calls, allocations = executor_env
    descriptor = BatchDescriptor(num_tokens=4, num_reqs=4)

    executor.submit(_tasks(calls), descriptor)
    executor.release()
    calls.clear()
    allocations_before = list(allocations)

    tasks = _tasks(calls)[:-1]
    if changed_task == "added":
        tasks = _tasks(calls)
    if changed_task != "removed":
        tasks += (DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: None, 99),)
    with pytest.raises(RuntimeError, match="frontiers changed"):
        executor.submit(tasks, descriptor)

    assert calls == []
    assert allocations == allocations_before
    assert not executor.submission_in_flight


@pytest.mark.parametrize("consumer_order", [(10, 20), (20, 10)])
@pytest.mark.parametrize("full_graph", [False, True])
def test_sas_order_follows_first_consumption(executor_env, consumer_order, full_graph):
    executor, calls, allocations = executor_env
    compressor = DeviceMetadataStage.COMPRESSOR
    attention = DeviceMetadataStage.ATTENTION
    indexer = DeviceMetadataStage.INDEXER
    descriptor = BatchDescriptor(num_tokens=4, num_reqs=4) if full_graph else None

    def task(stage, group_id):
        return DeviceMetadataTask(stage, lambda: calls.append(("task", stage, group_id)), group_id)

    tasks = (task(indexer, 1), task(attention, 10), task(compressor, 10), task(attention, 20), task(compressor, 20))
    executor.submit(tasks, descriptor)
    assert [call[1:] for call in calls if call[0] == "task"] == [
        (compressor, 10),
        (compressor, 20),
        (attention, 10),
        (attention, 20),
        (indexer, 1),
    ]
    for group_id in consumer_order:
        executor.wait(attention, group_id)
        executor.wait(attention, group_id)
    executor.release()
    allocations_before = list(allocations)
    calls.clear()

    executor.submit(tasks, descriptor)
    assert allocations == allocations_before
    assert [call[1:] for call in calls if call[0] == "task"] == [
        (compressor, 10),
        (compressor, 20),
        *((attention, group_id) for group_id in consumer_order),
        (indexer, 1),
    ]
    first_sas = calls.index(("task", attention, consumer_order[0]))
    ready_event = calls[first_sas + 1][2]
    assert calls[first_sas + 1][:2] == ("metadata", "record")
    assert first_sas < calls.index(("task", indexer, 1))
    executor.wait(attention, consumer_order[0])
    executor.wait(attention, consumer_order[0])
    wait_kind = "external_wait" if full_graph else "wait"
    assert calls.count(("model", wait_kind, ready_event)) == 1
    executor.release()
    calls.clear()

    # A new graph shape reuses the order and keeps unseen SAS groups at the tail.
    next_descriptor = BatchDescriptor(num_tokens=8, num_reqs=8) if full_graph else None
    executor.submit((task(attention, 30), *tasks), next_descriptor)
    assert [call[1:] for call in calls if call[0] == "task"] == [
        (compressor, 10),
        (compressor, 20),
        *((attention, group_id) for group_id in consumer_order),
        (attention, 30),
        (indexer, 1),
    ]
    executor.release()


def test_submit_failure_keeps_partial_submission_in_flight(executor_env):
    executor, calls, _ = executor_env

    def fail_task() -> None:
        raise RuntimeError("task failed")

    tasks = (
        DeviceMetadataTask(
            DeviceMetadataStage.COMPRESSOR,
            lambda: calls.append(("task", "started")),
            1,
        ),
        DeviceMetadataTask(DeviceMetadataStage.INDEXER, fail_task, 2),
    )

    with pytest.raises(RuntimeError, match="task failed"):
        executor.submit(tasks)

    assert executor.submission_in_flight
    with pytest.raises(RuntimeError, match="has not been released"):
        executor.submit(tasks)


def test_wait_uses_group_specific_frontier(executor_env):
    executor, calls, _ = executor_env
    tasks = (
        DeviceMetadataTask(DeviceMetadataStage.INDEXER, lambda: None, 2),
        DeviceMetadataTask(DeviceMetadataStage.INDEXER, lambda: None, 4),
    )

    executor.submit(tasks)
    executor.wait(DeviceMetadataStage.INDEXER, 2)
    executor.wait(DeviceMetadataStage.INDEXER, 4)

    waits = [call for call in calls if call[:2] == ("model", "wait")]
    assert len(waits) == 2
    assert waits[0][2] != waits[1][2]


def test_submit_rejects_empty_tasks(executor_env):
    executor, _, _ = executor_env

    with pytest.raises(ValueError, match="At least one"):
        executor.submit(())


def test_wait_helper_uses_active_forward_executor(monkeypatch):
    calls = []
    executor = SimpleNamespace(wait=lambda stage, group_id: calls.append((stage, group_id)))
    monkeypatch.setattr(device_metadata, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        device_metadata,
        "get_forward_context",
        lambda: SimpleNamespace(device_metadata_executor=executor),
    )

    wait_for_device_metadata(DeviceMetadataStage.ATTENTION, 7)

    assert calls == [(DeviceMetadataStage.ATTENTION, 7)]


def test_wait_helper_is_noop_without_forward_context(monkeypatch):
    monkeypatch.setattr(device_metadata, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(
        device_metadata,
        "get_forward_context",
        lambda: pytest.fail("forward context should not be read"),
    )

    wait_for_device_metadata(DeviceMetadataStage.ATTENTION, 7)
