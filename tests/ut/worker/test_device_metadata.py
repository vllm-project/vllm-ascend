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

import vllm_ascend.worker.device_metadata as device_metadata
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataExecutor,
    DeviceMetadataStage,
    DeviceMetadataTask,
    DeviceMetadataTaskProvider,
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


@pytest.fixture
def executor_env(monkeypatch):
    calls: list[tuple] = []
    allocations: list[str] = []
    model_stream = _FakeStream("model", calls)
    metadata_stream = _FakeStream("metadata", calls)
    event_names = iter(("inputs", "reusable", "compressor", "indexer", "attention", "indexer-2"))

    def make_stream():
        allocations.append("stream")
        return metadata_stream

    def make_event():
        name = next(event_names)
        allocations.append(name)
        return _FakeEvent(name, calls)

    monkeypatch.setattr(torch.npu, "Stream", make_stream)
    monkeypatch.setattr(torch.npu, "Event", make_event)
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


def test_submit_records_inputs_and_stage_frontiers(executor_env):
    executor, calls, allocations = executor_env

    assert allocations == [
        "stream",
        "inputs",
        "reusable",
    ]
    executor.submit(_tasks(calls))
    assert allocations == [
        "stream",
        "inputs",
        "reusable",
        "compressor",
        "indexer",
        "attention",
    ]

    assert calls == [
        ("model", "record", "inputs"),
        ("metadata", "wait", "inputs"),
        ("task", "compressor"),
        ("metadata", "record", "compressor"),
        ("task", "indexer"),
        ("metadata", "record", "indexer"),
        ("task", "attention"),
        ("metadata", "record", "attention"),
    ]


def test_wait_and_release_fence_buffer_reuse(executor_env):
    executor, calls, _ = executor_env
    tasks = _tasks(calls)

    executor.submit(tasks)
    executor.wait(DeviceMetadataStage.INDEXER, 2)
    executor.release()
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


def test_wait_records_each_stage_once_per_submission(executor_env):
    executor, calls, _ = executor_env
    executor.submit(_tasks(calls))

    executor.wait(DeviceMetadataStage.INDEXER, 2)
    executor.wait(DeviceMetadataStage.INDEXER, 2)

    assert calls.count(("model", "wait", "indexer")) == 1


def test_submission_in_flight_tracks_release(executor_env):
    executor, calls, _ = executor_env

    assert not executor.submission_in_flight
    executor.submit(_tasks(calls))
    assert executor.submission_in_flight
    executor.release()
    assert not executor.submission_in_flight


def test_executor_rejects_overlapping_submissions(executor_env):
    executor, calls, _ = executor_env
    tasks = _tasks(calls)
    executor.submit(tasks)

    with pytest.raises(RuntimeError, match="has not been released"):
        executor.submit(tasks)


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


def test_submit_orders_tasks_by_stage(executor_env):
    executor, calls, _ = executor_env
    tasks = tuple(reversed(_tasks(calls)))

    executor.submit(tasks)

    assert [call for call in calls if call[0] == "task"] == [
        ("task", "compressor"),
        ("task", "indexer"),
        ("task", "attention"),
    ]


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


def test_task_provider_is_structural():
    class LegacyBuilder:
        pass

    class ProviderBuilder:
        def enable_device_metadata(self):
            pass

        def take_device_metadata_tasks(self):
            return (DeviceMetadataTask(DeviceMetadataStage.INDEXER, lambda: None, 1),)

    assert not isinstance(LegacyBuilder(), DeviceMetadataTaskProvider)
    assert isinstance(ProviderBuilder(), DeviceMetadataTaskProvider)


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
