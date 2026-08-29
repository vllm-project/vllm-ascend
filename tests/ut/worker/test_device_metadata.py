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
    event_names = iter(("inputs", "compressor", "indexer", "attention", "reusable"))

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
        ),
        DeviceMetadataTask(
            DeviceMetadataStage.INDEXER,
            lambda: calls.append(("task", "indexer")),
        ),
        DeviceMetadataTask(
            DeviceMetadataStage.ATTENTION,
            lambda: calls.append(("task", "attention")),
        ),
    )


def test_submit_records_inputs_and_stage_frontiers(executor_env):
    executor, calls, allocations = executor_env

    assert allocations == [
        "stream",
        "inputs",
        "compressor",
        "indexer",
        "attention",
        "reusable",
    ]
    executor.submit(_tasks(calls))

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
    executor.wait(DeviceMetadataStage.INDEXER)
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

    executor.wait(DeviceMetadataStage.INDEXER)
    executor.wait(DeviceMetadataStage.INDEXER)

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


def test_submit_orders_tasks_by_stage(executor_env):
    executor, calls, _ = executor_env
    tasks = tuple(reversed(_tasks(calls)))

    executor.submit(tasks)

    assert [call for call in calls if call[0] == "task"] == [
        ("task", "compressor"),
        ("task", "indexer"),
        ("task", "attention"),
    ]


def test_task_provider_is_structural():
    class LegacyBuilder:
        pass

    class ProviderBuilder:
        def build_device_metadata_tasks(self, _context):
            return (DeviceMetadataTask(DeviceMetadataStage.INDEXER, lambda: None),)

    assert not isinstance(LegacyBuilder(), DeviceMetadataTaskProvider)
    assert isinstance(ProviderBuilder(), DeviceMetadataTaskProvider)


def test_submit_rejects_empty_tasks(executor_env):
    executor, _, _ = executor_env

    with pytest.raises(ValueError, match="At least one"):
        executor.submit(())


def test_wait_helper_uses_active_forward_executor(monkeypatch):
    calls = []
    executor = SimpleNamespace(wait=lambda stage: calls.append(stage))
    monkeypatch.setattr(device_metadata, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        device_metadata,
        "get_forward_context",
        lambda: SimpleNamespace(device_metadata_executor=executor),
    )

    wait_for_device_metadata(DeviceMetadataStage.ATTENTION)

    assert calls == [DeviceMetadataStage.ATTENTION]


def test_wait_helper_is_noop_without_forward_context(monkeypatch):
    monkeypatch.setattr(device_metadata, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(
        device_metadata,
        "get_forward_context",
        lambda: pytest.fail("forward context should not be read"),
    )

    wait_for_device_metadata(DeviceMetadataStage.ATTENTION)
