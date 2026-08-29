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

import pytest
import torch

from vllm_ascend.worker.device_metadata import (
    DeviceMetadataExecutor,
    DeviceMetadataPlan,
    DeviceMetadataStage,
    DeviceMetadataTask,
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
    model_stream = _FakeStream("model", calls)
    metadata_stream = _FakeStream("metadata", calls)
    event_names = iter(("inputs", "compressor", "indexer", "attention", "reusable"))

    monkeypatch.setattr(torch.npu, "Stream", lambda: metadata_stream)
    monkeypatch.setattr(
        torch.npu,
        "Event",
        lambda: _FakeEvent(next(event_names), calls),
    )
    monkeypatch.setattr(torch.npu, "current_stream", lambda: model_stream)
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())

    return DeviceMetadataExecutor(), calls


def _plan(calls: list[tuple]) -> DeviceMetadataPlan:
    return DeviceMetadataPlan(
        tasks=(
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
    )


def test_submit_records_inputs_and_stage_frontiers(executor_env):
    executor, calls = executor_env

    executor.submit(_plan(calls))

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
    executor, calls = executor_env
    plan = _plan(calls)

    executor.submit(plan)
    executor.wait(DeviceMetadataStage.INDEXER)
    executor.release()
    assert calls[-2:] == [
        ("model", "wait", "indexer"),
        ("model", "record", "reusable"),
    ]
    calls.clear()
    executor.submit(plan)

    assert calls[:3] == [
        ("model", "record", "inputs"),
        ("metadata", "wait", "inputs"),
        ("metadata", "wait", "reusable"),
    ]


def test_executor_rejects_overlapping_plans(executor_env):
    executor, calls = executor_env
    plan = _plan(calls)
    executor.submit(plan)

    with pytest.raises(RuntimeError, match="has not been released"):
        executor.submit(plan)


def test_plan_rejects_out_of_order_tasks():
    with pytest.raises(ValueError, match="ordered by stage"):
        DeviceMetadataPlan(
            tasks=(
                DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: None),
                DeviceMetadataTask(DeviceMetadataStage.COMPRESSOR, lambda: None),
            )
        )
