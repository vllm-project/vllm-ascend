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

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
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


@contextmanager
def execution(state):
    try:
        yield
    finally:
        state.finish()


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
    monkeypatch.setattr(device_metadata, "DeviceMetadataExecutor", Executor)
    return device_metadata.TargetDeviceMetadata()


def prepare(state, builder, full=False, fail=False):
    groups = [[SimpleNamespace(get_metadata_builder=lambda _: builder)]]
    with state.build(groups, full):
        assert builder.enabled and builder.in_graph == full
        builder.tasks = (DeviceMetadataTask(DeviceMetadataStage.ATTENTION, lambda: None, 1),)
        if fail:
            raise ValueError("build failure")
    assert not builder.enabled


def test_eager_lifecycle(state):
    with execution(state):
        prepare(state, Builder())
        assert state.executor.calls == ["submit"]
    assert state.executor.calls == ["submit", "wait", "release"]


def test_graph_producers_only_run_inside_forward_not_runtime_prepare(state):
    with execution(state):
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
    with pytest.raises(ValueError), execution(state):
        prepare(state, builder, fail=True)
    assert not builder.enabled and builder.tasks == ()
    assert state.executor.calls == []


@pytest.mark.parametrize("full", [False, True])
def test_unretired_inputs_cannot_be_rebuilt(state, full):
    with execution(state):
        prepare(state, Builder(), full=full)
        with pytest.raises(RuntimeError, match="not retired"):
            prepare(state, Builder(), full=full)


def test_failure_joins_only_submitted_stream(state, monkeypatch):
    waits: list[object] = []
    state.executor.stream = object()
    monkeypatch.setattr(device_metadata.torch.npu, "current_stream", lambda: SimpleNamespace(wait_stream=waits.append))

    def fail(tasks):
        state.executor.submission_in_flight = True
        raise ValueError("producer failure")

    state.executor.submit = fail
    with pytest.raises(ValueError), execution(state):
        prepare(state, Builder(), full=True)
        state.begin_forward()
    assert waits == [state.executor.stream]
    assert state.executor.calls == ["release"]
    with pytest.raises(RuntimeError, match="recreate"):
        prepare(state, Builder(), full=True)


def test_executor_belongs_to_built_metadata(state):
    builder = Builder()
    resource = SimpleNamespace()
    result = state.run_build(
        lambda **kwargs: {"cache": resource},
        attn_groups=[[SimpleNamespace(get_metadata_builder=lambda _: builder)]],
        full_graph_mode=True,
    )
    assert result["cache"].device_metadata_executor is state.executor


def test_wait_uses_forward_metadata_without_leaking_into_draft(monkeypatch):
    import vllm_ascend.worker.device_metadata as common

    calls = []
    executor = SimpleNamespace(wait=lambda *args: calls.append(args))
    context = SimpleNamespace(attn_metadata={"cache": SimpleNamespace(device_metadata_executor=executor)})
    monkeypatch.setattr(common, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(common, "get_forward_context", lambda: context)
    common.wait_for_device_metadata(DeviceMetadataStage.INDEXER, 7)
    context.attn_metadata = {"draft": SimpleNamespace()}
    common.wait_for_device_metadata(DeviceMetadataStage.INDEXER, 8)
    assert calls == [(DeviceMetadataStage.INDEXER, 7)]


def test_capture_preparation_failure_retires_metadata(state, monkeypatch):
    from contextlib import nullcontext

    import torch

    from vllm_ascend.worker.v2 import aclgraph_utils

    prepare(state, Builder(), full=True)
    manager = aclgraph_utils.ModelAclGraphManager.__new__(aclgraph_utils.ModelAclGraphManager)
    manager.model_runner = SimpleNamespace()
    manager._capture_descs = {}
    manager._v41_sharded_descs = {}
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)

    def fail(*args, **kwargs):
        raise ValueError("capture input preparation failed")

    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", fail)
    with pytest.raises(ValueError, match="capture input preparation"):
        manager.capture(torch.nn.Identity(), SimpleNamespace(device_metadata=state), None, None, None, [], None)
    # A new preparation succeeds; stale work from the failed capture is gone.
    prepare(state, Builder(), full=True)
    state.finish()
    assert state.executor.calls == []


def test_v41_full_decode_replays_the_graph_matching_indexer_metadata(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from vllm_ascend.attention.dsa_v41_indexer_shard import V41_FULL_INDEXER_SHARD
    from vllm_ascend.worker.v2 import aclgraph_utils
    from vllm_ascend.worker.v2.model_states import default as model_state_module

    short = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 8, uniform_token_count=1)
    dspark = BatchExecutionDescriptor(CUDAGraphMode.FULL, 24, 4, uniform_token_count=6)
    small = BatchExecutionDescriptor(CUDAGraphMode.FULL, 4, 4, uniform_token_count=1)

    def init(self, vllm_config, *args, **kwargs):
        self.vllm_config = vllm_config
        self.graphs = {}
        self._capture_descs = {CUDAGraphMode.FULL: [dspark, short, small]}

    def capture(self, *_args, **_kwargs):
        for desc in self._capture_descs[CUDAGraphMode.FULL]:
            assert desc not in self.graphs
            self.graphs[desc] = V41_FULL_INDEXER_SHARD.get()

    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "__init__", init)
    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", capture)
    monkeypatch.setattr(aclgraph_utils, "set_graph_params", lambda _sizes: None)
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)
    monkeypatch.setattr(aclgraph_utils, "set_current_vllm_config", lambda _config: nullcontext())
    monkeypatch.setattr(aclgraph_utils, "_get_graph_update_backend", lambda _groups: None)
    monkeypatch.setattr(aclgraph_utils, "use_updatable_graph", lambda _backend: False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, decode_context_parallel_size=1, prefill_context_parallel_size=2
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        lora_config=None,
    )
    config.model_config.hf_text_config = SimpleNamespace(index_topk=512)
    config.model_config.max_model_len = 8192
    model_state = model_state_module.AscendModelState.__new__(model_state_module.AscendModelState)
    model_state.vllm_config = config
    model_state.model_config = config.model_config
    model_state.decoder_replay = None
    global_batch = SimpleNamespace(num_reqs=2, is_dummy=False)
    pcp_context = SimpleNamespace(global_batch=global_batch, min_local_num_tokens=0)
    model_state.pcp_manager = SimpleNamespace(
        is_target_attn_groups=lambda _groups: True,
        build_attention_context=lambda *args: pcp_context,
    )
    monkeypatch.setattr(model_state_module, "get_ascend_config", SimpleNamespace)
    monkeypatch.setattr(
        model_state_module, "build_attn_metadata", lambda **_kwargs: {"sharded": V41_FULL_INDEXER_SHARD.get()}
    )
    runner = SimpleNamespace(model_state=model_state, attn_groups=[], update_stream=object())
    manager = aclgraph_utils.ModelAclGraphManager(
        config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, 6, runner
    )
    manager.capture(torch.nn.Identity(), model_state, None, None, None, [], None)
    assert manager.capture_sizes == [4, 8, 24]
    assert len(manager.graphs) == 5
    assert V41_FULL_INDEXER_SHARD.get() is None

    replayed = []
    monkeypatch.setattr(manager, "_graph_relay", lambda _backend, desc, _num_tokens, _metadata: replayed.append(desc))
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=0,
        num_computed_tokens_np=np.zeros(1, dtype=np.int32),
        query_start_loc_np=np.zeros(2, dtype=np.int32),
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        is_prefilling_np=np.zeros(1, dtype=np.bool_),
        num_scheduled_tokens=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.zeros(1, dtype=torch.int32),
        seq_lens_np=np.zeros(1, dtype=np.int32),
        positions=torch.empty(0, dtype=torch.int64),
        dcp_local_seq_lens=None,
        attn_state=None,
        is_dummy=False,
    )
    for desc in (short, dspark, small):
        batch.num_tokens_after_padding = desc.num_tokens
        batch.num_reqs_after_padding = desc.num_reqs
        for hist, min_rows in ((128, desc.num_tokens), (8192, desc.num_tokens), (8192, 0)):
            global_batch.num_computed_tokens_np = np.array([hist, hist], dtype=np.int32)
            pcp_context.min_local_num_tokens = min_rows
            batch.num_tokens = min_rows
            expected = hist == 8192 and min_rows > 0 and desc is not small
            model_state.prepare_attn(batch, CUDAGraphMode.FULL, (), None, [], None)
            manager.run_fullgraph(desc)
            replay_desc = replayed[-1]
            assert replay_desc.num_tokens == desc.num_tokens
            assert replay_desc.uniform_token_count == desc.uniform_token_count
            assert manager.graphs[replay_desc] is model_state.attn_metadata["sharded"] is expected
