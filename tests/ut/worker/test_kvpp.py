# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.kvpp_utils import (
    ManualExecutor,
    indexer_name,
    layer_name,
    make_cache_config,
    make_kvpp_config,
    make_kvpp_specs,
)
from vllm_ascend.core.kv_cache_placement import KVPPPhysicalCachePlan
from vllm_ascend.worker.v2 import kvpp


@pytest.fixture
def scheduler_device(monkeypatch):
    events: list[object] = []
    compute, transfer = Mock(), Mock()
    compute.wait_event.side_effect = lambda event: events.append(("wait", event))

    def make_event():
        event = Mock()
        event.record.side_effect = lambda stream: events.append(("ready", event, stream))
        return event

    monkeypatch.setattr(kvpp.torch.npu, "current_device", lambda: 3)
    monkeypatch.setattr(kvpp.torch.npu, "set_device", lambda device: events.append(("device", device)))
    monkeypatch.setattr(kvpp.torch.npu, "current_stream", lambda: compute)
    monkeypatch.setattr(kvpp.torch.npu, "Stream", lambda: transfer)
    monkeypatch.setattr(kvpp.torch.npu, "Event", make_event)
    monkeypatch.setattr(kvpp.torch.npu, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(kvpp, "ThreadPoolExecutor", ManualExecutor)
    return events, compute, transfer


@pytest.mark.parametrize("explicit_caches", [False, True])
def test_runtime_binds_complete_layer_storage(monkeypatch, scheduler_device, explicit_caches):
    main, indexer, mtp = layer_name(11), indexer_name(11), layer_name(17)
    specs = {name: make_kvpp_specs()[name] for name in (main, indexer, mtp)}
    plan = KVPPPhysicalCachePlan(
        logical_cache_spec=specs,
        layer_owner_ranks={main: 1, indexer: 1},
        layer_bundles={main: (main, indexer), mtp: (mtp,)},
        tensor_sizes={main: (6, 2), indexer: (8, 2), mtp: (4,)},
        kvpp_rank=0,
    )
    monkeypatch.setattr(kvpp, "create_kvpp_cache_allocation_plan", lambda *_args: plan)
    monkeypatch.setattr(
        kvpp, "get_kvpp_group", lambda: SimpleNamespace(rank_in_group=0, ranks=[4, 9], device_group=object())
    )
    backing = torch.full((40,), -1, dtype=torch.int8)
    components = (
        backing[2:14].view(torch.float16),
        backing[14:18].view(torch.float16),
        backing[18:34],
        backing[34:38].view(torch.float16),
    )
    caches = {main: components[:2], indexer: components[2:], mtp: (torch.zeros(8, dtype=torch.int8),)}
    context = {name: SimpleNamespace(kv_cache=value, impl=SimpleNamespace()) for name, value in caches.items()}
    runtime = kvpp.KVPPRuntime.create_from_kv_cache(
        vllm_config=make_kvpp_config(2),
        kv_cache_config=make_cache_config(specs, 2),
        static_forward_context=context,
        kv_caches=caches if explicit_caches else None,
    )
    scheduler = runtime.scheduler
    assert scheduler.attention_layer_names == (main,)
    assert context[main].impl.layerwise_kv_cache_hook is scheduler
    assert not hasattr(context[indexer].impl, "layerwise_kv_cache_hook")
    assert not hasattr(context[mtp].impl, "layerwise_kv_cache_hook")
    assert set(scheduler.transport._layer_buffers) == {main}
    payload = scheduler.transport._layer_buffers[main]
    assert payload.dtype == torch.int8
    assert payload.shape == (36,)
    assert payload.storage_offset() == 2
    for value, (start, end) in enumerate(((0, 12), (12, 16), (16, 32), (32, 36)), 1):
        payload[start:end].fill_(value)
    for value, component in enumerate(components, 1):
        assert torch.all(component.view(torch.int8) == value)
    assert torch.all(backing[:2] == -1)
    assert torch.all(backing[38:] == -1)
    assert torch.count_nonzero(caches[mtp][0]) == 0


def test_prefetch_sequence_across_forwards(scheduler_device):
    events, compute, transfer = scheduler_device
    transport = Mock()
    transport.prefetch.side_effect = lambda name, ready, stream: events.append(("prefetch", name, ready, stream))
    names = tuple(layer_name(i) for i in range(3))
    scheduler = kvpp.KVPPScheduler(transport, names)
    executor = scheduler._prefetch_executor
    for has_history in (False, True, False, True):
        before = len(executor.submitted)
        scheduler.schedule_forward(has_history)
        assert len(executor.submitted) == before + int(has_history)
        for index, name in enumerate(names):
            if has_history:
                future, args = executor.submitted[-1]
                assert args[0] == name
                assert not future.done()
                assert events[-1] == ("ready", args[1], compute)
                executor.run_next()
                assert future.done()
                assert events[-2:] == [("device", 3), ("prefetch", name, args[1], transfer)]
            scheduler.wait_for_layer(name)
            expected_count = min(index + 2, len(names)) if has_history else 0
            assert len(executor.submitted) == before + expected_count
        scheduler.complete_forward()
        assert not executor.pending
    assert transport.prefetch.call_count == 6


def test_hook_propagates_failed_future_without_scheduling_next(scheduler_device):
    scheduler = kvpp.KVPPScheduler(Mock(), (layer_name(0), layer_name(1)))
    scheduler.schedule_forward(True)
    error = RuntimeError("broadcast failed")
    scheduler._prefetch_executor.fail_next(error)
    with pytest.raises(RuntimeError) as raised:
        scheduler.wait_for_layer(layer_name(0))
    assert raised.value is error
    assert len(scheduler._prefetch_executor.submitted) == 1


@pytest.mark.parametrize("has_history", [False, True])
def test_graph_prefetch_runs_one_layer_ahead(scheduler_device, has_history):
    events, compute, transfer = scheduler_device
    transport = Mock()
    names = tuple(layer_name(i) for i in (2, 5, 8))
    transport.prefetch_on_stream.side_effect = lambda name, available, stream, ready: events.append(
        ("broadcast", name, available, stream, ready)
    )
    scheduler = kvpp.KVPPScheduler(transport, names)
    scheduler.schedule_forward(has_history, full_graph=True)
    assert not scheduler._prefetch_executor.submitted
    assert not transport.prefetch_on_stream.called
    for index, name in enumerate(names):
        scheduler.wait_for_layer(name)
        graph_events = scheduler._graph_events
        ready = graph_events[index][1]
        wait_index = events.index(("wait", ready))
        if index + 1 < len(names):
            available, next_ready = graph_events[index + 1]
            assert events[wait_index + 1 :] == [
                ("ready", available, compute),
                ("broadcast", names[index + 1], available, transfer, next_ready),
            ]
        events.append(("compute", name))
    assert transport.prefetch_on_stream.call_count == len(names)
    assert not transport.prefetch.called
    assert not scheduler._prefetch_executor.submitted
    captured = scheduler._captured_events[0]
    scheduler.complete_forward()
    # Replay runs no Python hooks and must launch no eager prefetch.
    scheduler.schedule_forward(True, full_graph=True)
    scheduler.complete_forward()
    assert scheduler._captured_events == [captured]
    assert transport.prefetch_on_stream.call_count == len(names)


def test_graph_capture_has_independent_events_and_checks_layer_order(scheduler_device):
    scheduler = kvpp.KVPPScheduler(Mock(), (layer_name(0), layer_name(1)))
    for _ in range(2):
        scheduler.schedule_forward(False, full_graph=True)
        with pytest.raises(RuntimeError, match="Unexpected KVPP graph layer"):
            scheduler.wait_for_layer(layer_name(1))
        scheduler.wait_for_layer(layer_name(0))
        scheduler.wait_for_layer(layer_name(1))
        scheduler.complete_forward()
    first, second = scheduler._captured_events
    assert all(a is not b for pair_a, pair_b in zip(first, second) for a, b in zip(pair_a, pair_b))


def test_graph_dependencies_preserve_overlap_and_scratch_lifetime(monkeypatch, scheduler_device):
    """Check device dependencies, including transitive serialization."""
    dependencies = {}

    class Stream:
        def __init__(self):
            self.tail = None

        def enqueue(self, name, extra=None):
            dependencies[name] = {node for node in (self.tail, extra) if node is not None}
            self.tail = name

        def wait_event(self, event):
            self.enqueue((id(self), len(dependencies)), event.node)

    class Event:
        def record(self, stream):
            self.node = stream.tail

    compute, transfer = Stream(), Stream()
    monkeypatch.setattr(kvpp.torch.npu, "current_stream", lambda: compute)
    monkeypatch.setattr(kvpp.torch.npu, "Stream", lambda: transfer)
    monkeypatch.setattr(kvpp.torch.npu, "Event", Event)
    names = tuple(layer_name(i) for i in (2, 5, 8, 9))

    def prefetch(name, available, stream, ready):
        stream.wait_event(available)
        stream.enqueue(("broadcast", names.index(name)))
        ready.record(stream)

    transport = Mock()
    transport.prefetch_on_stream.side_effect = prefetch
    scheduler = kvpp.KVPPScheduler(transport, names)
    scheduler.schedule_forward(False, full_graph=True)
    for index, name in enumerate(names):
        scheduler.wait_for_layer(name)
        compute.enqueue(("compute", index))

    def ancestors(node):
        result = set(dependencies[node])
        for parent in dependencies[node]:
            result.update(ancestors(parent))
        return result

    for index in range(len(names)):
        # All ranks, including the owner, finish reading/writing the received
        # history before this layer can update its KV cache.
        assert ("broadcast", index) in ancestors(("compute", index))
        if index > 0:
            # No device dependency orders next communication after current
            # compute, or makes current compute wait for next communication.
            assert ("compute", index - 1) not in ancestors(("broadcast", index))
            assert ("broadcast", index) not in ancestors(("compute", index - 1))
        if index > 1:
            assert ("compute", index - 2) in ancestors(("broadcast", index))
