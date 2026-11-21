# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.expert_transfer import expert_offload_manager as mod
from vllm_ascend.distributed.expert_transfer.expert_offload_utils import prepare_batch


class FakeEvent:
    pass


class FakeStream:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def wait_event(self, event):
        self.calls.append((self.name, "wait", event))

    def record_event(self, event):
        self.calls.append((self.name, "record", event))

    def synchronize(self):
        self.calls.append((self.name, "synchronize"))


class FakePool:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.owner = None
        self.tensors = {"pool": name}
        self.fail = False

    def load_layer(self, binding, *, stream=None, synchronize=True):
        if self.owner is binding:
            return self.tensors
        self.calls.append((self.name, "load", binding.layer_name, stream, synchronize))
        self.owner = None
        if self.fail:
            raise RuntimeError("copy failed")
        self.owner = binding
        return self.tensors


def make_pipeline(monkeypatch):
    calls = []
    compute, load = FakeStream("compute", calls), FakeStream("load", calls)
    monkeypatch.setattr(
        mod.torch, "npu", SimpleNamespace(current_stream=lambda _: compute, Event=FakeEvent), raising=False
    )
    bindings = [SimpleNamespace(layer_name=str(i), device="npu") for i in range(4)]
    pools = [FakePool("pool0", calls), FakePool("pool1", calls)]
    return mod.PrefillPipeline(bindings, pools, load), bindings, pools, calls


def test_prefetch_waits_for_slot_read_and_compute_waits_for_load(monkeypatch):
    pipe, bindings, pools, calls = make_pipeline(monkeypatch)
    assert pipe.prepare(bindings[0]) is pools[0].tensors
    pipe.finish(bindings[0], prefetch=True)
    load1 = pipe.ready[1]
    assert ("pool1", "load", "1", pipe.load_stream, False) in calls
    assert pipe.prepare(bindings[1]) is pools[1].tensors
    assert ("compute", "wait", load1) in calls
    last0 = pipe.last_use[0]
    pipe.finish(bindings[1], prefetch=True)
    assert calls.index(("load", "wait", last0)) < calls.index(("pool0", "load", "2", pipe.load_stream, False))
    assert pipe.prepare(bindings[2]) is pools[0].tensors
    assert pools[0].owner is bindings[2]


def test_interrupted_order_waits_before_overwrite(monkeypatch):
    pipe, bindings, pools, calls = make_pipeline(monkeypatch)
    pipe.prepare(bindings[0])
    pipe.finish(bindings[0], prefetch=True)
    pipe.prepare(bindings[1])
    pipe.finish(bindings[1], prefetch=True)
    pending = pipe.ready[0]
    pipe.prepare(bindings[3])
    assert calls.index(("compute", "wait", pending)) < calls.index(("pool0", "load", "3", None, True))
    assert pools[0].owner is bindings[3]


def test_failed_prefetch_drains_and_invalidates(monkeypatch):
    pipe, bindings, pools, calls = make_pipeline(monkeypatch)
    pipe.prepare(bindings[0])
    pools[1].fail = True
    with pytest.raises(RuntimeError, match="copy failed"):
        pipe.finish(bindings[0], prefetch=True)
    assert pools[1].owner is None
    assert ("load", "synchronize") in calls
    assert pipe.active is None


def test_failed_compute_records_use_without_prefetch(monkeypatch):
    pipe, bindings, pools, calls = make_pipeline(monkeypatch)
    pipe.prepare(bindings[0])
    pipe.finish(bindings[0], prefetch=False)
    assert pipe.last_use[0] is not None
    assert pools[1].owner is None
    assert not any(call[0] == "pool1" for call in calls)


def test_decode_drains_interrupted_prefetch_before_resident_updates(monkeypatch):
    pipe, bindings, _, calls = make_pipeline(monkeypatch)
    manager = mod.ExpertOffloadManager()
    manager.prefill_pipeline = pipe
    pipe.prepare(bindings[0])
    pipe.finish(bindings[0], prefetch=True)
    pipe.prepare(bindings[1])  # The event was waited on-device, not necessarily completed on the host.
    assert not any(pipe.ready)
    assert pipe.inflight
    runner = SimpleNamespace(model=SimpleNamespace(_expert_offload_manager=manager))
    prepare_batch(runner, SimpleNamespace(has_prefill=False))
    assert ("load", "synchronize") in calls
    assert pipe.ready == [None, None]
    assert not manager.has_prefill


def make_graph_manager(monkeypatch):
    pipe, bindings, pools, calls = make_pipeline(monkeypatch)
    manager = mod.ExpertOffloadManager()
    manager.prefill_pipeline = pipe
    manager.full_pool = pools[0]
    manager.graph_streams.add(object())
    monkeypatch.setattr(
        mod.torch.npu,
        "synchronize",
        lambda device: calls.append(("graph", "synchronize", device)),
        raising=False,
    )
    return manager, pipe, bindings, pools, calls


def test_captured_graph_keeps_prefill_pipeline_and_fences_residency_once(monkeypatch):
    manager, pipe, bindings, pools, calls = make_graph_manager(monkeypatch)
    manager.capture(lambda: None)
    manager.set_batch_phase(True)
    assert calls == []  # Batch classification does not wait on the device.

    for index, binding in enumerate(bindings):
        assert manager.prepare_full_weights(binding) is pools[index % 2].tensors
        manager.finish_prefill_layer(binding, prefetch=True)

    assert calls[0] == ("graph", "synchronize", bindings[0].device)
    assert sum(call[:2] == ("graph", "synchronize") for call in calls) == 1
    prefetches = [call for call in calls if call[1] == "load" and call[-1] is False]
    assert [call[2] for call in prefetches] == ["1", "2", "3"]
    assert all(call[3] is pipe.load_stream for call in prefetches)
    assert ("load", "synchronize") not in calls


def test_each_decode_phase_requires_a_new_fence_before_prefill(monkeypatch):
    manager, pipe, bindings, _, calls = make_graph_manager(monkeypatch)
    manager.prepare_full_weights(bindings[0])
    manager.finish_prefill_layer(bindings[0], prefetch=True)
    before_decode = len(calls)
    for _ in range(3):
        # Even a transient prefill classification must not fence decode-only work.
        manager.set_batch_phase(True)
        manager.set_batch_phase(False)
    assert calls[before_decode:] == [("load", "synchronize")]
    assert not pipe.inflight

    manager.set_batch_phase(True)
    manager.prepare_full_weights(bindings[0])  # Also fence when the pool's owner is unchanged.
    assert sum(call[:2] == ("graph", "synchronize") for call in calls) == 2
    assert calls.index(("load", "synchronize")) < max(
        i for i, call in enumerate(calls) if call[:2] == ("graph", "synchronize")
    )
    manager.finish_prefill_layer(bindings[0], prefetch=True)
    manager.prepare_full_weights(bindings[1])
    manager.finish_prefill_layer(bindings[1], prefetch=True)
    assert pipe.pools[0].owner is bindings[2]


def test_failed_graph_fence_can_retry_before_any_pool_load(monkeypatch):
    manager, _, bindings, _, calls = make_graph_manager(monkeypatch)

    def synchronize(device):
        calls.append(("graph", "synchronize", device))
        if len(calls) == 1:
            raise RuntimeError("graph fence failed")

    monkeypatch.setattr(mod.torch.npu, "synchronize", synchronize)
    with pytest.raises(RuntimeError, match="graph fence failed"):
        manager.prepare_full_weights(bindings[0])
    assert calls == [("graph", "synchronize", bindings[0].device)]
    manager.prepare_full_weights(bindings[0])
    assert calls[:2] == [("graph", "synchronize", bindings[0].device)] * 2
    assert sum(call[1] == "load" for call in calls) == 1


def test_eager_decode_to_prefill_keeps_pipeline_without_device_fence(monkeypatch):
    manager, pipe, bindings, pools, calls = make_graph_manager(monkeypatch)
    manager.graph_streams.clear()
    manager.set_batch_phase(False)
    manager.set_batch_phase(True)
    assert manager.prepare_full_weights(bindings[0]) is pools[0].tensors
    manager.finish_prefill_layer(bindings[0], prefetch=True)
    assert ("pool1", "load", "1", pipe.load_stream, False) in calls
    assert not any(call[:2] == ("graph", "synchronize") for call in calls)


def test_single_layer_pipeline_reuses_pool_without_prefetch(monkeypatch):
    pipe, bindings, pools, calls = make_pipeline(monkeypatch)
    pipe.bindings = bindings[:1]
    for _ in range(2):
        assert pipe.prepare(bindings[0]) is pools[0].tensors
        pipe.finish(bindings[0], prefetch=True)
    assert sum(call[1] == "load" for call in calls) == 1
    assert pools[1].owner is None
    assert not pipe.inflight
