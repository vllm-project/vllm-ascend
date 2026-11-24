# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.expert_transfer import expert_offload_manager as mod
from vllm_ascend.distributed.expert_transfer.expert_offload_utils import prepare_batch


class Task:
    def __init__(self, action, dependencies):
        self.action, self.dependencies, self.done = action, dependencies, False

    def run(self, visiting=()):
        assert self not in visiting, "cyclic stream dependency"
        if not self.done:
            for task in self.dependencies:
                task.run((*visiting, self))
            self.action()
            self.done = True

    def depends_on(self, other):
        return other in self.dependencies or any(task.depends_on(other) for task in self.dependencies)


class Stream:
    def __init__(self, name, calls):
        self.name, self.calls, self.tail = name, calls, None

    def submit(self, action=lambda: None, dependencies=()):
        self.tail = Task(action, (*dependencies, *((self.tail,) if self.tail else ())))
        return self.tail

    def wait_stream(self, other):
        self.calls.append((self.name, "wait", other.name))
        self.submit(dependencies=(other.tail,) if other.tail else ())

    def synchronize(self):
        self.calls.append((self.name, "sync"))
        if self.tail:
            self.tail.run()


def make_pool(monkeypatch, layers=4):
    calls = []
    compute, load = Stream("compute", calls), Stream("load", calls)
    manager = mod.ExpertOffloadManager()
    pool = manager.full_pool
    state = SimpleNamespace(compute=compute, fail=None, copies={})
    bindings = [
        SimpleNamespace(
            layer_name=str(i),
            device=SimpleNamespace(type="npu"),
            manager=manager,
            adapter=i,
            host_store=None,
            resident={},
            slot_to_expert=(0,),
            num_experts=4,
        )
        for i in range(layers)
    ]
    pool.bindings = bindings
    pool.buffers = [{"layer": None} for _ in range(2)]
    pool.tensors = pool.buffers[0]
    pool.owners = [None] * len(pool.buffers)
    pool.load_stream = load

    def copy(adapter, store, destination, loads, device, *, stream=None, **kwargs):
        synchronize = stream is None
        stream = stream or state.compute
        calls.append((stream.name, "copy", adapter))
        state.copies[adapter] = stream.submit(lambda: destination.update(layer=adapter))
        if synchronize or state.fail == adapter:
            stream.synchronize()
        if state.fail == adapter:
            raise RuntimeError("copy failed")

    def synchronize(device):
        calls.append(("graph", "sync"))
        compute.synchronize()
        load.synchronize()

    monkeypatch.setattr(
        mod.torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda _: state.compute,
            synchronize=synchronize,
            Stream=lambda **kwargs: load,
        ),
        raising=False,
    )
    monkeypatch.setattr(mod.TorchExpertTransfer, "copy_experts", copy)
    return manager, pool, bindings, state, load, calls


def submit_compute(pool, binding, state):
    weights = pool.load_layer(binding)

    def read():
        assert weights["layer"] == binding.adapter, "buffer overwritten before computation finished"

    return state.compute.submit(read)


@pytest.mark.parametrize("load_first", [False, True])
def test_prefetch_overlaps_compute_and_preserves_weights(monkeypatch, load_first):
    _, pool, bindings, state, load, calls = make_pool(monkeypatch)
    addresses = list(map(id, pool.buffers))
    for _ in range(2):
        for index, binding in enumerate(bindings):
            compute = submit_compute(pool, binding, state)
            if index + 1 < len(bindings):
                following_copy = state.copies[index + 1]
                assert not following_copy.depends_on(compute)
                assert not compute.depends_on(following_copy)
        streams = (load, state.compute) if load_first else (state.compute, load)
        for stream in streams:
            stream.synchronize()
    assert list(map(id, pool.buffers)) == addresses
    assert sum(call[:2] == ("compute", "copy") for call in calls) == 2


@pytest.mark.parametrize("switch_stream", [False, True])
def test_interrupted_order_and_partial_compute_are_fenced(monkeypatch, switch_stream):
    _, pool, bindings, state, load, calls = make_pool(monkeypatch)
    submit_compute(pool, bindings[0], state)
    # Simulate a partial MoE submission followed by an exception: no finish hook.
    if switch_stream:
        state.compute = Stream("replacement", calls)
    for index in (3, 0, 1, 2):
        submit_compute(pool, bindings[index], state)
    load.synchronize()
    state.compute.synchronize()


@pytest.mark.parametrize("captured_graph", [False, True])
def test_decode_drains_once_before_resident_updates(monkeypatch, captured_graph):
    manager, pool, bindings, state, load, calls = make_pool(monkeypatch)
    if captured_graph:
        manager.graph_streams.add(object())
        manager.capture(lambda: None)
        pool.load_layer(bindings[0])  # Resolve capture callbacks before measuring eager steps.
        calls.clear()
        manager.has_prefill = True
    submit_compute(pool, bindings[0], state)
    assert not state.copies[1].done
    runner = SimpleNamespace(model=SimpleNamespace(_expert_offload_manager=manager))
    for _ in range(3):
        prepare_batch(runner, SimpleNamespace(has_prefill=False))
        for binding in bindings:
            submit_compute(pool, binding, state)
    assert state.copies[1].done
    assert not pool.inflight
    assert calls.count(("load", "sync")) == 1
    assert ("graph", "sync") not in calls
    state.compute.synchronize()


def test_prompt_tail_full_replay_drains_and_fences_before_pool_miss(monkeypatch):
    manager, pool, bindings, state, load, calls = make_pool(monkeypatch)
    manager.graph_streams.add(object())
    runner = SimpleNamespace(model=SimpleNamespace(_expert_offload_manager=manager))
    prepare_batch(runner, SimpleNamespace(has_prefill=True, decode_graph_eligible=True))
    submit_compute(pool, bindings[0], state)
    assert not state.copies[1].done

    def replay(desc):
        assert manager.has_prefill and not pool.inflight and state.copies[1].done
        assert pool.graph_residency_pending
        state.compute.submit(lambda: setattr(bindings[2], "slot_to_expert", (2,)))
        return desc

    graph = SimpleNamespace(
        device=bindings[0].device, _capture_descs={}, _candidates={}, capture=lambda: None, run_fullgraph=replay
    )
    manager.prepare_graph_capture(graph, lambda: True)
    assert graph.run_fullgraph("prompt-tail") == "prompt-tail"
    assert bindings[2].slot_to_expert == (0,)  # Replay callback is still pending.
    copy = mod.TorchExpertTransfer.copy_experts

    def checked_copy(*args, **kwargs):
        if args[0] == 2:
            assert kwargs["resident_loads"] == ((0, 2),), "planned from stale graph residency"
        return copy(*args, **kwargs)

    monkeypatch.setattr(mod.TorchExpertTransfer, "copy_experts", checked_copy)
    submit_compute(pool, bindings[2], state)  # Evict layer 0 from the first buffer.
    load.synchronize()
    state.compute.synchronize()
    assert calls.count(("graph", "sync")) == 1


def test_graph_decode_returns_to_prefetch_with_one_fence(monkeypatch):
    manager, pool, bindings, state, load, calls = make_pool(monkeypatch)
    manager.graph_streams.add(object())
    for _ in range(2):
        manager.has_prefill = False
        manager.run_fullgraph(lambda: None)
        manager.has_prefill = True
        for binding in bindings:
            submit_compute(pool, binding, state)
        load.synchronize()
        state.compute.synchronize()
    assert calls.count(("graph", "sync")) == 2
    assert sum(call[:2] == ("load", "copy") for call in calls) == 6


def test_failed_graph_fence_retries_before_reusing_cached_weights(monkeypatch):
    manager, pool, bindings, state, _, calls = make_pool(monkeypatch, layers=1)
    submit_compute(pool, bindings[0], state)
    manager.graph_streams.add(object())
    manager.has_prefill = False
    manager.run_fullgraph(lambda: None)
    manager.has_prefill = True
    synchronize = mod.torch.npu.synchronize

    def fail(device):
        raise RuntimeError("graph fence failed")

    monkeypatch.setattr(mod.torch.npu, "synchronize", fail)
    with pytest.raises(RuntimeError, match="graph fence failed"):
        pool.load_layer(bindings[0])
    assert pool.graph_residency_pending
    monkeypatch.setattr(mod.torch.npu, "synchronize", synchronize)
    submit_compute(pool, bindings[0], state)
    assert not pool.graph_residency_pending
    assert calls.count(("compute", "copy", 0)) == 1


def test_failed_prefetch_invalidates_buffer_and_can_retry(monkeypatch):
    manager, pool, bindings, state, load, _ = make_pool(monkeypatch)
    state.fail = 1
    with pytest.raises(RuntimeError, match="copy failed"):
        pool.load_layer(bindings[0])
    assert pool.owners[1] is None
    manager.has_prefill = False
    assert not pool.inflight
    manager.has_prefill = True
    state.fail = None
    for binding in bindings:
        submit_compute(pool, binding, state)
    load.synchronize()
    state.compute.synchronize()


def test_single_layer_keeps_two_buffers_without_prefetch(monkeypatch):
    _, pool, bindings, state, _, calls = make_pool(monkeypatch, layers=1)
    for _ in range(2):
        submit_compute(pool, bindings[0], state)
    state.compute.synchronize()
    assert calls.count(("compute", "copy", 0)) == 1
    assert not pool.inflight
    assert len(pool.buffers) == 2
    assert not any(call[:2] == ("load", "copy") for call in calls)


@pytest.mark.parametrize("layers", [1, 2, 4])
@pytest.mark.parametrize("device_type", ["cpu", "npu"])
def test_native_allocation_always_uses_two_buffers(monkeypatch, layers, device_type):
    _, _, bindings, _, load, _ = make_pool(monkeypatch, layers=layers)
    pool = mod.FullExpertPool()
    for binding in bindings:
        binding.device = SimpleNamespace(type=device_type)
        binding.adapter = SimpleNamespace(allocate_weights=lambda *args: {"layer": None})
    monkeypatch.setattr(pool, "_expert_layout", lambda _: None)
    monkeypatch.setattr(mod.torch.npu, "Stream", lambda **kwargs: load, raising=False)
    pool.allocate(bindings)
    assert len(pool.buffers) == 2
    assert len({id(buffer) for buffer in pool.buffers}) == 2
    assert (pool.load_stream is not None) == (device_type == "npu")
