# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41.engram.executor import EngramMultiStreamExecutor


class _Stream:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def wait_event(self, event):
        self.calls.append((self.name, "wait", event.name))

    def synchronize(self):
        self.calls.append((self.name, "synchronize"))


class _Event:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def record(self, stream):
        self.calls.append((stream.name, "record", self.name))


@pytest.fixture
def executor_env(monkeypatch):
    calls = []
    main = _Stream("main", calls)
    current = main
    stream_names = iter(("engram",))
    names = iter(("inputs", "ready-1", "ready-14"))

    @contextmanager
    def stream_context(stream):
        nonlocal current
        previous, current = current, stream
        try:
            yield
        finally:
            current = previous

    monkeypatch.setattr(torch.npu, "Stream", lambda: _Stream(next(stream_names), calls))
    monkeypatch.setattr(torch.npu, "Event", lambda: _Event(next(names), calls))
    monkeypatch.setattr(torch.npu, "current_stream", lambda: current)
    monkeypatch.setattr(torch.npu, "stream", stream_context)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda tensor, stream: calls.append(("tensor", "use", stream.name))
    )

    def hash_inputs(tokens, **kwargs):
        calls.append((current.name, "hash"))
        return tokens[:, None, None].expand(-1, 2, 1), tokens >= 0

    executor = EngramMultiStreamExecutor(
        (1, 14),
        hash_fn=SimpleNamespace(forward_packed=hash_inputs),
        engram_layers={layer: _Engram(layer, calls) for layer in (1, 14)},
    )
    return executor, calls


class _Engram:
    def __init__(self, layer_id, calls):
        self.layer_id = layer_id
        self.layer_hash_index = (1, 14).index(layer_id)
        self.calls = calls

        self.pending = None

    def start_prefetch(self, ids):
        assert self.pending is None
        self.calls.append((torch.npu.current_stream().name, "start", self.layer_id))
        self.pending = SimpleNamespace(ids=ids)
        return self.pending

    def finish_prefetch(self, handle):
        assert handle is self.pending
        self.calls.append((torch.npu.current_stream().name, "finish", self.layer_id))
        self.pending = None
        return handle.ids.float() + self.layer_id

    def precompute_kv(self, embedding):
        self.calls.append((torch.npu.current_stream().name, "wkv", self.layer_id))
        return embedding * 2, embedding * 3, torch.ones(1)

    def apply_precomputed(self, hidden, key, value, weight, mask):
        self.calls.append((torch.npu.current_stream().name, "gate", self.layer_id))
        return hidden + (key + value) * weight * mask[:, None]


def test_serial_side_stream_joins_only_at_consumers_and_tracks_tensor_lifetimes(executor_env):
    executor, calls = executor_env
    early, late = (executor.engram_layers[layer] for layer in (1, 14))
    for tokens in (torch.tensor([1, -1, 3]), torch.tensor([7, 4, -1])):
        calls.clear()
        hashes, mask = executor.begin(tokens)
        calls.append(("main", "layer-0"))
        early_kv = executor.project(early, hashes)
        hidden = executor.consume(early, torch.zeros(3, 1), *early_kv, mask)
        calls.append(("main", "layer-1"))
        late_kv = executor.project(late, hashes)
        calls.append(("main", "layers-2-13"))
        hidden = executor.consume(late, hidden, *late_kv, mask)
        executor.release()
        expected = (tokens[:, None].float() * 2 + 15) * 5 * mask[:, None]
        torch.testing.assert_close(hidden, expected)
        stages = [call for call in calls if call[1] != "use"]
        assert stages == [
            ("main", "record", "inputs"),
            ("engram", "wait", "inputs"),
            ("engram", "hash"),
            ("engram", "start", 1),
            ("main", "layer-0"),
            ("engram", "finish", 1),
            ("engram", "wkv", 1),
            ("engram", "record", "ready-1"),
            ("engram", "start", 14),
            ("main", "wait", "ready-1"),
            ("main", "gate", 1),
            ("main", "layer-1"),
            ("engram", "finish", 14),
            ("engram", "wkv", 14),
            ("engram", "record", "ready-14"),
            ("main", "layers-2-13"),
            ("main", "wait", "ready-14"),
            ("main", "gate", 14),
        ]
        # K/V/weight/mask cross to main; hash never crosses to another side stream.
        assert calls.count(("tensor", "use", "main")) == 8
        assert calls.count(("tensor", "use", "engram")) == 1
        assert not executor._pending_fetches
        assert early.pending is late.pending is None

        # Reconstruct device dependencies from stream FIFO and event edges.
        # Serial L14 work must not become a predecessor of the L1 consumer.
        ancestors, stream_tail, records = {}, {}, {}
        for index, (stream, action, *detail) in enumerate(stages):
            parents = [stream_tail[stream]] if stream in stream_tail else []
            if action == "wait":
                parents.append(records[detail[0]])
            ancestors[index] = set(parents)
            for parent in parents:
                ancestors[index].update(ancestors[parent])
            stream_tail[stream] = index
            if action == "record":
                records[detail[0]] = index
        hash_op = stages.index(("engram", "hash"))
        late_start = stages.index(("engram", "start", 14))
        assert hash_op in ancestors[late_start]
        assert records["ready-1"] in ancestors[late_start]
        assert late_start not in ancestors[records["ready-1"]]
        early_gate = stages.index(("main", "gate", 1))
        assert late_start not in ancestors[early_gate]
        assert early_gate not in ancestors[late_start]
        assert early_gate not in ancestors[records["ready-14"]]
        for layer in (1, 14):
            gate = stages.index(("main", "gate", layer))
            wkv = stages.index(("engram", "wkv", layer))
            assert wkv in ancestors[records[f"ready-{layer}"]]
            assert wkv in ancestors[gate]
            assert records[f"ready-{layer}"] in ancestors[gate]


@pytest.mark.parametrize("failure", ["wkv", "gate"])
def test_failure_blocks_reuse_and_abort_drains_pending_fetch(executor_env, failure):
    executor, calls = executor_env
    engram = executor.engram_layers[1]
    hashes, mask = executor.begin(torch.ones(2))
    method = "precompute_kv" if failure == "wkv" else "apply_precomputed"
    setattr(engram, method, lambda *args: (_ for _ in ()).throw(RuntimeError("Engram failed")))
    with pytest.raises(RuntimeError, match="Engram failed"):
        kv = executor.project(engram, hashes)
        executor.consume(engram, torch.zeros(2, 1), *kv, mask)
    with pytest.raises(RuntimeError, match="closed or failed"):
        executor.begin(torch.ones(2))
    executor.abort()
    assert calls.count(("engram", "finish", 14)) == (1 if failure == "gate" else 0)
    assert ("engram", "wkv", 14) not in calls
    assert executor.engram_layers[14].pending is None
    assert not executor._pending_fetches
    assert calls[-2:] == [("engram", "synchronize"), ("main", "synchronize")]
    executor.close()
    before = len(calls)
    executor.close()
    assert len(calls) == before


def test_abort_attempts_main_drain_even_if_side_drain_fails(executor_env):
    executor, calls = executor_env
    executor._stream.synchronize = lambda: (_ for _ in ()).throw(RuntimeError("side failed"))
    with pytest.raises(RuntimeError, match="Failed to drain Engram streams"):
        executor.abort()
    assert calls[-1] == ("main", "synchronize")


def test_unknown_consumer_fails_before_submission(executor_env):
    executor, calls = executor_env
    with pytest.raises(ValueError, match="Unknown Engram layer"):
        executor.wait(9)
    assert not calls


def test_one_layer_needs_no_hash_fork_event(executor_env, monkeypatch):
    _, calls = executor_env
    names = iter(("inputs", "ready-1"))
    monkeypatch.setattr(torch.npu, "Stream", lambda: _Stream("engram", calls))
    monkeypatch.setattr(torch.npu, "Event", lambda: _Event(next(names), calls))
    early = _Engram(1, calls)
    executor = EngramMultiStreamExecutor(
        (1,),
        hash_fn=SimpleNamespace(forward_packed=lambda ids, **kwargs: (ids[:, None, None], ids >= 0)),
        engram_layers={1: early},
    )
    hashes, mask = executor.begin(torch.tensor([1, 2]))
    kv = executor.project(early, hashes)
    executor.consume(early, torch.zeros(2, 1), *kv, mask)
    executor.release()
    assert not executor.next_layer_ids
    assert [call for call in calls if call[1] == "wait"] == [
        ("engram", "wait", "inputs"),
        ("main", "wait", "ready-1"),
    ]


def test_late_launch_failure_does_not_repeat_completed_early_fetch(executor_env):
    executor, calls = executor_env
    executor.engram_layers[14].start_prefetch = lambda _: (_ for _ in ()).throw(RuntimeError("launch failed"))
    hashes, _ = executor.begin(torch.ones(2))
    with pytest.raises(RuntimeError, match="launch failed"):
        executor.project(executor.engram_layers[1], hashes)
    executor.abort()
    assert calls.count(("engram", "finish", 1)) == 1
    assert not executor._pending_fetches
    assert executor.engram_layers[1].pending is None
    assert calls.count(("engram", "wkv", 1)) == 1


@pytest.mark.parametrize("failed_layer", [1, 14])
def test_abort_drains_streams_even_when_pending_wait_fails(executor_env, failed_layer):
    executor, calls = executor_env
    hashes, _ = executor.begin(torch.ones(2))
    if failed_layer == 14:
        executor.project(executor.engram_layers[1], hashes)
    executor.engram_layers[failed_layer].finish_prefetch = lambda _: (_ for _ in ()).throw(RuntimeError("wait failed"))
    with pytest.raises(RuntimeError, match="Failed to drain Engram streams") as exc:
        executor.abort()
    assert str(exc.value.__cause__) == "wait failed"
    assert not executor._pending_fetches
    assert calls[-2:] == [("engram", "synchronize"), ("main", "synchronize")]


def test_unfinished_forward_cannot_be_released_or_overwritten(executor_env):
    executor, calls = executor_env
    hashes, _ = executor.begin(torch.ones(2))
    before = len(calls)
    with pytest.raises(RuntimeError, match="unfinished fetches"):
        executor.release()
    with pytest.raises(RuntimeError, match="unfinished fetches"):
        executor.begin(torch.ones(2))
    with pytest.raises(ValueError, match="does not own"):
        executor.project(_Engram(1, calls), hashes)
    assert len(calls) == before
    with pytest.raises(RuntimeError, match="No pending"):
        executor.project(executor.engram_layers[14], hashes)
    executor.project(executor.engram_layers[1], hashes)
    with pytest.raises(RuntimeError, match="No pending"):
        executor.project(executor.engram_layers[1], hashes)
    executor.abort()
