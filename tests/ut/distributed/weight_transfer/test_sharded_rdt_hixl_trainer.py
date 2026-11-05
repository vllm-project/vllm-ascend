# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM 24420b7a341a tests/distributed/test_sharded_rdt_trainer.py.
"""Tests for the sharded-RDT trainer-side engine.

The trainer engine's own surface: init info and factory registration, the
send_weights round trip against an in-process fake producer server, the
ownership table it resolves and ships to the workers, and the producer's packed
destination-view cache.

The plan/replay half (including `RdtRouter`, which is consumer-side) lives in
test_sharded_rdt_plan.py; the producer server's protocol in
test_sharded_rdt_producer.py.
"""

from dataclasses import asdict
from functools import cached_property
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.weight_transfer import (
    ParamMeta,
    WeightSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.base import layerwise_groups
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
)

from vllm_ascend.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTHixlTrainerWeightTransferEngine as ShardedRDTTrainerWeightTransferEngine,
)


class RecordingClient:
    """A fake VLLMWeightSyncClient that records the order of calls."""

    def __init__(self):
        self.order: list[str] = []
        self.last_init_info: dict | None = None
        self.last_update_info: dict | None = None

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        self.order.append("init")
        self.last_init_info = init_info

    def start_weight_update(self) -> None:
        self.order.append("start")

    def update_weights(self, update_info: dict) -> None:
        self.order.append("update")
        self.last_update_info = update_info

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        self.order.append("finish")


class _ListSource(WeightSource):
    """A WeightSource over an explicit ordered (name, cpu-tensor) list, so the
    sharded-RDT group/order logic can be tested without a real model."""

    def __init__(self, pairs):
        self._pairs = list(pairs)

    @cached_property
    def metadata(self):
        return [ParamMeta(n, t.dtype, tuple(t.shape)) for n, t in self._pairs]

    def iter_full(self):
        yield from self._pairs


class _FakeProducerServer:
    """In-process stand-in for the _RDTProducerServer Ray actor. Records the
    engine->server call sequence and, by default, frees each group as soon as
    it is published (simulating the consumers' free_group barrier) so the
    gather loop's credit gate never parks. Mirrors the real server's
    per-group barrier: publish/free are keyed by GROUP INDEX, signals count
    to the ``begin_sync`` live total, publish_group returns nothing, and freed
    groups flow back only through wait_freed / end_sync (see
    test_sharded_rdt_producer.TestFakeServerAgreesWithTheRealOne)."""

    def __init__(self, auto_free=True):
        self.order: list[str] = []
        self.published: list[int] = []
        self.live_count = 1
        self._inflight_groups: list[int] = []
        self.auto_free = auto_free
        self.free_counts: dict[int, int] = {}
        self._pending_freed: list[int] = []

    # Alias for tests that inspect in-flight groups.
    @property
    def inflight(self):
        return self._inflight_groups

    def begin_sync(self, live_count, live_consumer_ids=None):
        self.order.append("begin")
        self.live_count = max(1, int(live_count))
        self.live_ids = live_consumer_ids

    def publish_group(self, group_idx, entries):
        self.order.append("publish")
        self.published.append(group_idx)
        self._inflight_groups.append(group_idx)
        if self.auto_free or self.free_counts.get(group_idx, 0) >= self.live_count:
            self._inflight_groups.remove(group_idx)
            self._pending_freed.append(group_idx)

    def free_group(self, group_idx):
        """Consumer back-edge; may arrive before the group's publish."""
        self.free_counts[group_idx] = self.free_counts.get(group_idx, 0) + 1
        if group_idx in self._inflight_groups and self.free_counts[group_idx] >= self.live_count:
            self._inflight_groups.remove(group_idx)
            self._pending_freed.append(group_idx)

    def wait_freed(self):
        """The engine's credit gate. The real server blocks here; the fake
        must already have a banked credit when the gate asks (auto_free, or a
        test's own free_group) — anything else is the deadlock the real
        watchdog would kill, so fail loudly. Not appended to ``order``: it is
        pacing, not a lifecycle milestone."""
        assert self._pending_freed, "wait_freed with nothing freed: the gather loop would deadlock"
        freed = self._pending_freed
        self._pending_freed = []
        return freed

    def end_sync(self):
        self.order.append("end")
        freed = self._pending_freed
        self._pending_freed = []
        return freed

    def set_gather_error(self, message):
        self.order.append("error")


def _rdt_engine_with_fake_server(source, *, is_sender, client, server, monkeypatch, world=1):
    """Build a ShardedRDTTrainerWeightTransferEngine wired to an in-process fake
    server (no Ray, no NPU IPC): bypass trainer_init's spawn, set the
    group-major metadata, and route _rpc to the fake."""
    import vllm_ascend.distributed.weight_transfer.sharded_rdt_trainer as mod

    # reduce_tensor needs NPU; the fake server never rebuilds, so stub it.
    monkeypatch.setattr(mod, "_reduce_npu_tensor", lambda t: (None, ("fake",)))
    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self)
    monkeypatch.setattr(torch.npu, "current_stream", lambda *a: SimpleNamespace(synchronize=lambda: None))

    init_info = ShardedRDTTrainerInitInfo(num_consumers=1, rank=0 if is_sender else 1)
    engine = ShardedRDTTrainerWeightTransferEngine(
        client=client, source=source, is_sender=is_sender, init_info=init_info
    )
    engine._meta = list(source.metadata)
    names = [m.name for m in engine._meta]
    engine._groups = layerwise_groups(names)
    engine._server = server
    engine._rpc = lambda method, *args: getattr(server, method)(*args)
    engine._resolve_ownership(world, init_info.rank)
    return engine


def _rdt_source_two_layers():
    return _ListSource(
        [
            ("embed.weight", torch.zeros(2)),
            ("model.layers.0.w", torch.zeros(2)),
            ("model.layers.1.w", torch.zeros(2)),
            ("norm.weight", torch.zeros(2)),
        ]
    )


class TestShardedRDTTrainerInitInfo:
    def test_declares_backend(self):
        assert ShardedRDTTrainerInitInfo.backend == "sharded_rdt"

    def test_rank_is_keyword_only_and_drives_is_sender(self):
        assert ShardedRDTTrainerInitInfo(num_consumers=4, rank=0).is_sender is True
        assert ShardedRDTTrainerInitInfo(num_consumers=4, rank=1).is_sender is False
        with pytest.raises(TypeError):
            # rank is keyword-only.
            ShardedRDTTrainerInitInfo(4, 0)  # type: ignore[misc]

    def test_registered_in_trainer_factory(self, monkeypatch):
        from vllm_ascend.distributed.weight_transfer import _register_sharded_rdt

        monkeypatch.setattr(WeightTransferTrainerFactory, "_registry", dict(WeightTransferTrainerFactory._registry))
        from vllm.distributed.weight_transfer import WeightTransferEngineFactory

        monkeypatch.setattr(WeightTransferEngineFactory, "_registry", dict(WeightTransferEngineFactory._registry))
        from vllm.platforms import current_platform

        monkeypatch.setattr(current_platform, "device_type", "npu")
        _register_sharded_rdt()
        cls = WeightTransferTrainerFactory._registry["sharded_rdt"]()
        assert cls is ShardedRDTTrainerWeightTransferEngine


def test_sharded_rdt_trainer_init_requires_source():
    with pytest.raises(ValueError, match="requires a WeightSource"):
        ShardedRDTTrainerWeightTransferEngine.trainer_init(
            ShardedRDTTrainerInitInfo(num_consumers=1, rank=0),
            client=RecordingClient(),
            source=None,
        )


def test_sharded_rdt_worker_init_info_is_group_major(monkeypatch):
    source = _rdt_source_two_layers()
    engine = _rdt_engine_with_fake_server(
        source,
        is_sender=True,
        client=RecordingClient(),
        server=_FakeProducerServer(),
        monkeypatch=monkeypatch,
    )
    worker_init = engine._build_worker_init_info(["srv_rk0"])
    # 4 params -> pre / layer0 / layer1 / post = 4 groups of length 1.
    assert worker_init.names == [
        "embed.weight",
        "model.layers.0.w",
        "model.layers.1.w",
        "norm.weight",
    ]
    assert worker_init.group_lens == [1, 1, 1, 1]
    assert worker_init.trainer_actor_names == ["srv_rk0"]
    assert worker_init.produce_method_name == "rdt_produce_weights_batched"
    assert sum(worker_init.group_lens) == len(worker_init.names)


def test_sharded_rdt_send_weights_drives_client_in_order(monkeypatch):
    server = _FakeProducerServer(auto_free=True)
    client = RecordingClient()
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=True,
        client=client,
        server=server,
        monkeypatch=monkeypatch,
    )
    engine.send_weights()

    assert client.order == ["start", "update", "finish"]
    # begin, one publish per group (4), end.
    assert server.order == ["begin", "publish", "publish", "publish", "publish", "end"]
    assert len(server.published) == 4
    # every group freed -> no engine-held refs remain.
    assert engine._inflight == {}


def test_sharded_rdt_send_weights_group_order_mismatch_raises(monkeypatch):
    # Source whose iteration order disagrees with its metadata order.
    class _BadSource(_ListSource):
        def iter_full(self):
            reordered = list(self._pairs)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            return iter(reordered)

    server = _FakeProducerServer()
    engine = _rdt_engine_with_fake_server(
        _BadSource(_rdt_source_two_layers()._pairs),
        is_sender=True,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(RuntimeError, match="must yield the owned names"):
        engine.send_weights()
    assert "error" in server.order  # gather error propagated to the server


def test_sharded_rdt_non_sender_skips_client(monkeypatch):
    class _RaisingClient(RecordingClient):
        def start_weight_update(self):
            raise AssertionError("non-sender must not touch the client")

        def update_weights(self, update_info):
            raise AssertionError("non-sender must not touch the client")

        def finish_weight_update(self, weight_version: str | None = None):
            raise AssertionError("non-sender must not touch the client")

    server = _FakeProducerServer(auto_free=True)
    client = _RaisingClient()
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=False,
        client=client,
        server=server,
        monkeypatch=monkeypatch,
    )
    engine.send_weights()  # gathers only; must not raise
    assert client.order == []
    assert server.order == ["begin", "publish", "publish", "publish", "publish", "end"]


def test_sharded_rdt_send_weights_surfaces_update_error(monkeypatch):
    class _FailingUpdateClient(RecordingClient):
        def update_weights(self, update_info):
            self.order.append("update")
            raise RuntimeError("inference side rejected update")

    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=True,
        client=_FailingUpdateClient(),
        server=server,
        monkeypatch=monkeypatch,
    )
    with pytest.raises(RuntimeError, match="inference side rejected update"):
        engine.send_weights()


class _DeclaredSource(_ListSource):
    """A source that declares owners per name and yields only what this rank
    owns."""

    def __init__(self, pairs, owners, rank):
        super().__init__(pairs)
        self._owners = owners
        self._rank = rank

    @cached_property
    def metadata(self):
        return [ParamMeta(n, t.dtype, tuple(t.shape), owners=self._owners.get(n)) for n, t in self._pairs]

    def iter_full(self):
        for n, t in self._pairs:
            if n not in self._owners or self._rank in self._owners[n]:
                yield n, t


# embed and layer 0 on stage 0, layer 1 on stage 1, norm everywhere.
_PP_OWNERS = {
    "embed.weight": (0,),
    "model.layers.0.w": (0,),
    "model.layers.1.w": (1,),
}


def test_sharded_rdt_plans_ownership_from_metadata(monkeypatch):
    """Owners come from metadata alone, so every rank builds the same table."""
    owned = [
        {"embed.weight", "model.layers.0.w", "norm.weight"},
        {"model.layers.1.w", "norm.weight"},
    ]
    tables = []
    for rank in (0, 1):
        engine = _rdt_engine_with_fake_server(
            _DeclaredSource(_rdt_source_two_layers()._pairs, _PP_OWNERS, rank),
            is_sender=rank == 0,
            client=RecordingClient(),
            server=_FakeProducerServer(),
            monkeypatch=monkeypatch,
            world=2,
        )
        tables.append((engine._owner_sets, engine._name_owner_class))
        assert engine._owned_names == owned[rank]
    assert tables[0] == tables[1] == ([[0], [1], [0, 1]], [0, 0, 1, 2])


def test_sharded_rdt_publishes_only_the_groups_it_owns(monkeypatch):
    """The engine groups the source's stream per layer and publishes only the
    groups this rank owns part of."""
    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _DeclaredSource(_rdt_source_two_layers()._pairs, _PP_OWNERS, 1),
        is_sender=False,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
        world=2,
    )
    engine.send_weights()
    assert server.published == [2, 3]
    assert server.order == ["begin", "publish", "publish", "end"]


def test_sharded_rdt_rejects_owners_outside_the_trainer(monkeypatch):
    with pytest.raises(ValueError, match="only 2 trainer ranks"):
        _rdt_engine_with_fake_server(
            _DeclaredSource(_rdt_source_two_layers()._pairs, {"embed.weight": (0, 2)}, 0),
            is_sender=True,
            client=RecordingClient(),
            server=_FakeProducerServer(),
            monkeypatch=monkeypatch,
            world=2,
        )


def test_sharded_rdt_begin_sync_carries_the_live_count(monkeypatch):
    """The free barrier's target is one integer per sync: the live consumer
    count, defaulting to the whole provisioned fleet."""
    server = _FakeProducerServer(auto_free=True)
    engine = _rdt_engine_with_fake_server(
        _rdt_source_two_layers(),
        is_sender=False,
        client=RecordingClient(),
        server=server,
        monkeypatch=monkeypatch,
    )
    engine.send_weights()
    assert server.live_count == 1  # num_consumers=1 in the harness
    engine.send_weights(live_consumer_ids=[0])
    assert server.live_count == 1


class TestLiveCountPlumbing:
    """``send_weights(live_consumer_ids)`` -> the barrier target. The
    provisioned geometry is frozen; a degraded sync only lowers the target and
    narrows each slot-sharing group's rendezvous."""

    @staticmethod
    def _engine(num_consumers, world=2, rank=0):
        engine = ShardedRDTTrainerWeightTransferEngine.__new__(ShardedRDTTrainerWeightTransferEngine)
        engine._init_info = ShardedRDTTrainerInitInfo(rank=rank, num_consumers=num_consumers)
        engine.source = object()  # send_weights asserts a source is present
        received: list = []
        engine._send_weights_inner = lambda count, ids: received.append((count, ids))
        return engine, received

    def test_none_counts_the_whole_provisioned_fleet(self):
        engine, got = self._engine(num_consumers=8)
        engine.send_weights(None)
        assert got == [(8, list(range(8)))]

    def test_a_live_set_counts_its_distinct_members(self):
        engine, got = self._engine(num_consumers=8)
        engine.send_weights([0, 1, 4, 5, 5])
        assert got == [(4, [0, 1, 4, 5])]

    def test_a_full_live_set_matches_the_provisioned_count(self):
        engine, got = self._engine(num_consumers=8)
        engine.send_weights(list(range(8)))
        engine.send_weights(None)
        assert got == [(8, list(range(8)))] * 2

    def test_which_consumers_died_does_not_matter_only_how_many(self):
        """The whole point of the barrier: no routed per-producer targets, so
        the identity of the dead consumer is irrelevant to the producers."""
        counts = []
        for live in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]):
            engine, got = self._engine(num_consumers=8)
            engine.send_weights(live)
            counts += [c for c, _ids in got]
        assert counts == [4, 4, 4]

    def test_the_live_ids_travel_with_the_count(self):
        """The count sizes the free barrier, the ids size the slot-sharing
        rendezvous, so a producer that shares slots can tell WHICH consumers it
        is still waiting for. They must describe the same set."""
        engine, got = self._engine(num_consumers=8)
        engine.send_weights([6, 0, 2, 2])
        ((count, ids),) = got
        assert ids == [0, 2, 6] and count == len(ids)


def test_sharded_rdt_worker_init_info_carries_the_ownership_table(monkeypatch):
    import json

    engine = _rdt_engine_with_fake_server(
        _DeclaredSource(_rdt_source_two_layers()._pairs, _PP_OWNERS, 0),
        is_sender=True,
        client=RecordingClient(),
        server=_FakeProducerServer(auto_free=True),
        monkeypatch=monkeypatch,
        world=2,
    )
    worker_init = engine._build_worker_init_info(["srv_rk0", "srv_rk1"])
    # The payload crosses the control plane as JSON: nested lists must survive.
    round_tripped = json.loads(json.dumps(asdict(worker_init)))
    assert round_tripped["owner_sets"] == [[0], [1], [0, 1]]
    assert round_tripped["name_owner_class"] == [0, 0, 1, 2]


def test_sharded_rdt_rejects_metadata_disagreement_across_ranks(monkeypatch):
    """Each rank plans ownership from its own metadata, so a rank whose owners
    differ must fail at init rather than serve a plan the consumers don't route
    by. The digest rides on the server-name all-gather."""
    engine = _rdt_engine_with_fake_server(
        _DeclaredSource(_rdt_source_two_layers()._pairs, _PP_OWNERS, 0),
        is_sender=True,
        client=RecordingClient(),
        server=_FakeProducerServer(),
        monkeypatch=monkeypatch,
        world=2,
    )
    engine._server_name = "srv_rk0"

    def _fake_all_gather(out, mine):
        out[:] = [mine, ("srv_rk1", "deadbeefdeadbeef")]

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_gather_object", _fake_all_gather)
    with pytest.raises(ValueError, match="disagrees across trainer ranks"):
        engine._all_gather_server_names(2, 0)
