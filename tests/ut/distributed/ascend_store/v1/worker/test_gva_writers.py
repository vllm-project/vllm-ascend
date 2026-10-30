"""Shared GVA publication through prototype services and real CPU byte copies."""

from __future__ import annotations

import ctypes
import threading
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import FakeEvent, FakeLoadStartGate, make_coordinator, make_topology
from tests.ut.distributed.ascend_store.v1.native_copy_fixtures import ByteCopySDK, native_backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    GVA_SHARD_ALIGNMENT_BYTES,
    resolve_backend_spec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache import create_key_builder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import Resources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.topology import ContextRank
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import Worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import worker as worker_module


class ThreadCollective:
    def __init__(self, rank, ranks, barrier, outcomes):
        self.rank, self.ranks = rank, ranks
        self.barrier, self.outcomes = barrier, outcomes
        self.entered = threading.Event()
        self.calls = []
        self.error = None
        self.closed = False

    def all_gather(self, value):
        self.calls.append(value)
        self.outcomes[self.rank] = value
        self.entered.set()
        self.barrier.wait(timeout=5)
        result = dict(self.outcomes)
        self.barrier.wait(timeout=5)
        if self.error is not None:
            raise self.error
        return result

    def close(self):
        self.closed = True


@contextmanager
def cluster(tp=2, pcp=1, dcp=2, put_step=2, *, separate_allocations=False):
    ranks = (8, 13, 21, 34)[: tp * pcp]
    members = tuple(
        ContextRank(rank, index % tp, ((index % tp) * pcp + index // tp) % dcp) for index, rank in enumerate(ranks)
    )
    objects, readable, outcomes = {}, set(), {}
    barrier = threading.Barrier(len(ranks))
    workers, resources, sdks, channels, topologies, sources = [], [], [], [], [], []
    try:
        for index, member in enumerate(members):
            topology = make_topology(
                physical_layers=(3, 4),
                tp_rank=member.tp_rank,
                tp_size=tp,
                put_step=put_step,
                head_or_tp_rank=member.tp_rank // put_step,
                dcp_rank=member.dcp_rank,
                dcp_size=dcp,
                pcp_rank=index // tp,
                pcp_size=pcp,
                context_ranks=members,
                global_rank=member.global_rank,
                pp_size=2,
                pp_rank=1,
            )
            shared = dcp > 1 and (put_step > 1 or pcp > 1)
            topology = replace(
                topology,
                config=replace(
                    topology.config,
                    pp_partition=replace(topology.config.pp_partition, global_layer_count=7),
                    cp_partition=replace(
                        topology.config.cp_partition,
                        object_shard_count=dcp if shared else 1,
                        shard_alignment_bytes=GVA_SHARD_ALIGNMENT_BYTES if shared else 1,
                    ),
                ),
                object_shard_index=member.dcp_rank if shared else 0,
                stage_layer_start=3,
            )
            sdk = ByteCopySDK({} if separate_allocations else objects, readable)
            backend = native_backend("memcache", sdk)
            owner = Resources(backend, resolve_backend_spec("memcache"), topology.transfer_groups)
            channel = ThreadCollective(member.global_rank, ranks, barrier, outcomes) if shared else None
            caches = {
                layer.layer_names[0]: torch.arange(64, dtype=torch.float32).reshape(8, 4, 2, 1)
                + member.dcp_rank * 100
                + member.tp_rank // put_step * 1000
                + layer.physical_layer_id * 10
                for layer in topology.groups[0].layers
            }
            worker = Worker(
                topology,
                owner,
                coordinator=make_coordinator(topology),
                layerwise=True,
                source_ready_event_factory=FakeEvent,
                start_gate_factory=lambda: FakeLoadStartGate(opened=True),
                key_builder=create_key_builder(None, None, topology.config),
                collective=channel,
            )
            worker.bind_kv_caches(caches)
            workers.append(worker)
            resources.append(owner)
            sdks.append(sdk)
            channels.append(channel)
            topologies.append(topology)
            sources.append(caches)
        yield SimpleNamespace(
            workers=workers,
            resources=resources,
            sdks=sdks,
            channels=channels,
            topologies=topologies,
            sources=sources,
            objects=objects,
            readable=readable,
        )
    finally:
        # These CPU SDK calls are synchronous. After every executor joins, test
        # teardown can confirm disposal that the real Memcache SDK cannot prove.
        for worker, owner in zip(workers, resources, strict=True):
            worker.transfer.end_step(wait=False)
            owner.stop_executors()
            owner.backend.revoke_gva = lambda keys: (0,) * len(keys)
            owner.close()


def submit_stores(group, *, skip_layer_rank=None, failed_hook_rank=None):
    for index, worker in enumerate(group.workers):
        commands = (RangeStoreCommand("store", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17),)
        worker.begin_step(KVTransferStep(store=StoreCommandBatch(commands)))
        for layer in (3, 4):
            if index == failed_hook_rank:
                with pytest.raises(RuntimeError, match="source_record"):
                    worker.save_layer(f"layers.{layer}.group.0")
                break
            if index != skip_layer_rank or layer != 4:
                worker.save_layer(f"layers.{layer}.group.0")
        worker.end_step()  # Seal without waiting for peers that have not entered yet.
        if index == 0 and group.channels[0] is not None:
            assert group.channels[0].entered.wait(timeout=3)
            assert not group.readable
            assert not any(call[0] == "publish" for sdk in group.sdks for call in sdk.calls)


@pytest.mark.parametrize(
    "tp,pcp,dcp,put_step,writers",
    (
        (2, 1, 2, 2, {8, 13}),
        (4, 1, 2, 4, {8, 13}),
        (2, 2, 2, 2, {8, 21}),
        (2, 2, 4, 2, {8, 13, 21, 34}),
        (2, 2, 2, 1, {8, 13, 21, 34}),
        (2, 2, 4, 1, {8, 13, 21, 34}),
    ),
)
def test_every_required_writer_fills_both_keys_before_publication_and_loads_its_own_shard(
    tp, pcp, dcp, put_step, writers
):
    with cluster(tp, pcp, dcp, put_step) as group:
        expected = [{name: cache[[1, 2]].clone() for name, cache in sources.items()} for sources in group.sources]
        submit_stores(group)
        results = [worker.fence_previous_store() for worker in group.workers]
        assert all(result.succeeded and result.source_release_confirmed for batch in results for result in batch)
        assert all(worker.take_released_store_job_ids() == {17} for worker in group.workers)
        for topology, channel, sdk, sources in zip(
            group.topologies, group.channels, group.sdks, group.sources, strict=True
        ):
            assert len(channel.calls) == 1
            allocations = [call for call in sdk.calls if call[0] == "allocate"]
            assert bool(allocations) is (topology.global_rank in writers)
            if not allocations:
                assert not channel.calls[0]
                continue
            keys = allocations[0][1]
            assert len(keys) == 2 and set(keys) <= group.readable
            for block, key in zip((1, 2), keys, strict=True):
                payload = b"".join(cache[block].numpy().tobytes() for cache in sources.values())
                offset = topology.object_shard_index * GVA_SHARD_ALIGNMENT_BYTES + 3 * 32
                assert group.objects[key].raw[offset : offset + len(payload)] == payload
                assert channel.calls[0][key].address == ctypes.addressof(group.objects[key])
                assert channel.calls[0][key].size == ctypes.sizeof(group.objects[key])
        for worker, sources, saved in zip(group.workers, group.sources, expected, strict=True):
            for cache in sources.values():
                cache[4:7].fill_(-999)
            worker.begin_step(KVTransferStep((LoadCommand("load", 0, 8, ((5, 6),), (b"a", b"b")),)))
            worker.start_load()
            for layer in (3, 4):
                worker.wait_for_layer_load(f"layers.{layer}.group.0")
            assert not worker.collect_load_result().failed_locations
            worker.end_step()
            for name, cache in sources.items():
                torch.testing.assert_close(cache[[5, 6]], saved[name])
                assert torch.all(cache[4] == -999)
        assert all(len(channel.calls) == 1 for channel in group.channels)
        for worker, sdk, channel in zip(group.workers, group.sdks, group.channels, strict=True):
            worker.close()
            assert sdk.closed and channel.closed


@pytest.mark.parametrize(
    "failure",
    (
        "candidate",
        "invalid_range",
        "exists",
        "open_exception",
        "regions_exception",
        "missing_identity",
        "source_wait",
        "source_record",
        "open_code",
        "filtered",
        "copy_code",
        "copy_exception",
        "missing_layer",
        "regions",
        "malformed",
    ),
)
def test_failed_or_empty_writer_still_joins_and_prevents_partial_publication(failure, monkeypatch):
    with cluster(separate_allocations=failure == "regions") as group:
        peer, sdk = group.workers[1], group.sdks[1]
        error = RuntimeError(f"failed at {failure}")
        if failure == "candidate":
            monkeypatch.setattr(peer.store, "_candidates", Mock(side_effect=error))
        elif failure == "invalid_range":
            candidates = peer.store._candidates

            def invalid_range(commands):
                return candidates(tuple(replace(command, start_token=command.end_token + 1) for command in commands))

            monkeypatch.setattr(peer.store, "_candidates", invalid_range)
        elif failure == "exists":
            monkeypatch.setattr(peer.transfer.backend, "exists", Mock(side_effect=error))
        elif failure == "filtered":
            monkeypatch.setattr(peer.transfer.backend, "exists", lambda keys: (True,) * len(keys))
        elif failure == "open_exception":
            monkeypatch.setattr(sdk, "batch_alloc", Mock(side_effect=error))
        elif failure == "open_code":
            monkeypatch.setattr(sdk, "batch_alloc", lambda keys, _sizes: [0] * len(keys))
        elif failure == "regions_exception":
            monkeypatch.setattr(peer.transfer, "_store_regions", Mock(side_effect=error))
        elif failure == "missing_identity":
            opening = peer.transfer._open_read

            def omit_identity(batch, prepared):
                opening(batch, prepared)
                batch.identities = {}

            monkeypatch.setattr(peer.transfer, "_open_read", omit_identity)
        elif failure in ("source_wait", "source_record"):
            event = FakeEvent()
            monkeypatch.setattr(event, "synchronize" if failure == "source_wait" else "record", Mock(side_effect=error))
            peer._source_ready_event_factory = lambda: event
        elif failure == "copy_code":
            monkeypatch.setattr(sdk, "batch_copy", lambda *_args: -7)
        elif failure == "copy_exception":
            monkeypatch.setattr(sdk, "batch_copy", Mock(side_effect=error))
        elif failure == "malformed":
            copy = peer.transfer.copy

            def malformed(*args, **kwargs):
                copy(*args, **kwargs)
                return ()

            monkeypatch.setattr(peer.transfer, "copy", malformed)
        submit_stores(
            group,
            skip_layer_rank=1 if failure == "missing_layer" else None,
            failed_hook_rank=1 if failure == "source_record" else None,
        )
        (result,) = group.workers[0].fence_previous_store()
        assert not result.succeeded and result.result_codes == (None, None) and result.source_release_confirmed
        assert group.workers[0].take_released_store_job_ids() == {17}
        if failure in ("copy_exception", "malformed"):
            with pytest.raises(RuntimeError if failure == "copy_exception" else ValueError):
                peer.fence_previous_store()
            assert not peer.take_released_store_job_ids()
            with pytest.raises(RuntimeError if failure == "copy_exception" else ValueError):
                peer.close()
            assert not sdk.closed
        else:
            (result,) = peer.fence_previous_store()
            assert result.source_release_confirmed
            assert result.succeeded is (failure == "filtered")
            assert peer.take_released_store_job_ids() == {17}
        assert all(len(channel.calls) == 1 for channel in group.channels)
        assert not group.readable
        assert not any(call[0] == "publish" for sdk in group.sdks for call in sdk.calls)


def test_collective_failure_reports_source_release_then_blocks_new_work():
    with cluster() as group:
        error = RuntimeError("Store channel failed")
        for channel in group.channels:
            channel.error = error
        submit_stores(group)
        for worker in group.workers:
            (result,) = worker.fence_previous_store()
            assert not result.succeeded and result.source_release_confirmed and result.error is error
            assert worker.take_released_store_job_ids() == {17}
            with pytest.raises(RuntimeError, match="fatal Store failure"):
                worker.begin_step(KVTransferStep())
            with pytest.raises(RuntimeError, match="Store channel failed"):
                worker.close()
        assert not group.readable
        assert all(len(channel.calls) == 1 for channel in group.channels)


def test_single_shard_replica_election_needs_no_collective():
    with cluster(tp=4, dcp=1, put_step=4) as group:
        submit_stores(group)
        assert all(result.succeeded for worker in group.workers for result in worker.fence_previous_store())
        assert sum(call[0] == "allocate" for sdk in group.sdks for call in sdk.calls) == 1
        assert all(channel is None for channel in group.channels)


@pytest.mark.parametrize("failure", ("channel", "sessions", "context", None))
def test_worker_owns_channel_before_binding_buffers_and_closes_it_after_sessions(failure, monkeypatch):
    topology = make_topology(tp_size=2, put_step=2, dcp_size=2, physical_layers=(0,))
    topology = replace(
        topology,
        config=replace(topology.config, cp_partition=replace(topology.config.cp_partition, object_shard_count=2)),
    )
    sdk = ByteCopySDK({})
    backend = native_backend("memcache", sdk)
    owner = Resources(backend, resolve_backend_spec("memcache"), topology.transfer_groups)
    trace = []
    channel = Mock(close=Mock(side_effect=lambda: trace.append("channel")))
    acquire = Mock(return_value=channel)
    monkeypatch.setattr(worker_module, "CollectiveGroup", acquire)
    sessions = backend.create_sessions()
    close_sessions = sessions.close
    monkeypatch.setattr(sessions, "close", lambda: (trace.append("sessions"), close_sessions()))
    monkeypatch.setattr(backend, "create_sessions", Mock(return_value=sessions))
    close_backend = backend.close
    monkeypatch.setattr(backend, "close", lambda: (trace.append("backend"), close_backend()))
    prepare = owner.prepare_kv_caches

    def prepare_buffers(caches):
        assert owner._execution_cleanup is not None
        return prepare(caches)

    monkeypatch.setattr(owner, "prepare_kv_caches", prepare_buffers)
    error = RuntimeError(f"{failure} binding failed")
    if failure == "channel":
        acquire.side_effect = error
    elif failure == "sessions":
        backend.create_sessions.side_effect = error
    elif failure == "context":
        monkeypatch.setattr(worker_module, "bind_context", Mock(side_effect=error))
    caches = {"layers.0.group.0": torch.empty(8, 4, 2, 1)}
    if failure is not None:
        with pytest.raises(RuntimeError) as caught:
            Worker(topology, owner, coordinator=make_coordinator(topology), layerwise=True).bind_kv_caches(caches)
        assert caught.value is error
    else:
        worker = Worker(topology, owner, coordinator=make_coordinator(topology), layerwise=True)
        worker.bind_kv_caches(caches)
        assert not trace
        worker.close()
        worker.close()
    acquire.assert_called_once_with(topology.context_global_ranks)
    expected = ["backend"] if failure == "channel" else ["channel", "backend"]
    if failure in ("context", None):
        expected.insert(0, "sessions")
    assert trace == expected
    assert owner.executors_stopped and sdk.closed


@pytest.mark.parametrize("parallelism", ("tp", "pp", "load_only"))
def test_only_shared_store_objects_acquire_a_channel(parallelism, monkeypatch):
    topology = make_topology(
        physical_layers=(0,),
        tp_size=1 if parallelism == "pp" else 2,
        put_step=1 if parallelism == "pp" else 2,
        pp_size=2 if parallelism == "pp" else 1,
        pp_rank=1 if parallelism == "pp" else 0,
    )
    if parallelism == "load_only":
        topology = replace(
            topology,
            config=replace(topology.config, cp_partition=replace(topology.config.cp_partition, object_shard_count=2)),
        )
    acquire = Mock(side_effect=AssertionError("No shared Store needs communication"))
    monkeypatch.setattr(worker_module, "CollectiveGroup", acquire)
    sdk = ByteCopySDK({})
    backend = native_backend("memcache", sdk)
    owner = Resources(backend, resolve_backend_spec("memcache"), topology.transfer_groups)
    worker = Worker(
        topology,
        owner,
        coordinator=make_coordinator(topology),
        layerwise=True,
        store_enabled=parallelism != "load_only",
    )
    worker.bind_kv_caches({"layers.0.group.0": torch.empty(8, 4, 2, 1)})
    worker.close()
    acquire.assert_not_called()
    assert sdk.closed


def test_successive_batches_keep_key_ranges_and_collective_order_even_when_all_keys_exist():
    with cluster(tp=4, put_step=4) as group:
        for job, start, end, hashes in (
            (17, 0, 8, (b"a", b"b")),
            (18, 8, 12, (b"a", b"b", b"c")),
            (19, 8, 12, (b"a", b"b", b"c")),
        ):
            for worker in group.workers:
                command = RangeStoreCommand("store", start, end, ((1, 2, 3),), hashes, 12, job)
                worker.begin_step(KVTransferStep(store=StoreCommandBatch(source_ready_commands=(command,))))
                worker.end_step()
            results = [worker.fence_previous_store() for worker in group.workers]
            assert all(result.succeeded and result.source_release_confirmed for batch in results for result in batch)
            assert all(worker.take_released_store_job_ids() == {job} for worker in group.workers)
            assert all(len(channel.calls) == job - 16 for channel in group.channels)
        assert len(group.objects) == len(group.readable) == 3
        for channel in group.channels:
            assert not channel.calls[-1]
        for sdk in group.sdks[:2]:
            allocations = [call[1] for call in sdk.calls if call[0] == "allocate"]
            assert list(map(len, allocations)) == [2, 1]
            assert allocations[1][0].endswith("@63@0")
        for sdk in group.sdks[2:]:
            assert not any(call[0] in ("allocate", "gva_copy", "publish") for call in sdk.calls)
