"""Store admission keeps missing-key rows and checkpoint sources aligned."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    begin_step,
    make_topology,
    make_worker,
    store_one,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    CheckpointStoreCommand,
    RangeStoreCommand,
    StateCheckpointSource,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
)


@pytest.mark.parametrize("layerwise", (False, True), ids=("bulk", "layerwise"))
@pytest.mark.parametrize("scenario", ("full_hit", "partial_hit", "duplicate", "lookup_error"))
def test_store_admission_copies_only_missing_unique_keys_and_releases_all_jobs(
    layerwise, scenario, monkeypatch, caplog
) -> None:
    backend = FakeBackend()
    presence = {"full_hit": [1, 1], "partial_hit": [1, 0], "duplicate": [0], "lookup_error": [0, 0]}
    backend.presence = presence[scenario]
    if scenario == "lookup_error":

        def fail_lookup(keys):
            backend.calls.append(("exists", tuple(keys)))
            raise RuntimeError("exists failed")

        monkeypatch.setattr(backend, "exists", fail_lookup)
    events = []

    def make_event():
        event = FakeEvent()
        events.append(event)
        return event

    worker, resources, _ = make_worker(
        backend, layerwise=layerwise, requires_exists_before_put=True, source_ready_event_factory=make_event
    )
    first = RangeStoreCommand("first", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    second_hash = b"a" if scenario == "duplicate" else b"b"
    second = RangeStoreCommand("second", TokenRange(0, 4), ((3,),), (second_hash,), 4, 18)
    try:
        begin_step(worker, store=(first, second))
        if layerwise:
            worker.save_layer("layers.0.group.0")
            worker.save_layer("layers.1.group.0")
        worker.finish_step()
        completions = worker.fence_previous_store()

        if not layerwise:
            assert len(completions) == 2
            assert all(item.evidence.succeeded for item in completions) is (scenario != "lookup_error")
            assert all(item.evidence.source_release_confirmed for item in completions)
        if scenario == "lookup_error":
            assert "KV cache Store failed for request first" in caplog.text
            assert "KV cache Store failed for request second" in caplog.text
        assert worker.take_released_store_job_ids() == {17, 18}
        assert [call[0] for call in backend.calls].count("exists") == 1
        copies = [call for call in backend.calls if call[0] == ("batch_copy_put" if layerwise else "put")]
        if scenario in ("full_hit", "lookup_error"):
            assert copies == []
            assert not any(call[0] in ("batch_put_start", "batch_commit") for call in backend.calls)
            assert all(event.recorded and not event.synchronized for event in events)
        else:
            source_block = 3 if scenario == "partial_hit" else 1
            assert len(copies) == (2 if layerwise else 1)
            assert all(len(call[1]) == 1 for call in copies)
            addresses: list[tuple[tuple[int, ...], ...]] = [
                ((1000 + source_block * 64,),),
                ((2000 + source_block * 64,),),
            ]
            expected_addresses = addresses if layerwise else [((1000 + source_block * 64, 2000 + source_block * 64),)]
            assert [call[2] for call in copies] == expected_addresses
            assert events and all(event.recorded and event.synchronized for event in events)
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("layerwise", (False, True), ids=("bulk", "layerwise"))
def test_store_admission_preserves_missing_rows_in_each_group(layerwise) -> None:
    backend = FakeBackend()
    backend.presence = [1, 0, 0, 1]
    worker, resources, _ = make_worker(
        backend, topology=make_topology(group_ids=(0, 1)), layerwise=layerwise, requires_exists_before_put=True
    )
    command = RangeStoreCommand("request", TokenRange(0, 8), ((1, 2), (3, 4)), (b"a", b"b"), 8, 17)
    try:
        begin_step(worker, store=(command,))
        if layerwise:
            worker.save_layer("layers.0.group.0")
            worker.save_layer("layers.1.group.0")
        worker.finish_step()
        worker.fence_previous_store()

        candidates = next(call[1] for call in backend.calls if call[0] == "exists")
        selected_keys = (candidates[1], candidates[2])
        copies = [call for call in backend.calls if call[0] == ("batch_copy_put" if layerwise else "put")]
        assert [call[1] for call in copies] == [selected_keys] * (2 if layerwise else 1)
        expected_addresses: list[tuple[tuple[int, ...], ...]] = (
            [((1128,), (11192,)), ((2128,), (12192,))] if layerwise else [((1128, 2128), (11192, 12192))]
        )
        assert [call[2] for call in copies] == expected_addresses
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize(
    "topology,presence,key_axis",
    (
        (make_topology(tp_mismatch=True), [1, 0], "@head_or_tp_rank:1@"),
        (make_topology(consumer_pipeline_partitions=(1, 1)), [0, 1], "@pp_rank:0@"),
    ),
    ids=("tp_head", "pipeline_stage"),
)
def test_store_admission_selects_only_the_missing_effective_key_axis(topology, presence, key_axis) -> None:
    backend = FakeBackend()
    backend.presence = presence
    worker, resources, _ = make_worker(backend, topology=topology, requires_exists_before_put=True)
    try:
        completion = store_one(worker, RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17))
        candidates = next(call[1] for call in backend.calls if call[0] == "exists")
        put_call = next(call for call in backend.calls if call[0] == "put")
        assert put_call[1] == (candidates[presence.index(0)],)
        assert key_axis in put_call[1][0]
        (evidence,) = completion.evidence.transfer_evidence
        assert evidence.source.key == put_call[1][0]
        assert (evidence.source.group_id, evidence.source.block_id) == (0, 1)
        assert evidence.result_code == 0
        assert worker.take_released_store_job_ids() == {17}
    finally:
        worker.close()
    assert resources.closed


def test_checkpoint_store_uses_exact_state_source_after_admission() -> None:
    base = make_topology(group_ids=(0, 1))
    groups = (
        KVPoolGroupTopology(
            0,
            FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32),
            (KVPoolLayerTopology(0, ("layers.0.attention",)),),
            base.groups[0].key_metadata,
        ),
        KVPoolGroupTopology(
            1,
            MambaSpec(block_size=8, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align"),
            (KVPoolLayerTopology(1, ("layers.1.state",)),),
            base.groups[1].key_metadata,
        ),
    )
    topology = replace(base, cache_transfer_granularity=8, hash_block_size=4, groups=groups)
    backend = FakeBackend()
    backend.presence = [1, 0]
    worker, resources, _ = make_worker(backend, topology=topology, requires_exists_before_put=True)
    command = CheckpointStoreCommand("checkpoint", ((3,), (99,)), (b"a",), 0, (StateCheckpointSource(1, 7, 4),), 23)
    try:
        completion = store_one(worker, command)
        put_call = next(call for call in backend.calls if call[0] == "put")
        assert len(put_call[1]) == 1 and "@group:1@" in put_call[1][0]
        assert put_call[2:] == (((12448,),), ((32,),))
        (evidence,) = completion.evidence.transfer_evidence
        assert (evidence.source.group_id, evidence.source.block_id) == (1, 7)
        assert evidence.result_code == 0
        assert worker.take_released_store_job_ids() == {23}
    finally:
        worker.close()
    assert resources.closed
