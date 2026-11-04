"""Store binds the first eligible source before querying the deduplicated Batch."""

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import make_topology
from tests.ut.distributed.ascend_store.v1.worker.fixtures import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    RangeStoreCommand,
    StoreCommandBatch,
)


@pytest.mark.parametrize(
    "rank,stride,expected",
    ((0, 1, [2, 5, 7, 8]), (0, 2, [2, 5, 8]), (1, 2, [4, 7]), (0, 3, [2, 7]), (1, 3, [4, 8]), (2, 3, [5])),
)
def test_store_writer_counts_filtered_candidates_before_dedup(rank, stride, expected, monkeypatch):
    worker, resources, backend = make_worker()
    worker.store._writers[0] = slice(rank, None, stride)
    monkeypatch.setattr(
        worker.store.coordinator, "select_for_store", lambda *_args: ((True, True, True, True, False, True, True),)
    )
    command = RangeStoreCommand(
        "r", 4, 32, ((1, 2, -1, 4, 5, 6, 7, 8),), (b"a", b"b", b"c", b"b", b"d", b"e", b"f", b"g"), 32, 17
    )
    try:
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
        worker.finish_step()
        (result,) = worker.fence_previous_store()
        (copy,) = [call for call in backend.calls if call[0] == "put"]
        assert copy[2] == tuple(
            tuple(cache[block_id].data_ptr() for cache in resources.kv_caches.values()) for block_id in expected
        )
        assert len(copy[1]) == len(set(copy[1])) == len(expected)
        assert result.request_id == "r" and result.keys == copy[1] and result.succeeded
        assert result.source_release_confirmed and worker.take_released_store_job_ids() == {17}
        worker.end_step()
    finally:
        worker.close()


@pytest.mark.parametrize("present", ((False, False), (True, False), (True, True)))
def test_store_queries_unique_keys_and_reads_first_source_across_hash_representations(present):
    worker, resources, backend = make_worker(requires_exists=True)
    commands = (
        RangeStoreCommand("first", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17),
        RangeStoreCommand("second", 0, 8, ((3, 4),), ("61", "62"), 8, 18),
    )
    backend.presence = list(present)
    try:
        worker.begin_step(KVTransferStep(store=StoreCommandBatch(commands)))
        worker.finish_step()
        first, second = worker.fence_previous_store()
        (query,) = [call for call in backend.calls if call[0] == "exists"]
        assert len(query[1]) == len(set(query[1])) == 2
        assert [key.rsplit("@", 1)[-1] for key in query[1]] == ["61", "62"]
        retained = [index for index, hit in enumerate(present) if not hit]
        expected_keys = tuple(query[1][index] for index in retained)
        copies = [call for call in backend.calls if call[0] == "put"]
        if retained:
            (copy,) = copies
            assert copy[1] == expected_keys
            assert copy[2] == tuple(
                tuple(cache[index + 1].data_ptr() for cache in resources.kv_caches.values()) for index in retained
            )
        else:
            assert not copies
        assert first.request_id == "first" and first.keys == expected_keys
        assert second.request_id == "second" and not second.keys
        assert all(result.succeeded and result.source_release_confirmed for result in (first, second))
        assert worker.take_released_store_job_ids() == {17, 18}
        worker.end_step()
    finally:
        worker.close()


def test_store_final_keys_deduplicate_cross_family_aliases_with_original_partition_owner():
    worker, resources, backend = make_worker(topology=make_topology(group_ids=(0, 1)))
    first, second = worker.context.partition_order
    worker.context.key_parts[0][first] = ("ab", "")
    worker.context.key_parts[1][second] = ("a", "")
    commands = (
        RangeStoreCommand("first", 0, 4, ((1,), ()), ("c",), 4, 17),
        RangeStoreCommand("second", 0, 4, ((), (2,)), ("bc",), 4, 18),
    )
    try:
        worker.begin_step(KVTransferStep(store=StoreCommandBatch(commands)))
        worker.finish_step()
        first_result, second_result = worker.fence_previous_store()
        (copy,) = [call for call in backend.calls if call[0] == "put"]
        assert copy[1] == ("abc",)
        assert copy[2] == (
            tuple(cache[1].data_ptr() for name, cache in resources.kv_caches.items() if name.endswith(".group.0")),
        )
        assert first_result.request_id == "first" and first_result.keys == ("abc",)
        assert second_result.request_id == "second" and not second_result.keys
        assert all(result.succeeded and result.source_release_confirmed for result in (first_result, second_result))
        assert worker.take_released_store_job_ids() == {17, 18}
        worker.end_step()
    finally:
        worker.close()
