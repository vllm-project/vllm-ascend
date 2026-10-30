"""Store binds the first eligible source before querying the deduplicated Batch."""

from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import make_topology
from tests.ut.distributed.ascend_store.v1.worker.fixtures import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import RangeStoreCommand


@pytest.mark.parametrize(
    "rank,stride,expected",
    ((0, 1, [2, 5, 7, 8]), (0, 2, [2, 5, 8]), (1, 2, [4, 7]), (0, 3, [2, 7]), (1, 3, [4, 8]), (2, 3, [5])),
)
def test_store_writer_counts_filtered_candidates_before_dedup(rank, stride, expected, monkeypatch):
    worker, _, _ = make_worker()
    worker.store._writers[0] = slice(rank, None, stride)
    monkeypatch.setattr(
        worker.store.coordinator, "select_for_store", lambda *_args: ((True, True, True, True, False, True, True),)
    )
    command = RangeStoreCommand(
        "r", 4, 32, ((1, 2, -1, 4, 5, 6, 7, 8),), (b"a", b"b", b"c", b"b", b"d", b"e", b"f", b"g"), 32, 17
    )
    try:
        selection = worker.store._candidates((command,))
        assert [block_id for _, block_id, _ in selection.sources] == expected
        assert all(owner == 0 for _, _, owner in selection.sources)
        assert len(selection.keys) == len(set(selection.keys))
    finally:
        worker.close()


@pytest.mark.parametrize("present", ((False, False), (True, False), (True, True)))
def test_store_queries_batch_once_and_keeps_first_source_across_hash_representations(present, monkeypatch):
    worker, _, backend = make_worker(requires_exists=True)
    commands = (
        RangeStoreCommand("first", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17),
        RangeStoreCommand("second", 0, 8, ((3, 4),), ("61", "62"), 8, 18),
    )
    candidates = worker.store._candidates(commands)
    backend.exists = Mock(return_value=present)
    try:
        selection = worker.store._prepare(commands)
        backend.exists.assert_called_once_with(candidates.keys)
        assert len(candidates.keys) == 2
        retained = [index for index, hit in enumerate(present) if not hit]
        assert selection.keys == [candidates.keys[index] for index in retained]
        assert [block_id for _, block_id, _ in selection.sources] == [index + 1 for index in retained]
        assert selection.sources == [candidates.sources[index] for index in retained]
        assert all(owner == 0 for _, _, owner in selection.sources)
    finally:
        worker.close()


def test_store_final_keys_deduplicate_cross_family_aliases_with_original_partition_owner():
    worker, _, _ = make_worker(topology=make_topology(group_ids=(0, 1)))
    first, second = worker.context.partition_order
    worker.context.key_parts[0][first] = ("ab", "")
    worker.context.key_parts[1][second] = ("a", "")
    commands = (
        RangeStoreCommand("first", 0, 4, ((1,), ()), ("c",), 4, 17),
        RangeStoreCommand("second", 0, 4, ((), (2,)), ("bc",), 4, 18),
    )
    try:
        selection = worker.store._candidates(commands)
        assert selection.keys == ["abc"]
        assert selection.sources == [(first, 1, 0)]
    finally:
        worker.close()
