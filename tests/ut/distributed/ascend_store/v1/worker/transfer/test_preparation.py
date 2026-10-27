"""Guard runtime work boundaries without machine-dependent timing thresholds."""

from unittest.mock import Mock

import numpy as np
import pytest

from tests.ut.distributed.ascend_store.v1.helpers import make_worker
from tests.ut.distributed.ascend_store.v1.worker.transfer.fixtures import make_group
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import LoadCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import key_range
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer import rows
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import KVGroupBatch


@pytest.mark.parametrize("layerwise", (False, True), ids=("bulk", "key_range"))
def test_native_scalar_templates_are_bound_once_and_copies_do_not_alias(layerwise, monkeypatch):
    worker, resources, _ = make_worker(layerwise=layerwise)
    try:
        batch = worker._build_load_batch((LoadCommand("r", 0, 8, ((1, 2),), (b"a", b"b")),))
        assert not hasattr(batch.groups[0], "token_counts")
        with monkeypatch.context() as runtime:
            runtime.setattr(np, "broadcast_to", Mock(side_effect=AssertionError("static conversion after binding")))
            if layerwise:
                # Membership and native layouts already hold all static Mapping facts.
                runtime.setattr(worker._mappings[0], "layer_layouts", {})
                worker._prepare_load_layers(batch)

                def materialize():
                    return key_range.materialize_load_layer_ranges(
                        worker._key_range_layouts, worker._load_plan, 0, worker._load_key_merges[0]
                    )
            else:

                def materialize():
                    return worker._materialize_bulk_arguments(batch)

            first = materialize()
            expected_sizes = first.sizes[1].copy()
            first.sizes[0][0] = -1
            assert first.sizes[1] == expected_sizes
            if layerwise:
                expected_offsets = first.offsets[1].copy()
                first.offsets[0][0] = -1
                assert first.offsets[1] == expected_offsets
            second = materialize()
            assert second.sizes[0] == expected_sizes
            if layerwise:
                assert second.offsets[0] == expected_offsets
        registration = resources.prepare_kv_caches({"cache": object()})
        registration["base_addresses"] = {
            group_id: [address + 100000 for address in addresses]
            for group_id, addresses in registration["base_addresses"].items()
        }
        worker._bind_mappings(registration)
        if layerwise:
            worker._prepare_load_layers(batch)
        rebound = materialize()
        assert rebound.addresses[0][0] == second.addresses[0][0] + 100000
        if layerwise:
            worker._finish_load()
            assert worker._load_plan is None and worker._load_key_merges == {}
    finally:
        worker.close()


@pytest.mark.parametrize("duplicate", (False, True))
def test_key_merge_detection_runs_once_per_membership_and_success_needs_no_provenance(duplicate, monkeypatch):
    worker, _, backend = make_worker(layerwise=True, physical_layers=(2, 5, 9))
    try:
        commands = (
            LoadCommand("first", 0, 4, ((1,),), (b"a",)),
            LoadCommand("second", 0, 4, ((7,),), (b"a" if duplicate else b"b",)),
        )
        batch = worker._build_load_batch(commands)
        detect = Mock(side_effect=set)
        ownership = Mock(side_effect=KVGroupBatch.request_index)
        monkeypatch.setattr(key_range, "set", detect, raising=False)
        monkeypatch.setattr(KVGroupBatch, "request_index", lambda group, index: ownership(group, index))
        worker._prepare_load_layers(batch)
        detect.assert_called_once()
        ownership.assert_not_called()
        selection = worker._load_key_merges[2]
        assert selection is worker._load_key_merges[5] is worker._load_key_merges[9]
        assert (selection is not None) is duplicate
        detect.side_effect = AssertionError("repeated duplicate detection during layer execution")
        for layer in (2, 5, 9):
            assert all(not c.failures for c in worker._copy_load_layer(layer))
        ownership.assert_not_called()
        backend.session_copy_result = (-7,) if duplicate else (-7, 0)
        completions = worker._copy_load_layer(5)
        assert ownership.call_count == (2 if duplicate else 1)
        assert [f.source.local_block_id for c in completions for f in c.failures] == ([1, 7] if duplicate else [1])
    finally:
        worker.close()


def test_select_keys_uses_implicit_full_selection_without_allocating_all_true_mask(monkeypatch):
    group = make_group(0, (0, 2, 4), partition_count=2)
    monkeypatch.setattr(np, "ones", Mock(side_effect=AssertionError("implicit selection needs no mask")))
    accepted = {group.keys_by_partition[0][1], group.keys_by_partition[1][2]}
    selected = group.select_keys(accepted)
    np.testing.assert_array_equal(
        selected.transfer_candidate_mask, (False, True, False, False, False, False, True, False)
    )
    assert set(selected.selected_keys()) == accepted
    assert selected.select_keys({group.keys_by_partition[0][0]}).selected_keys() == ()


@pytest.mark.parametrize("start", (0, 997 * 128, 998 * 128 + 1))
def test_load_range_starts_at_local_window_or_requested_prefix(start, monkeypatch):
    visited = []

    def record_range(*args):
        indices = range(*args)
        visited.extend(indices)
        return indices

    monkeypatch.setattr(rows, "range", record_range, raising=False)
    counts, hashes, ids = rows.load_transfer_records(
        1000 * 128,
        tuple(f"h{i}" for i in range(1000)),
        (11, 12, 13),
        block_size=128,
        hash_block_size=128,
        start_token=start,
    )
    first = max(997, start // 128)
    assert visited == list(range(first, 1000))
    assert hashes == tuple(f"h{i}" for i in range(first, 1000))
    assert ids.tolist() == [11, 12, 13][first - 997 :]
    assert counts.tolist() == [128] * len(ids)


def test_empty_load_window_still_validates_tail_key_boundary():
    with pytest.raises(ValueError, match="does not identify the Load tail"):
        rows.load_transfer_records(
            128, ("h",), (), block_size=128, hash_block_size=128, start_token=256, tail_boundary_token=256
        )
