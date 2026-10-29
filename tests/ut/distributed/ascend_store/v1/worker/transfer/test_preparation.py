"""Check native argument isolation and request-window boundaries."""

import numpy as np
import pytest

from tests.ut.distributed.ascend_store.v1.helpers import make_worker
from tests.ut.distributed.ascend_store.v1.worker.transfer.fixtures import make_group
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import LoadCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import key_range
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer import rows


@pytest.mark.parametrize("layerwise", (False, True), ids=("bulk", "key_range"))
def test_native_argument_rows_do_not_alias_other_rows_or_later_calls(layerwise):
    worker, _, _ = make_worker(layerwise=layerwise)
    try:
        batch = worker._build_load_batch((LoadCommand("r", 0, 8, ((1, 2),), (b"a", b"b")),))
        if layerwise:
            worker._prepare_load_layers(batch)

        def materialize():
            if layerwise:
                return key_range.materialize_load_layer_ranges(
                    worker._key_range_layouts, worker._load_plan, 0, worker._load_key_merges[0]
                )
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
    finally:
        worker.close()


def test_select_keys_intersects_with_existing_selection():
    group = make_group(0, (0, 2, 4), partition_count=2)
    accepted = {group.keys_by_partition[0][1], group.keys_by_partition[1][2]}
    selected = group.select_keys(accepted)
    np.testing.assert_array_equal(
        selected.transfer_candidate_mask, (False, True, False, False, False, False, True, False)
    )
    assert set(selected.selected_keys()) == accepted
    assert selected.select_keys({group.keys_by_partition[0][0]}).selected_keys() == ()


@pytest.mark.parametrize("start", (0, 997 * 128, 998 * 128 + 1))
def test_load_range_starts_at_local_window_or_requested_prefix(start):
    counts, hashes, ids = rows.load_transfer_records(
        1000 * 128,
        tuple(f"h{i}" for i in range(1000)),
        (11, 12, 13),
        block_size=128,
        hash_block_size=128,
        start_token=start,
    )
    first = max(997, start // 128)
    assert hashes == tuple(f"h{i}" for i in range(first, 1000))
    assert ids.tolist() == [11, 12, 13][first - 997 :]
    assert counts.tolist() == [128] * len(ids)


def test_empty_load_window_still_validates_tail_key_boundary():
    with pytest.raises(ValueError, match="does not identify the Load tail"):
        rows.load_transfer_records(
            128, ("h",), (), block_size=128, hash_block_size=128, start_token=256, tail_boundary_token=256
        )
