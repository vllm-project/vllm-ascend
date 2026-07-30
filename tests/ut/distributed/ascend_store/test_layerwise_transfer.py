from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    GroupBatchPlan,
    GroupTransferData,
    LayerBlockRange,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_transfer import (
    LayerTransferArrayBuilder,
    LayerwiseTransferPreparer,
)


def _token_database(layer_offsets):
    return SimpleNamespace(
        group_block_len={0: [10, 2, 20]},
        group_kv_caches_base_addr={0: [100, 200, 300]},
        group_block_stride={0: [1000, 2000, 3000]},
        group_layer_offsets={0: layer_offsets},
    )


def test_array_builder_supports_variable_cache_legs_per_layer():
    builder = LayerTransferArrayBuilder(
        _token_database([0, 2, 3]),
        num_layers=2,
    )
    data = GroupTransferData(
        block_ids_arr=np.asarray([1], dtype=np.int64),
        base_gvas_arr=np.asarray([10_000], dtype=np.int64),
    )

    layer_zero = builder.build_addrs(data, 0)
    layer_one = builder.build_addrs(data, 1)

    np.testing.assert_array_equal(layer_zero.addr_array, [1100, 2200])
    np.testing.assert_array_equal(layer_zero.size_array, [10, 2])
    np.testing.assert_array_equal(layer_zero.gvas_array, [10_000, 10_010])
    np.testing.assert_array_equal(layer_one.addr_array, [3300])
    np.testing.assert_array_equal(layer_one.size_array, [20])
    np.testing.assert_array_equal(layer_one.gvas_array, [10_012])


def test_array_builder_rejects_invalid_layer_offsets():
    with pytest.raises(ValueError, match="Invalid layerwise offsets"):
        LayerTransferArrayBuilder(
            _token_database([0, 1, 2]),
            num_layers=2,
        )


def test_preparer_releases_shared_lease_after_last_request():
    backend = MagicMock()
    backend.batch_remove_lease.return_value = 0
    preparer = LayerwiseTransferPreparer(
        backend,
        "model",
        0,
        16,
        enabled=True,
        can_allocate=True,
        num_groups=1,
    )
    preparer.register_load_leases(
        {
            "request-a": {"shared", "only-a"},
            "request-b": {"shared"},
        }
    )

    preparer.release_finished_load_leases({"request-a"})
    backend.batch_remove_lease.assert_called_once_with(["only-a"])

    backend.batch_remove_lease.reset_mock()
    preparer.release_finished_load_leases({"request-b"})
    backend.batch_remove_lease.assert_called_once_with(["shared"])


def _load_plan(block_hashes):
    requests = [
        ReqMeta(
            f"request-{index}",
            token_len_chunk=16,
            block_ids_by_group=[[index + 1]],
            block_hashes=[block_hash],
            is_last_chunk=True,
        )
        for index, block_hash in enumerate(block_hashes)
    ]
    return GroupBatchPlan(
        group_id=0,
        block_size=16,
        full_load_ranges=[LayerBlockRange(request, start_block=0, end_block=1) for request in requests],
    )


def _key_info(gvas):
    return SimpleNamespace(size=lambda: 1, gva_list=lambda: gvas)


@pytest.mark.parametrize("gvas", [[], [0]])
def test_preparer_rejects_invalid_gva_before_acquiring_lease(gvas):
    backend = MagicMock()
    backend.batch_get_key_info.return_value = [_key_info(gvas)]
    preparer = LayerwiseTransferPreparer(
        backend,
        "model",
        0,
        16,
        enabled=True,
        can_allocate=False,
        num_groups=1,
    )

    with pytest.raises(RuntimeError, match="invalid GVA metadata"):
        preparer.resolve_load_groups([_load_plan(["aa"])])

    backend.batch_add_lease.assert_not_called()
    backend.batch_remove_lease.assert_not_called()
    assert preparer.load_lease_keys_by_request == {}


@pytest.mark.parametrize("lease_results", [[0, 17], [0]])
def test_preparer_rolls_back_successful_leases_after_acquisition_failure(lease_results):
    backend = MagicMock()
    backend.batch_get_key_info.return_value = [
        _key_info([1_000]),
        _key_info([2_000]),
    ]
    backend.batch_add_lease.return_value = lease_results
    backend.batch_remove_lease.return_value = 0
    preparer = LayerwiseTransferPreparer(
        backend,
        "model",
        0,
        16,
        enabled=True,
        can_allocate=False,
        num_groups=1,
    )
    plan = _load_plan(["aa", "bb"])

    with pytest.raises(RuntimeError, match="lease acquisition failed"):
        preparer.resolve_load_groups([plan])

    first_key = preparer.make_gva_key(0, "aa")
    backend.batch_remove_lease.assert_called_once_with([first_key])
    assert preparer.load_lease_keys_by_request == {}


def _partial_plan():
    request = ReqMeta(
        "request-a",
        token_len_chunk=16,
        target_token_len=20,
        num_prompt_tokens=64,
        block_ids_by_group=[[7, 8]],
        block_hashes=["aa"],
        is_last_chunk=False,
    )
    block_range = LayerBlockRange(
        request,
        start_block=0,
        end_block=2,
        partial_end_token=20,
    )
    return GroupBatchPlan(
        group_id=0,
        block_size=16,
        save_ranges=[block_range],
        full_load_ranges=[block_range],
    )


def test_preparer_saves_partial_block_with_request_scoped_key():
    backend = MagicMock()
    backend.batch_alloc.return_value = [1_000, 2_000]
    preparer = LayerwiseTransferPreparer(
        backend,
        "model",
        0,
        16,
        enabled=True,
        can_allocate=True,
        num_groups=1,
    )
    preparer.configure_layout({0: [10, 20]})
    plan = _partial_plan()

    resolved = preparer.resolve_save_groups([plan])

    full_key = preparer.make_gva_key(0, "aa")
    partial_key = "model@partial@request-a@0@1@20@0"
    backend.batch_alloc.assert_called_once_with(
        [full_key, partial_key],
        [30, 30],
    )
    np.testing.assert_array_equal(resolved[0][0].block_ids_arr, [7, 8])


def test_preparer_loads_partial_block_with_same_request_scoped_key():
    backend = MagicMock()
    backend.batch_get_key_info.return_value = [
        _key_info([1_000]),
        _key_info([2_000]),
    ]
    backend.batch_add_lease.return_value = [0, 0]
    preparer = LayerwiseTransferPreparer(
        backend,
        "model",
        0,
        16,
        enabled=True,
        can_allocate=False,
        num_groups=1,
    )
    plan = _partial_plan()

    resolved = preparer.resolve_load_groups([plan])

    full_key = preparer.make_gva_key(0, "aa")
    partial_key = "model@partial@request-a@0@1@20@0"
    backend.batch_get_key_info.assert_called_once_with(
        [full_key, partial_key],
        flag=1,
    )
    backend.batch_add_lease.assert_called_once_with(
        [full_key, partial_key],
        5 * 60 * 1000,
    )
    np.testing.assert_array_equal(resolved[(0, False)][0].block_ids_arr, [7, 8])
    np.testing.assert_array_equal(resolved[(0, False)][0].base_gvas_arr, [1_000, 2_000])


def test_save_preparation_protects_load_keys_before_allocating():
    backend = MagicMock()
    load_preparation = MagicMock()
    load_ready = False

    def mark_load_ready():
        nonlocal load_ready
        load_ready = True

    def allocate_after_load(keys, sizes):
        assert load_ready
        return [1_000, 2_000]

    load_preparation.ensure_ready.side_effect = mark_load_ready
    backend.batch_alloc.side_effect = allocate_after_load
    preparer = LayerwiseTransferPreparer(
        backend,
        "model",
        0,
        16,
        enabled=True,
        can_allocate=True,
        num_groups=1,
    )
    preparer.configure_layout({0: [10, 20]})

    save_preparation = preparer.create_save_preparation(
        [_partial_plan()],
        [],
        None,
        load_preparation,
    )
    save_preparation.ensure_ready()

    load_preparation.ensure_ready.assert_called_once_with()
    backend.batch_alloc.assert_called_once()
