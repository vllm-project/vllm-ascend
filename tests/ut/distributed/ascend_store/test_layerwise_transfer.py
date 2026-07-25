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
