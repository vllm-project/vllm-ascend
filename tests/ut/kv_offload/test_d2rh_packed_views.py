# SPDX-License-Identifier: Apache-2.0
"""Packed KV views retain aliases only within the same named cache."""

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_d2rh_connector as d2rh


def staging(caches, prefill_tp=1, decode_tp=1, pcp=1, dcp=1):
    worker = object.__new__(d2rh.MooncakeConnectorWorker)
    worker._prefill_tp_size = prefill_tp
    worker.tp_size = decode_tp
    worker.pcp_size = pcp
    worker.dcp_size = dcp
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    with patch.object(d2rh.torch, "empty", side_effect=cpu_empty):
        result = worker._make_cpu_staging_caches(caches)
    return worker, result


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
def test_data_scale_and_packed_views_share_staging_bytes(dtype):
    packed = torch.arange(528, dtype=torch.int32).to(torch.uint8).view(dtype).view(4, 132)
    data = packed[:, :128]
    scale = packed[:, 128:].view(torch.float32)
    worker, result = staging({"indexer": (data, scale, packed)})
    cpu_data, cpu_scale, cpu_packed = result["indexer"]
    assert cpu_data.data_ptr() == cpu_packed.data_ptr()
    assert cpu_scale.data_ptr() == cpu_packed.data_ptr() + 128
    assert cpu_data.stride() == data.stride()
    assert cpu_scale.stride() == scale.stride()
    cpu_packed.copy_(packed)
    assert torch.equal(cpu_data.view(torch.uint8), data.view(torch.uint8))
    assert torch.equal(cpu_scale.view(torch.uint8), scale.view(torch.uint8))
    assert worker._cpu_register_lengths == [d2rh.HUGEPAGE_SIZE_2M]
    assert worker._cpu_register_ptrs[0] % d2rh.HUGEPAGE_SIZE_2M == 0


def test_cross_name_aliases_remain_independent():
    tensor = torch.ones((4, 16), dtype=torch.uint8)
    worker, result = staging({"attention": tensor, "state": tensor})
    assert result["attention"][0].data_ptr() != result["state"][0].data_ptr()
    result["attention"][0].fill_(1)
    result["state"][0].fill_(2)
    assert torch.equal(result["attention"][0], tensor)
    assert worker._cpu_register_lengths == [2 * d2rh.HUGEPAGE_SIZE_2M]


@pytest.mark.parametrize("parallelism", [(2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 1, 1, 2)])
def test_sharded_layouts_keep_independent_contiguous_staging(parallelism):
    packed = torch.empty((4, 132), dtype=torch.uint8)
    _, result = staging({"indexer": (packed[:, :128], packed)}, *parallelism)
    data, combined = result["indexer"]
    assert data.data_ptr() != combined.data_ptr()
    assert data.is_contiguous() and combined.is_contiguous()


def test_mixed_dtype_region_start_keeps_element_alignment():
    backing = torch.empty(32, dtype=torch.uint8)
    byte_view = backing[1:17]
    float_view = backing[4:20].view(torch.float32)
    _, result = staging({"mixed": (byte_view, float_view)})
    cpu_byte, cpu_float = result["mixed"]
    assert cpu_float.data_ptr() % 4 == 0
    assert cpu_float.data_ptr() - cpu_byte.data_ptr() == 3


@pytest.mark.parametrize("num_group_pulls", [1, 2])
def test_prune_requires_matching_aliases_on_both_peers(num_group_pulls):
    pairs = [(0, 1), (1, 2), (2, 0), (3, 3)]
    kwargs = dict(
        local_addrs=[1000, 1128, 1000, 3000],
        remote_addrs=[2000, 2000, 2128, 4000],
        block_lengths=[128, 4, 132, 128],
        local_strides=[132, 132, 132, 128],
        remote_strides=[132, 132, 132, 128],
        num_group_pulls=num_group_pulls,
    )
    expected = [(2, 0), (3, 3)] if num_group_pulls == 1 else pairs
    assert d2rh._get_non_redundant_cache_slot_pairs(pairs, **kwargs) == expected
    kwargs["remote_addrs"] = [2000, 5000, 6000, 4000]
    assert d2rh._get_non_redundant_cache_slot_pairs(pairs, **kwargs) == pairs


@pytest.mark.parametrize("mismatch", ["offset", "stride", "no_cover"])
def test_prune_does_not_discard_distinct_or_unselected_data(mismatch):
    pairs = [(0, 0), (1, 1)]
    local = [1000, 1000]
    remote = [2000, 2000]
    strides = [132, 132]
    if mismatch == "offset":
        remote[0] += 4
    elif mismatch == "stride":
        strides[0] = 128
    else:
        pairs = [(0, 0)]
    assert d2rh._get_non_redundant_cache_slot_pairs(pairs, local, remote, [128, 132], strides, [132, 132], 1) == pairs


def test_identical_views_keep_one_stable_representative():
    pairs = [(0, 0), (1, 1)]
    assert d2rh._get_non_redundant_cache_slot_pairs(
        pairs, [1000, 1000], [2000, 2000], [132, 132], [132, 132], [132, 132], 1
    ) == [(0, 0)]
