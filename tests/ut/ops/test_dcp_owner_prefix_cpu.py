# SPDX-License-Identifier: Apache-2.0
"""Exercise actual prefix-search arithmetic against an independent full sort."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from tests.ut.ops.test_dcp_kernels_cpu import Pointer, load_kernels


@pytest.mark.parametrize("world,log_k", [(1, 0), (2, 1), (3, 3), (8, 5), (16, 7), (17, 11)])
@pytest.mark.parametrize(
    "pattern", ["random", "ties", "all_ties", "owner_ties", "owner_wins", "peer_wins", "invalid", "duplicates"]
)
def test_owner_prefix_exact(world, log_k, pattern):
    tl, kernels = load_kernels("vllm_ascend/ops/triton/sharded_indexer.py", ("_merge_owner_prefix",))
    rng = np.random.default_rng(5303)
    k, rows, interleave, owner = 1 << log_k, 3, 128, world - 1
    local = rng.integers(0, 10000, (world, rows, k), dtype=np.int32)
    ranks = np.arange(world, dtype=np.int32)[:, None, None]
    ids = (local // interleave * world + ranks) * interleave + local % interleave
    scores = rng.standard_normal(ids.shape).astype(np.float32)
    if pattern in ("ties", "duplicates"):
        scores = rng.integers(-2, 3, ids.shape).astype(np.float32)
    if pattern == "all_ties":
        scores.fill(1.0)
    if pattern == "duplicates":
        ids[:] = ranks * interleave
    if pattern == "owner_wins":
        scores[owner] += 100
    if pattern == "peer_wins" and world > 1:
        scores[0] += 100
    if pattern == "invalid":
        ids[:] = -1
    else:
        ids[:, 0] = -1
    if k >= 8 and pattern != "all_ties":
        scores[:, 1, :5] = [np.nan, np.inf, -np.inf, 0.0, -0.0]
        scores[:, 2, :2] = [1.0, np.nextafter(np.float32(1), np.float32(2))]
    if pattern == "owner_ties":
        scores[owner].fill(1.0)
    valid = (ids >= 0) & np.isfinite(scores)
    values = np.where(valid, scores, -np.inf).astype(np.float32)
    order = np.lexsort((np.where(valid, ids, np.inf), -values), axis=-1)
    ordered_ids = np.take_along_axis(np.where(valid, ids, -1), order, -1)
    ordered_scores = np.take_along_axis(values, order, -1)
    packed = np.stack((ordered_ids, ordered_scores.view(np.int32)), -1)
    output = np.full((rows, k), -123, np.int32)
    tile = min(k, 1 << (max(1, 2 * k // world) - 1).bit_length())
    tl.grid = (2,)
    for program in range(2):
        tl.pid = (program,)
        kernels["_merge_owner_prefix"](
            Pointer(packed), Pointer(output), rows, world, owner, interleave, log_k, tile, 1 << (world - 1).bit_length()
        )
    expected = np.full_like(output, -1)
    for row in range(rows):
        row_ids, row_scores = ids[:, row].reshape(-1), scores[:, row].reshape(-1)
        keep = (row_ids >= 0) & np.isfinite(row_scores)
        selected = row_ids[keep][np.lexsort((row_ids[keep], -row_scores[keep]))[:k]]
        selected = selected[(selected // interleave) % world == owner]
        selected = selected // (world * interleave) * interleave + selected % interleave
        expected[row, : selected.size] = selected
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("log_k", [3, 11])
def test_flat_fused_pack_keeps_mapped_integer_payloads(dtype, log_k):
    tl, kernels = load_kernels(
        "vllm_ascend/ops/triton/sharded_indexer.py", ("_pack_sorted_candidates", "_sort_candidate_pairs")
    )
    rows, k, world, rank, interleave = 3, 1 << log_k, 16, 15, 768
    rng = np.random.default_rng(53)
    ids = rng.integers(0, 1 << 24, (rows, k * 2), dtype=np.int32)
    scores = rng.integers(-3, 4, ids.shape).astype(dtype)
    ids[:, 6] = -1
    scores[:, :6:2] = [np.nan, np.inf, -np.inf]
    packed = np.full((rows, k, 2), -123, np.int32)
    tl.grid = (2,)
    for program in range(2):
        tl.pid = (program,)
        kernels["_pack_sorted_candidates"](
            Pointer(ids), Pointer(scores), Pointer(packed), rows, k * 2, 2, k * 2, 2, world, rank, interleave, log_k
        )
    local, values = ids[:, ::2], scores[:, ::2].astype(np.float32)
    valid = (local >= 0) & np.isfinite(values)
    mapped = (local // interleave * world + rank) * interleave + local % interleave
    values = np.where(valid, values, -np.inf).astype(np.float32)
    order = np.lexsort((np.where(valid, local, np.inf), -values), axis=-1)
    np.testing.assert_array_equal(packed[..., 0], np.take_along_axis(np.where(valid, mapped, -1), order, -1))
    np.testing.assert_array_equal(packed[..., 1], np.take_along_axis(values.view(np.int32), order, -1))


def test_owner_prefix_requires_explicit_owner_contract():
    _, scope = load_kernels("vllm_ascend/ops/triton/sharded_indexer.py", ("merge_packed_candidates",))
    prefix = Mock()
    prefix.__getitem__ = Mock(return_value=Mock())
    scope.update(
        triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        init_device_properties_triton=Mock(),
        get_vectorcore_num=lambda: 40,
        _merge_owner_prefix=prefix,
    )
    packed = SimpleNamespace(shape=(768, 2048, 2), device="npu")
    output = object()
    assert (
        scope["merge_packed_candidates"](
            packed, 48, 16, 2048, local_rank=15, presorted=True, owner_sharded=True, out=output
        )
        is output
    )
    prefix.__getitem__.assert_called_once_with((40,))
    assert prefix.__getitem__.return_value.call_args.args[-2:] == (256, 16)


@pytest.mark.parametrize("presorted,owner_sharded,rank", [(True, False, 7), (False, True, 7), (True, True, None)])
def test_generic_merge_does_not_assume_owner_sharding(presorted, owner_sharded, rank):
    _, scope = load_kernels("vllm_ascend/ops/triton/sharded_indexer.py", ("merge_packed_candidates",))
    prefix, generic = MagicMock(), MagicMock()
    scope.update(
        triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        init_device_properties_triton=Mock(),
        get_vectorcore_num=lambda: 64,
        _merge_owner_prefix=prefix,
        _merge_candidates=generic,
    )
    packed = SimpleNamespace(shape=(1024, 2048, 2), device="npu")
    output = object()
    assert (
        scope["merge_packed_candidates"](
            packed, 128, 8, 2048, local_rank=rank, presorted=presorted, owner_sharded=owner_sharded, out=output
        )
        is output
    )
    prefix.__getitem__.assert_not_called()
    generic.__getitem__.return_value.assert_called_once()


@pytest.mark.parametrize(
    "cores,rows,world,fused",
    [
        (40, 48, 16, True),
        (64, 128, 8, True),
        (64, 48, 16, True),
        (40, 8, 16, False),
        (64, 32, 8, False),
        (40, 128, 3, False),
        (64, 128, 7, True),
    ],
)
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_fused_pack_uses_workload_occupancy(cores, rows, world, fused, dtype):
    _, scope = load_kernels("vllm_ascend/ops/triton/sharded_indexer.py", ("pack_local_candidates",))
    sorted_pack, native_pack = MagicMock(), MagicMock()
    ids, scores = MagicMock(), MagicMock()
    scores.dtype = dtype
    ids.reshape.return_value = ids
    scores.reshape.return_value = scores
    ids.shape = (rows, 1, 2048)
    ids.numel.return_value = rows * 2048
    ids.stride.side_effect = lambda axis: 2048 if axis == 0 else 1
    scores.stride.side_effect = ids.stride.side_effect
    output = object()
    scope.update(
        torch=SimpleNamespace(empty=Mock(return_value=output), int32=np.int32, bfloat16="bf16"),
        triton=SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
        init_device_properties_triton=Mock(),
        get_vectorcore_num=lambda: cores,
        _sort_local_candidates=Mock(return_value=(ids, scores)),
        _pack_sorted_candidates=sorted_pack,
        _pack_candidates=native_pack,
    )
    assert (
        scope["pack_local_candidates"](ids, scores, world, world - 1, 768, sort_candidates=True, fuse_sort=True)
        is output
    )
    assert sorted_pack.__getitem__.called is fused
    assert native_pack.__getitem__.called is not fused
    if fused:
        assert sorted_pack.__getitem__.return_value.call_args.kwargs["RADIX"] is (dtype == "bf16")
