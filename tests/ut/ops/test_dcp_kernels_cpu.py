# SPDX-License-Identifier: Apache-2.0
"""Check actual kernel arithmetic against NumPy, without importing an NPU runtime.

This executes the source bodies with array operations. It cannot establish
Triton-Ascend compilation, UB usage, scheduling, graph or HCCL correctness.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


class Array(np.ndarray):
    def to(self, dtype, bitcast=False):
        return (self.view(dtype) if bitcast else self.astype(dtype)).view(Array)


def array(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(Array)


class Pointer:
    def __init__(self, storage, offset=0):
        self.storage = storage.reshape(-1)
        self.offset = offset

    def __add__(self, offset):
        return Pointer(self.storage, self.offset + offset)


class ArrayLanguage(SimpleNamespace):
    """Only the array/address operations used by these kernels; no algorithm."""

    def __init__(self):
        super().__init__(
            constexpr=int,
            cdiv=lambda n, d: (n + d - 1) // d,
            float32=np.float32,
            int32=np.int32,
            int64=np.int64,
            arange=lambda start, end: array(np.arange(start, end, dtype=np.int32)),
            static_range=range,
            range=lambda *args, **kwargs: range(*args),
            where=lambda *args: array(np.where(*args)),
            abs=np.abs,
            floor=np.floor,
            minimum=np.minimum,
            maximum=np.maximum,
            full=lambda shape, value, dtype: array(np.full(shape, value, dtype=dtype)),
            zeros=lambda shape, dtype: array(np.zeros(shape, dtype=dtype)),
            gather=lambda src, idx, axis: array(np.take_along_axis(src, idx, axis=axis)),
            max=lambda value, axis: array(np.max(value, axis=axis)),
            sum=lambda value, axis: array(np.sum(value, axis=axis)),
            exp=lambda value: array(np.exp(value)),
            exp2=lambda value: array(np.exp2(value)),
            trans=lambda value: value.T,
            join=lambda a, b: array(np.stack((a, b), axis=-1)),
            reshape=lambda value, shape: value.reshape(shape),
            split=lambda value: (value[..., 0], value[..., 1]),
        )
        self.pid = (0,)
        self.grid = (1,)

    def program_id(self, axis):
        return self.pid[axis]

    def num_programs(self, axis):
        return self.grid[axis]

    @staticmethod
    def load(ptr, mask=True, other=0.0):
        offsets, mask = np.broadcast_arrays(ptr.offset, mask)
        # NumPy raises for any unmasked out-of-allocation address.
        selected = ptr.storage[np.where(mask, offsets, 0)]
        return array(np.where(mask, selected, np.asarray(other, dtype=ptr.storage.dtype)))

    @staticmethod
    def store(ptr, value, mask=True):
        offsets, value, mask = np.broadcast_arrays(ptr.offset, value, mask)
        ptr.storage[offsets[mask]] = value[mask]


def load_kernels(path, names):
    tl = ArrayLanguage()
    scope = {"tl": tl}
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    constants = [node for node in tree.body if isinstance(node, ast.Assign)]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    for node in functions:
        node.decorator_list = []
    exec(compile(ast.Module(body=constants + functions, type_ignores=[]), str(ROOT / path), "exec"), scope)
    return tl, scope


@pytest.fixture
def indexer():
    return load_kernels(
        "vllm_ascend/ops/triton/sharded_indexer.py",
        ("_pack_candidates", "_sort_candidate_pairs"),
    )


def test_distributed_topk_selects_sorted_transport_without_chip_probe(monkeypatch):
    _, scope = load_kernels(
        "vllm_ascend/attention/context_parallel/sharded_indexer.py",
        ("gathered_candidate_row_tile", "distributed_topk"),
    )
    ops = ModuleType("vllm_ascend.ops.triton.sharded_indexer")
    ops.MAX_FUSED_TOPK = 2048
    ops.pack_local_candidates = Mock(return_value=SimpleNamespace(shape=(128, 2048, 2)))
    ops.merge_packed_candidates = Mock()
    probe = Mock(return_value=False)
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setitem(sys.modules, "vllm_ascend.device.device_config", SimpleNamespace(is_950=probe))
    output = MagicMock()
    scope["torch"] = SimpleNamespace(empty=Mock(return_value=output), int32=np.int32)
    indices = MagicMock(shape=(128, 1, 2048), device="npu")
    group = SimpleNamespace(world_size=8, rank_in_group=7, all_gather=Mock())

    assert scope["distributed_topk"](indices, MagicMock(), group, 2048, 768) is output
    assert ops.pack_local_candidates.call_args.kwargs["sort_candidates"] is True
    assert ops.merge_packed_candidates.call_args.kwargs["presorted"] is True
    group.all_gather.assert_called_once()
    probe.assert_not_called()


def test_supported_topk_uses_fused_merge_without_chip_probe():
    _, scope = load_kernels("vllm_ascend/ops/triton/sharded_indexer.py", ("merge_packed_candidates",))
    kernel = MagicMock()
    probe = Mock(return_value=False)
    scope.update(
        triton=SimpleNamespace(
            next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
            cdiv=lambda n, d: (n + d - 1) // d,
        ),
        init_device_properties_triton=Mock(),
        get_vectorcore_num=lambda: 64,
        _merge_candidates=kernel,
        is_950=probe,
    )
    packed = SimpleNamespace(shape=(1024, 2048, 2), device="npu")
    output = object()

    assert scope["merge_packed_candidates"](packed, 128, 8, 2048, out=output, presorted=True) is output
    kernel.__getitem__.assert_called_once_with((64,))
    kernel.__getitem__.return_value.assert_called_once()
    probe.assert_not_called()


def test_grouped_dispatch_preserves_original_owner_coordinates():
    _, scope = load_kernels("vllm_ascend/ops/triton/sharded_indexer.py", ("merge_packed_candidates",))
    kernel = MagicMock()
    partial = SimpleNamespace(shape=(512, 2048, 2), device="npu")
    scope.update(
        triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length(), cdiv=lambda n, d: (n + d - 1) // d),
        init_device_properties_triton=Mock(),
        get_vectorcore_num=lambda: 40,
        _merge_candidates=kernel,
        torch=SimpleNamespace(empty=Mock(return_value=partial), int32=np.int32),
    )
    packed = SimpleNamespace(shape=(2048, 2048, 2), device="npu")
    output = object()
    assert scope["merge_packed_candidates"](packed, 128, 16, 2048, out=output, local_rank=15, presorted=True) is output
    first, final = kernel.__getitem__.return_value.call_args_list
    assert first.args[5] == 16 and first.kwargs["GROUP_SIZE"] == 4
    assert final.args[0] is partial and final.args[5] == 4
    assert final.args[6] == 15 and final.kwargs["OWNER_WORLD"] == 16


@pytest.mark.parametrize("rows", [0, 1, 64, 65, 128, 256, 257, 2048, 2049])
def test_pr35_decode_ceiling_and_budget_boundaries(rows):
    _, scope = load_kernels(
        "vllm_ascend/attention/context_parallel/sharded_indexer.py", ("gathered_candidate_row_tile",)
    )
    tile = scope["gathered_candidate_row_tile"](rows, 8, 2048)
    assert tile == (2048 if rows else 0)
    if rows:
        assert len(range(0, rows, tile)) == (rows + 2047) // 2048
        scope["MAX_GATHERED_CANDIDATE_BYTES"] = 1
        assert scope["gathered_candidate_row_tile"](rows, 8, 2048) == 256


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("world,interleave", [(1, 1), (4, 128), (8, 768), (16, 768)])
def test_pack_casts_before_bit_transport_and_masks_tail(indexer, dtype, world, interleave):
    tl, kernels = indexer
    rows, k, rank = 9, 7, world - 1
    # Strided inputs and a final partial rows-per-program tile.
    ids = np.arange(rows * k * 2, dtype=np.int32).reshape(rows, k * 2)
    scores = np.linspace(-4, 4, rows * k * 2).astype(dtype).reshape(rows, k * 2)
    scores[:, :6:2] = [np.nan, np.inf, -np.inf]
    packed = np.full((rows, k, 2), 123, dtype=np.int32)
    tl.grid = (2, 1)
    for program in range(2):
        tl.pid = (program, 0)
        kernels["_pack_candidates"](
            Pointer(ids), Pointer(scores), Pointer(packed), rows, 2 * k, 2 * k, 2, 2, k, world, rank, interleave, 8, 8
        )
    local, values = ids[:, ::2], scores[:, ::2].astype(np.float32)
    valid = np.isfinite(values)
    global_ids = (local // interleave * world + rank) * interleave + local % interleave
    expected = np.stack((np.where(valid, global_ids, -1), np.where(valid, values, -np.inf).view(np.int32)), -1)
    np.testing.assert_array_equal(packed, expected)


def test_pack_native_sort_fp32_id_keys_and_invalid_sentinel(indexer):
    tl, kernels = indexer
    ids = np.array([[0, 767, 768, 1048575, np.inf, np.inf, np.inf, np.inf]], dtype=np.float32)
    scores = np.array([[3, 2, 1, -1, -np.inf, -np.inf, -np.inf, -np.inf]], dtype=np.float32)
    packed = np.full((1, 8, 2), 123, dtype=np.int32)
    tl.pid = (0, 0)
    kernels["_pack_candidates"](Pointer(ids), Pointer(scores), Pointer(packed), 1, 8, 8, 1, 1, 8, 16, 15, 768, 8, 1)
    local = ids[0, :4].astype(np.int32)
    expected = np.full(8, -1, dtype=np.int32)
    expected[:4] = local // 768 * (16 * 768) + 15 * 768 + local % 768
    np.testing.assert_array_equal(packed[0, :, 0], expected)
    np.testing.assert_array_equal(packed[0, :, 1], scores[0].view(np.int32))


@pytest.mark.parametrize("world", [1, 3, 8])
@pytest.mark.parametrize("rows,topk", [(3, 33), (128, 2048)])
def test_pack_can_establish_exact_order_before_collective(indexer, world, rows, topk):
    tl, kernels = indexer
    rng = np.random.default_rng(35)
    ids = rng.integers(0, 32768, (rows, topk), dtype=np.int32)
    scores = rng.integers(-2, 3, (rows, topk)).astype(np.float32)
    ids[0] = -1
    scores[:, :3] = [np.nan, np.inf, -np.inf]
    packed = np.empty((rows, topk, 2), dtype=np.int32)
    block = 1 << (topk - 1).bit_length()
    for row in range(rows):
        tl.pid = (row, 0)
        kernels["_pack_candidates"](
            Pointer(ids),
            Pointer(scores),
            Pointer(packed),
            rows,
            topk,
            topk,
            1,
            1,
            topk,
            world,
            0,
            768,
            block,
            1,
            block.bit_length() - 1,
        )
        valid = (ids[row] >= 0) & np.isfinite(scores[row])
        global_ids = ids[row] // 768 * (world * 768) + ids[row] % 768
        values = np.where(valid, scores[row], -np.inf)
        order = np.lexsort((np.where(valid, global_ids, np.inf), -values))
        np.testing.assert_array_equal(packed[row, :, 0], np.where(valid, global_ids, -1)[order])
        np.testing.assert_array_equal(packed[row, :, 1], values[order].view(np.int32))


@pytest.mark.parametrize("rows,k", [(1, 7), (9, 2048)])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_sort_preparation_masks_invalid_scores_and_preserves_bits(rows, k, dtype):
    tl, kernels = load_kernels("vllm_ascend/ops/triton/sharded_indexer.py", ("_prepare_local_sort_inputs",))
    rng = np.random.default_rng(44)
    ids = rng.integers(0, (1 << 24) + 1, (rows, k * 2), dtype=np.int32)
    scores = rng.standard_normal((rows, k * 2)).astype(dtype)
    ids[:, 6] = -1
    ids[:, 8] = 1 << 24
    scores[:, :6:2] = [np.nan, np.inf, -np.inf]
    scores[:, 10] = -0.0
    keys = np.full((rows, k), -123, dtype=np.float32)
    values = np.full_like(keys, np.nan)
    tl.grid = (3,)
    for program in range(3):
        tl.pid = (program,)
        kernels["_prepare_local_sort_inputs"](
            Pointer(ids),
            Pointer(scores),
            Pointer(keys),
            Pointer(values),
            rows,
            k * 2,
            2,
            k * 2,
            2,
            k,
            1 << (k - 1).bit_length(),
        )
    ids, scores = ids[:, ::2], scores[:, ::2].astype(np.float32)
    valid = (ids >= 0) & np.isfinite(scores)
    np.testing.assert_array_equal(keys, np.where(valid, ids.astype(np.float32), np.inf))
    expected = np.where(valid, scores, -np.inf)
    np.testing.assert_array_equal(values.view(np.int32), expected.view(np.int32))


@pytest.mark.parametrize("world", [1, 3, 8, 16])
@pytest.mark.parametrize("local_rank", [None, 0, 2])
@pytest.mark.parametrize("local_k,topk", [(7, 1), (7, 3), (31, 16), (33, 31), (2048, 2048)])
@pytest.mark.parametrize("presorted", [False, True])
@pytest.mark.parametrize("one_owner_wins", [False, True])
def test_fused_merge_exact_order_and_program_owned_rows(world, local_rank, local_k, topk, presorted, one_owner_wins):
    tl, kernels = load_kernels(
        "vllm_ascend/ops/triton/sharded_indexer.py",
        ("_sort_candidate_pairs", "_merge_candidates", "_finalize_candidates"),
    )
    kernels["_a5_sort"] = lambda value, descending=False: array(np.sort(value)[::-1] if descending else np.sort(value))
    rng = np.random.default_rng(29)
    rows, programs, block = 3, 2, 1 << (topk - 1).bit_length()
    ids = rng.integers(0, 1 << 24, (world, rows, local_k), dtype=np.int32)
    scores = rng.integers(-2, 3, ids.shape).astype(np.float32)
    if one_owner_wins:
        scores[-1] += 16
    ids[:, 0] = -1
    scores[..., :3] = [np.nan, np.inf, -np.inf]
    ids[0, 1, -1] = 1 << 24
    scores[0, 1, -1] = np.nextafter(np.float32(2), np.float32(3))
    valid = (ids >= 0) & np.isfinite(scores)
    packed = np.stack((ids, scores.view(np.int32)), -1)
    if presorted:
        order = np.lexsort((np.where(valid, ids, np.inf), -np.where(valid, scores, -np.inf)), axis=-1)
        packed = np.take_along_axis(packed, order[..., None], axis=2)
    output = np.full((rows, topk), -123, dtype=np.int32)
    tl.grid = (programs,)
    for program in range(programs):
        tl.pid = (program,)
        kernels["_merge_candidates"](
            Pointer(packed),
            Pointer(output),
            rows,
            local_k,
            topk,
            world,
            -1 if local_rank is None else local_rank,
            768,
            block.bit_length() - 1,
            presorted,
        )
    for row in range(rows):
        row_ids, row_scores = ids[:, row].reshape(-1), scores[:, row].reshape(-1)
        keep = valid[:, row].reshape(-1)
        selected = np.lexsort((row_ids[keep], -row_scores[keep]))[:topk]
        winners = row_ids[keep][selected]
        if local_rank is not None:
            winners = winners[(winners // 768) % world == local_rank]
            winners = winners // (world * 768) * 768 + winners % 768
        expected = np.full(topk, -1, dtype=np.int32)
        expected[: len(winners)] = winners
        np.testing.assert_array_equal(output[row], expected)



@pytest.mark.parametrize("world", [1, 3, 8, 16, 17])
@pytest.mark.parametrize("group_size", [2, 4, 8])
@pytest.mark.parametrize("topk", [7, 2048])
@pytest.mark.parametrize("return_local", [False, True])
def test_grouped_merge_preserves_global_selection_and_owner_mapping(world, group_size, topk, return_local):
    tl, kernels = load_kernels(
        "vllm_ascend/ops/triton/sharded_indexer.py",
        ("_sort_candidate_pairs", "_merge_candidates", "_finalize_candidates"),
    )
    kernels["_a5_sort"] = lambda value, descending=False: array(np.sort(value)[::-1] if descending else np.sort(value))
    rng = np.random.default_rng(43)
    rows, programs, block = 3, 2, 1 << (topk - 1).bit_length()
    ids = rng.integers(0, 1 << 24, (world, rows, topk), dtype=np.int32)
    scores = rng.integers(-2, 3, ids.shape).astype(np.float32)
    ids[:, 0] = -1
    scores[..., :3] = [np.nan, np.inf, -np.inf]
    ids[-1, 1, -1] = 1 << 24
    scores[-1, 1, -1] = np.nextafter(np.float32(2), np.float32(3))
    valid = (ids >= 0) & np.isfinite(scores)
    order = np.lexsort((np.where(valid, ids, np.inf), -np.where(valid, scores, -np.inf)), axis=-1)
    packed = np.take_along_axis(np.stack((ids, scores.view(np.int32)), -1), order[..., None], axis=2)
    groups = (world + group_size - 1) // group_size
    partial = np.full((groups, rows, topk, 2), 123, dtype=np.int32)
    output = np.full((rows, topk), -123, dtype=np.int32)
    owner = world - 1 if return_local else -1
    tl.grid = (programs,)
    for source, target, count, group, local in (
        (packed, partial, world, group_size, -1),
        (partial, output, groups, 0, owner),
    ):
        for program in range(programs):
            tl.pid = (program,)
            kernels["_merge_candidates"](
                Pointer(source),
                Pointer(target),
                rows,
                topk,
                topk,
                count,
                local,
                768,
                block.bit_length() - 1,
                True,
                group,
                world,
            )
    for row in range(rows):
        row_ids, row_scores = ids[:, row].reshape(-1), scores[:, row].reshape(-1)
        keep = valid[:, row].reshape(-1)
        selected = np.lexsort((row_ids[keep], -row_scores[keep]))[:topk]
        winners = row_ids[keep][selected]
        if return_local:
            winners = winners[(winners // 768) % world == owner]
            winners = winners // (world * 768) * 768 + winners % 768
        expected = np.full(topk, -1, dtype=np.int32)
        expected[: len(winners)] = winners
        np.testing.assert_array_equal(output[row], expected)


@pytest.mark.parametrize("world", [1, 3, 8])
@pytest.mark.parametrize("has_local", [False, True])
@pytest.mark.parametrize("block_rows", [8, 16, 32])
@pytest.mark.parametrize("scale", [1.0, 2.0**-100, 2.0**100])
@pytest.mark.parametrize("coalesced", [False, True])
def test_lse_combine_invalid_ranks_and_strided_local(world, has_local, block_rows, scale, coalesced):
    tl, kernels = load_kernels(
        "vllm_ascend/ops/triton/dcp/dcp_a2a_batched.py", ("_fused_dcp_lse_combine_batched_kernel",)
    )
    rng = np.random.default_rng(17)
    tokens, heads, dim = 17, 3, 32
    values = rng.normal(size=(world, heads, tokens, dim)).astype(np.float32)
    # Integer LSE values have an independently constructed exact four-field encoding.
    lses = rng.integers(-127, 128, (world, heads, tokens)).astype(np.float32)
    lses *= scale
    lses[:, 0, 0] = -np.inf
    values[:, 0, 0] = np.nan
    lses[0, 1, 0] = -np.inf
    values[0, 1, 0] = np.nan
    safe = np.where(np.isfinite(lses) & (lses != 0), np.abs(lses), 1)
    exponent = np.floor(np.log2(safe))
    mantissa = np.abs(np.where(np.isfinite(lses), lses, 0)) * np.exp2(23 - exponent)
    hi = np.floor(mantissa / 65536)
    mid = np.floor((mantissa - hi * 65536) / 256)
    lo = mantissa - hi * 65536 - mid * 256
    code = np.where(lses < 0, -(exponent + 128), exponent + 128)
    code = np.where(np.isfinite(lses), code, 0)
    recv = np.concatenate((values, np.stack((code, hi, mid, lo), -1)), -1).astype(np.float32)
    local = rng.normal(size=(tokens, heads, dim * 2)).astype(np.float32)
    local_lse = rng.normal(size=(tokens, heads, 2)).astype(np.float32)
    local_lse[0, 0, 0] = -np.inf
    local[0, 0] = np.nan
    output = np.full((tokens, heads, dim), np.nan, dtype=np.float32)
    strides = tuple(s // recv.itemsize for s in recv.strides)
    tl.grid = (2,)
    with np.errstate(invalid="ignore", over="ignore"):
        for program in range(2):
            tl.pid = (program,)
            kernels["_fused_dcp_lse_combine_batched_kernel"](
                Pointer(recv),
                Pointer(output),
                Pointer(local),
                Pointer(local_lse),
                heads * dim * 2,
                dim * 2,
                2,
                heads * 2,
                2,
                *strides,
                heads * dim,
                dim,
                1,
                dim,
                heads,
                tokens * heads,
                world,
                1 << (world - 1).bit_length(),
                dim,
                block_rows,
                has_local,
                coalesced,
            )
    all_values = values.transpose(0, 2, 1, 3)
    all_lses = lses.transpose(0, 2, 1)
    if has_local:
        all_values = np.concatenate((all_values, local[None, ..., ::2]), 0)
        all_lses = np.concatenate((all_lses, local_lse[None, ..., 0]), 0)
    valid = np.isfinite(all_lses)
    maximum = np.max(np.where(valid, all_lses, -np.inf), axis=0)
    maximum = np.where(np.isfinite(maximum), maximum, 0)
    weights = np.where(valid, np.exp(np.where(valid, all_lses, -np.inf) - maximum), 0)
    total = weights.sum(axis=0)
    expected = (np.where(valid[..., None], all_values, 0) * weights[..., None]).sum(axis=0)
    expected /= np.where(total > 0, total, 1)[..., None]
    np.testing.assert_allclose(output, expected, atol=3e-6, rtol=3e-5)
