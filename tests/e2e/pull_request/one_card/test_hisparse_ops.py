# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU interpreter execution of production kernels, not NPU qualification.

Run with tools/run_hisparse_ops_tests.sh cpu [all|resolver|gather|regression|invalidation].
The public NPU guard is deliberately bypassed through private launch methods.
"""

import importlib.util
import os
import random
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest
import torch
from vllm.triton_utils import triton

from .hisparse_reference import gather_reference, resolve_reference

InterpreterError = triton.runtime.errors.InterpreterError


@pytest.fixture(scope="module")
def ops_module():
    assert os.environ.get("TRITON_INTERPRET") == "1", "this suite requires the CPU interpreter"
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/triton/hisparse.py"
    spec = importlib.util.spec_from_file_location("hisparse_cpu_production", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # triton-ascend 3.6.0 GridExecutor discards launch debug= and does not
    # propagate TRITON_DEBUG to InterpreterOptions. Enable its real assertions,
    # without replacing a kernel or computing any expected result here.
    interpreter_builder = import_module("triton.runtime.interpreter").interpreter_builder
    previous = interpreter_builder.options
    interpreter_builder.options = replace(previous, debug=True)
    try:
        yield module
    finally:
        interpreter_builder.options = previous


class CpuMapping:
    """Only tests supply raw CPU addresses; production requires backend mapping."""

    device = torch.device("cpu")

    def __init__(self, storage):
        self.storage = storage

    def address_of(self, host):
        start = self.storage.untyped_storage().data_ptr()
        end = start + self.storage.untyped_storage().nbytes()
        assert start <= host.data_ptr() and host.data_ptr() + host.numel() * host.element_size() <= end
        return host.data_ptr()


def matrix(values):
    return torch.tensor(values, dtype=torch.int32)


def fixture_inputs(indices, *, requests=None, states=None, hot_size=8, union=8, optional=63, strided=False):
    rows, k = len(indices), len(indices[0])
    nstates = max(rows, 4)
    host = torch.arange(128 * 16, dtype=torch.int32).to(torch.uint8).reshape(128, 16)
    hot = torch.full((32, 2, 16), 151, dtype=torch.uint8)
    raw = torch.full((rows, k + 3), -97, dtype=torch.int32)
    raw[:, :k] = matrix(indices)
    table = torch.full((max(rows, 4), hot_size // 2 + 2), -98, dtype=torch.int32)
    for req in range(table.shape[0]):
        table[req, : hot_size // 2] = torch.arange(2 + req * (hot_size // 2), 2 + (req + 1) * (hot_size // 2))
    args = dict(
        host_cache=host,
        hot_cache=hot,
        hot_block_table=table[:, : hot_size // 2],
        global_indices=raw[:, :k] if strided else matrix(indices),
        hot_indices=torch.full((rows, k), -77, dtype=torch.int32),
        device_global_indices=torch.full((nstates, hot_size), -1, dtype=torch.int32),
        lru_slots=torch.arange(hot_size, dtype=torch.int16).repeat(nstates, 1),
        request_state_indices=None if states is None else matrix(states),
        region_stride=hot_size,
        max_union_rows=union,
    )
    if requests is not None:
        args.update(
            request_ids=matrix(requests),
            source_block_table=matrix([[i + 1 for i in range(32)]] * 4),
            source_block_size=2,
        )
    if optional & 1:
        args["miss_mask"] = torch.full((rows, k), -77, dtype=torch.int32)
    if optional & 2:
        args["stats"] = torch.tensor([2**53 + 7, 11], dtype=torch.uint64)
    if optional & 4:
        args["attention_indices"] = torch.full((rows, k + 3), -77, dtype=torch.int32)[:, :k]
        args["attention_block_stride"] = 5
    if optional & 8:
        args["resolved_global_indices"] = torch.full((rows, k), -77, dtype=torch.int32)
    if optional & 16:
        args["valid_counts"] = torch.full((rows * 2,), -77, dtype=torch.int32)[::2]
    if optional & 32:
        args.update(
            swap_host_physical_rows=torch.full((rows, k), -77, dtype=torch.int32),
            swap_device_physical_rows=torch.full((rows, k), -77, dtype=torch.int32),
            swap_counts=torch.full((rows,), -77, dtype=torch.int32),
        )
    return args


OUTPUTS = (
    "hot_indices",
    "device_global_indices",
    "lru_slots",
    "miss_mask",
    "stats",
    "attention_indices",
    "resolved_global_indices",
    "valid_counts",
    "swap_host_physical_rows",
    "swap_device_physical_rows",
    "swap_counts",
)


def compare_resolver(module, args, workspace=None):
    tensors = {
        key: value.tolist()
        for key, value in args.items()
        if isinstance(value, torch.Tensor) and key not in ("host_cache", "hot_cache")
    }
    expected = resolve_reference(
        tensors,
        args["host_cache"].shape[0],
        args["hot_cache"].shape[1],
        args.get("attention_block_stride", 0),
        args.get("source_block_size", 0),
        args.get("resident_block_size", 0),
        args.get("resident_null_block", 0),
        args["max_union_rows"],
    )
    rows, k = args["global_indices"].shape
    h = args["region_stride"]
    w = workspace or module.HiSparseResolverWorkspace("cpu", rows, args["lru_slots"].shape[0], k, h, h)
    payload = args["hot_cache"].clone()
    input_indices = args["global_indices"].clone()
    adapter = module.HiSparseTritonCacheOps(w, CpuMapping(args["host_cache"]))
    adapter._resolve(**args)
    for name in OUTPUTS:
        if name in args:
            assert args[name].tolist() == expected[name], name
    assert torch.equal(args["hot_cache"], payload), "resolver must not copy payload"
    assert torch.equal(args["global_indices"], input_indices)
    if "attention_indices" in args:
        assert torch.all(args["attention_indices"]._base[:, k:] == -77)
    if "valid_counts" in args:
        assert torch.all(args["valid_counts"]._base[1::2] == -77)
    return adapter


@pytest.mark.parametrize("optional", range(64))
def test_resolver_optional_outputs_and_defaults(ops_module, optional):
    compare_resolver(ops_module, fixture_inputs([[2, 7, -1, 127]], optional=optional))


@pytest.mark.parametrize(
    "scenario",
    [
        "cold",
        "hit",
        "mixed",
        "shared",
        "resident",
        "all_resident",
        "padding",
        "reorder",
        "invalid",
        "bad_lru",
        "integer",
        "overflow_stats",
    ],
)
def test_resolver_cuda_contract(ops_module, scenario):
    args = fixture_inputs([[0, 1, 2, 3], [2, 3, 4, 5]], requests=[1, 1], states=[2, 0, 1, 3], strided=True)
    if scenario in ("hit", "mixed"):
        args["device_global_indices"][0, :6] = matrix([2, 3, 4, 5, 6, 7])
        if scenario == "mixed":
            args["device_global_indices"][0, :3] = -1
        args["lru_slots"][0] = torch.tensor([3, 0, 5, 1, 7, 2, 4, 6], dtype=torch.int16)
    if scenario in ("resident", "all_resident"):
        table = torch.zeros((4, 32), dtype=torch.int32)
        count = 3 if scenario == "all_resident" else 1
        table[0, :count] = matrix([20, 21, 22][:count])
        args.update(resident_block_table=table, resident_block_size=2)
    if scenario == "padding":
        args["request_ids"] = matrix([1, -1])
        args["num_valid_rows"] = matrix([1])
    if scenario == "reorder":
        args["request_ids"] = matrix([2, 0])
    if scenario == "invalid":
        args["global_indices"] = matrix([[-1, 1000, 4, 5], [0, 1, 6, 7]])
        args["source_block_table"][1, 2] = 0
    if scenario == "bad_lru":
        args["lru_slots"][0, 0] = -1
    if scenario == "integer":
        # Untouched allocation supplies real legal geometry; resolver never
        # reads its payload. Odd host IDs above 2**24 must remain exact.
        args["host_cache"] = torch.empty((16777224, 16), dtype=torch.uint8)
        args["source_block_table"][1, :3] = matrix([8388609, 8388610, 8388611])
    if scenario == "overflow_stats":
        args["stats"] = torch.tensor([2**64 - 2, 2**64 - 2], dtype=torch.uint64)
    compare_resolver(ops_module, args)


@pytest.mark.parametrize("k", [1, 8, 32, 128])
def test_resolver_multistep_stable_eviction(ops_module, k):
    rng = random.Random(932 + k)
    args = fixture_inputs([list(range(k))], hot_size=2 * k, union=2 * k)
    # Larger logical geometry is valid here, no gather is attempted.
    args["hot_cache"] = torch.zeros((2 * k + 2, 2, 16), dtype=torch.uint8)
    args["hot_block_table"] = torch.arange(k, dtype=torch.int32).reshape(1, k)
    w = ops_module.HiSparseResolverWorkspace("cpu", 1, 4, k, 2 * k, 2 * k)
    for step in range(5):
        values = rng.sample(range(128), k)
        if step == 1:
            values = args["global_indices"][0].tolist()
        args["global_indices"] = matrix([values])
        compare_resolver(ops_module, args, w)


def test_resolver_union_bound_is_error(ops_module):
    args = fixture_inputs([[0, 1, 2, 3], [4, 5, 6, 7]], requests=[0, 0], states=[0, 1, 2, 3], union=4)
    w = ops_module.HiSparseResolverWorkspace("cpu", 2, 4, 4, 8, 4)
    w.order.fill_(-91)
    ownership = args["device_global_indices"].clone()
    with pytest.raises(InterpreterError, match="union exceeds"):
        ops_module.HiSparseTritonCacheOps(w)._resolve(**args)
    assert torch.equal(args["device_global_indices"], ownership)
    assert torch.all(w.order[1:] == -91)


@pytest.mark.parametrize(
    "case",
    [
        "state_padding",
        "valid_zero",
        "valid_clipped",
        "no_states",
        "different_blocks",
        "resident_zero",
        "hash_collision",
    ],
)
def test_resolver_optional_translation_inputs(ops_module, case):
    args = fixture_inputs([[0, 1, 2, 3], [2, 3, 4, 5]], requests=[1, 1], states=[2, 0, 1, 3])
    if case == "state_padding":
        args["request_state_indices"][1] = -1
    if case in ("valid_zero", "valid_clipped"):
        args["num_valid_rows"] = matrix([0 if case == "valid_zero" else 99])
    if case == "no_states":
        args["request_state_indices"] = None
    if case == "different_blocks":
        args["source_block_size"] = 4
        args.update(resident_block_table=matrix([[0, 0, 23]] * 4), resident_block_size=2)
    if case == "resident_zero":
        args.update(resident_block_table=matrix([[0, 9, -1]] * 4), resident_block_size=2, resident_null_block=9)
    if case == "hash_collision":
        args["source_block_table"][1, :4] = matrix([1, 9, 17, 25])
        args["global_indices"] = matrix([[0, 2, 4, 6], [0, 2, 4, 6]])
    compare_resolver(ops_module, args)


def test_resolver_reads_current_request_ids(ops_module):
    args = fixture_inputs([[0, 1, 2, 3]] * 4, requests=[1, 1, 1, 1], states=[2, 0, 1, 3])
    adapter = compare_resolver(ops_module, args)
    args["request_ids"].copy_(matrix([1, -1, -1, -1]))
    args["num_valid_rows"] = matrix([1])
    compare_resolver(ops_module, args, adapter.workspace)


def test_resolver_speculative_rows_count_a_shared_load_once(ops_module):
    # Upstream keeps K=128, two identical query rows, block_size=64.
    k = 128
    args = fixture_inputs([list(range(k))] * 2, requests=[0, 0], states=[0], hot_size=256, union=128)
    args.update(
        host_cache=torch.zeros((192, 256), dtype=torch.uint8),
        hot_cache=torch.zeros((5, 64, 64), dtype=torch.float32),
        hot_block_table=matrix([[1, 2, 3, 4]]),
        source_block_table=matrix([[1, 2]]),
        source_block_size=64,
        stats=torch.zeros((2,), dtype=torch.uint64),
        num_valid_rows=matrix([2]),
        attention_block_stride=64,
    )
    compare_resolver(ops_module, args)
    assert args["stats"].tolist() == [k, k]
    assert args["swap_counts"].tolist() == [k, 0]
    assert args["miss_mask"].sum(dim=1).tolist() == [k, 0]


@pytest.mark.parametrize("width", [16, 64, 256, 528, 656])
@pytest.mark.parametrize("layout", ["staging", "paged", "compact"])
def test_gather_bytes_layout_and_guards(ops_module, width, layout):
    block = 1 if layout == "staging" else 2
    pages, offset = 8, 1
    gap = 17
    page_bytes = block * width + gap
    storage = torch.full((offset + pages * page_bytes + 19,), 173, dtype=torch.uint8)
    shape = (pages, width) if block == 1 else (pages, block, width)
    strides = (page_bytes, 1) if block == 1 else (page_bytes, width, 1)
    hot = storage.as_strided(shape, strides, storage_offset=offset)
    # Host subview begins after two rows; include arbitrary bit patterns.
    host_storage = ((torch.arange(9 * width, dtype=torch.int32) * 37 + 139) % 256).to(torch.uint8).reshape(9, width)
    host = host_storage[2:7]
    source = matrix([[0, 1, -1, 9], [2, 3, 4, 1], [1, 2, 3, 4], [9, 1, 2, 3]])
    target = matrix([[0, 1, 2, 3], [4, 5, 6, pages * block], [1, 2, 3, 4], [-1, 5, 6, 7]])
    mask = matrix([[1, 0, 1, 2], [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 0, 0]])
    counts = matrix([99, 2, -7, 0]) if layout == "compact" else None
    states = None if counts is not None else matrix([2, 0, -1, 1])
    expected, att = gather_reference(
        storage.tolist(),
        host.tolist(),
        source.tolist(),
        target.tolist(),
        width=width,
        block=block,
        page_bytes=page_bytes,
        view_offset=offset,
        hot_rows=pages * block,
        mask=mask.tolist(),
        counts=None if counts is None else counts.tolist(),
        states=None if states is None else states.tolist(),
        attention_stride=block + 3,
    )
    w = ops_module.HiSparseResolverWorkspace("cpu", 4, 4, 4, 8, 8)
    adapter = ops_module.HiSparseTritonCacheOps(w, CpuMapping(host_storage))
    attention = torch.full_like(source, -77) if counts is None else None
    adapter._gather(
        host,
        hot,
        source,
        target,
        mask=None if counts is not None else mask,
        counts=counts,
        states=states,
        attention=attention,
        attention_block_stride=block + 3,
    )
    assert storage.tolist() == expected
    if attention is not None:
        assert attention.tolist() == att


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
def test_gather_preserves_payload_bits_across_dtypes(ops_module, dtype):
    host = torch.tensor([[0, 0, 192, 127, 255, 255, 255, 255] * 2], dtype=torch.uint8)
    hot = torch.zeros((1, 1, 16 // torch.empty((), dtype=dtype).element_size()), dtype=dtype)
    w = ops_module.HiSparseResolverWorkspace("cpu", 1, 1, 1, 1, 1)
    ops_module.HiSparseTritonCacheOps(w, CpuMapping(host))._gather(
        host, hot, matrix([[0]]), matrix([[0]]), counts=matrix([1])
    )
    assert hot.view(torch.uint8).flatten().tolist() == host.flatten().tolist()


def test_gather_zeroes_unaligned_destination(ops_module):
    # Exact upstream values, stride and guard; only platform launch differs.
    host = torch.ones((1, 16), dtype=torch.uint8)
    storage = torch.full((33,), 7, dtype=torch.uint8)
    hot = storage.as_strided((2, 1, 16), (17, 16, 1))
    w = ops_module.HiSparseResolverWorkspace("cpu", 1, 1, 1, 1, 1)
    ops_module.HiSparseTritonCacheOps(w, CpuMapping(host))._gather(
        host, hot, matrix([[1]]), matrix([[1]]), mask=matrix([[1]])
    )
    assert hot[1, 0].tolist() == [0] * 16
    assert storage[:17].tolist() == [7] * 17


def test_gather_rejects_short_attention_block_stride(ops_module):
    host = torch.ones((1, 16), dtype=torch.uint8)
    hot = torch.empty((1, 2, 16), dtype=torch.uint8)
    w = ops_module.HiSparseResolverWorkspace("cpu", 1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="shorter than a block"):
        ops_module.HiSparseTritonCacheOps(w, CpuMapping(host))._gather(
            host,
            hot,
            matrix([[0]]),
            matrix([[0]]),
            mask=matrix([[1]]),
            attention=matrix([[0]]),
            attention_block_stride=1,
        )


def test_regression_remaps_strided_hma_rows_for_attention(ops_module):
    args = fixture_inputs([[0, 1, 2, 3]])
    storage = torch.full((448,), 79, dtype=torch.uint8)
    args["hot_cache"] = storage.as_strided((2, 4, 16), (192, 16, 1), storage_offset=64)
    args["hot_block_table"] = matrix([[0, 1]])
    args["attention_block_stride"] = 12
    args["lru_slots"][0] = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3], dtype=torch.int16)
    adapter = compare_resolver(ops_module, args)
    assert args["attention_indices"].tolist() == [[12, 13, 14, 15]]
    expected, _ = gather_reference(
        storage.tolist(),
        args["host_cache"].tolist(),
        args["swap_host_physical_rows"].tolist(),
        args["swap_device_physical_rows"].tolist(),
        width=16,
        block=4,
        page_bytes=192,
        view_offset=64,
        hot_rows=8,
        counts=args["swap_counts"].tolist(),
    )
    adapter._gather(
        args["host_cache"],
        args["hot_cache"],
        args["swap_host_physical_rows"],
        args["swap_device_physical_rows"],
        counts=args["swap_counts"],
    )
    assert storage.tolist() == expected


@pytest.mark.parametrize("width", [256, 528, 656])
def test_regression_swap_in_preserves_rows_across_eviction(ops_module, width):
    args = fixture_inputs([[0, 1, 2, -1]] * 4, requests=[0, 1, 2, 3], states=[0, 1, 2, 3])
    args["host_cache"] = ((torch.arange(128 * width) * 71 + 3) % 256).to(torch.uint8).reshape(128, width)
    args["hot_cache"] = torch.zeros((32, 2, width), dtype=torch.uint8)
    args["stats"] = torch.zeros((2,), dtype=torch.uint64)
    w = ops_module.HiSparseResolverWorkspace("cpu", 4, 4, 4, 8, 8)
    for step in (0, 0, 1, 2, 0):
        args["global_indices"] = matrix([[(c + step * 4 + r * 17) % 64 for c in range(3)] + [-1] for r in range(4)])
        adapter = compare_resolver(ops_module, args, w)
        adapter._gather(
            args["host_cache"],
            args["hot_cache"],
            args["swap_host_physical_rows"],
            args["swap_device_physical_rows"],
            counts=args["swap_counts"],
        )
        for r in range(4):
            for c in range(3):
                src = int(args["resolved_global_indices"][r, c])
                dst = int(args["hot_indices"][r, c])
                assert torch.equal(args["hot_cache"].reshape(-1, width)[dst], args["host_cache"][src])
    hits, misses = args["stats"].tolist()
    assert hits >= 12 and misses > 36 and hits + misses == 60


def test_regression_resolve_gather_replay_and_rewrite(ops_module):
    args = fixture_inputs([[0, 1, 2, 3], [2, 3, 4, 5]], requests=[0, 0], states=[1, 0, 2, 3])
    adapter = compare_resolver(ops_module, args)
    adapter._gather(
        args["host_cache"],
        args["hot_cache"],
        args["swap_host_physical_rows"],
        args["swap_device_physical_rows"],
        counts=args["swap_counts"],
    )
    other_host = args["host_cache"] ^ 213
    other_hot = torch.full_like(args["hot_cache"], 19)
    replay = ops_module.HiSparseTritonCacheOps(adapter.workspace, CpuMapping(other_host))
    replay._gather(
        other_host,
        other_hot,
        args["resolved_global_indices"],
        args["hot_indices"],
        mask=args["miss_mask"],
        states=matrix([0, 0]),
    )
    for host, hot in ((args["host_cache"], args["hot_cache"]), (other_host, other_hot)):
        for r in range(2):
            for c in range(4):
                dst = int(args["hot_indices"][r, c])
                src = int(args["resolved_global_indices"][r, c])
                assert torch.equal(hot.reshape(-1, 16)[dst], host[src])
    # Rewrite host slot 4; invalidate every listed index group before reuse.
    args["host_cache"][4].fill_(231)
    table = args["device_global_indices"]
    other_table = table.clone()
    ptrs = torch.tensor([table.data_ptr(), other_table.data_ptr()], dtype=torch.int64)
    ops_module.hisparse_invalidate_written_slots_kernel[(1, table.shape[0], 2)](
        ptrs,
        args["request_state_indices"],
        matrix([0, 0]),
        torch.tensor([4, 4], dtype=torch.int64),
        2,
        4,
        REGION_STRIDE=8,
        BLOCK_SIZE=256,
    )
    assert 4 not in table[1].tolist() and 4 not in other_table[1].tolist()
    compare_resolver(ops_module, args, adapter.workspace)
    adapter._gather(
        args["host_cache"],
        args["hot_cache"],
        args["swap_host_physical_rows"],
        args["swap_device_physical_rows"],
        counts=args["swap_counts"],
    )
    dst = int(args["hot_indices"][1, 0])
    assert torch.equal(args["hot_cache"].reshape(-1, 16)[dst], args["host_cache"][4])


@pytest.mark.parametrize("failure", ["workspace", "compact", "attention", "mapping", "public"])
def test_resolver_gather_metadata_validation(ops_module, failure):
    args = fixture_inputs([[0, 1, 2, 3]])
    w = ops_module.HiSparseResolverWorkspace("cpu", 1, 4, 4, 8, 8)
    adapter = ops_module.HiSparseTritonCacheOps(w)
    if failure == "workspace":
        adapter.workspace = ops_module.HiSparseResolverWorkspace("cpu", 1, 4, 1, 8, 8)
    if failure == "compact":
        args.pop("swap_counts")
    if failure == "attention":
        args["attention_block_stride"] = 1
    with pytest.raises((ValueError, RuntimeError)):
        if failure == "mapping":
            adapter._gather(args["host_cache"], args["hot_cache"], matrix([[0]]), matrix([[0]]), counts=matrix([1]))
        elif failure == "public":
            adapter.hisparse_resolve_residency(**args)
        else:
            adapter._resolve(**args)


@pytest.mark.parametrize("dimension", ["rows", "top_k"])
def test_resolver_and_gather_empty_inputs(ops_module, dimension):
    args = fixture_inputs([[0, 1, 2, 3]])
    rows, k = (0, 4) if dimension == "rows" else (1, 0)
    for key in (
        "global_indices",
        "hot_indices",
        "miss_mask",
        "attention_indices",
        "resolved_global_indices",
        "swap_host_physical_rows",
        "swap_device_physical_rows",
    ):
        args[key] = torch.empty((rows, k), dtype=torch.int32)
    args["valid_counts"] = torch.full((rows,), -71, dtype=torch.int32)
    args["swap_counts"] = torch.full((rows,), -72, dtype=torch.int32)
    w = ops_module.HiSparseResolverWorkspace("cpu", 1, 4, 4, 8, 8)
    adapter = ops_module.HiSparseTritonCacheOps(w)
    stats = args["stats"].clone()
    ownership = args["device_global_indices"].clone()
    lru = args["lru_slots"].clone()
    hot = args["hot_cache"].clone()
    adapter._resolve(**args)
    adapter._gather(
        args["host_cache"], args["hot_cache"], args["global_indices"], args["hot_indices"], mask=args["miss_mask"]
    )
    adapter._gather(
        args["host_cache"],
        args["hot_cache"],
        args["swap_host_physical_rows"],
        args["swap_device_physical_rows"],
        counts=args["swap_counts"],
    )
    assert torch.equal(stats, args["stats"])
    assert torch.equal(ownership, args["device_global_indices"])
    assert torch.equal(lru, args["lru_slots"])
    assert torch.equal(hot, args["hot_cache"])
    assert args["valid_counts"].tolist() == [-71] * rows
    assert args["swap_counts"].tolist() == [-72] * rows
