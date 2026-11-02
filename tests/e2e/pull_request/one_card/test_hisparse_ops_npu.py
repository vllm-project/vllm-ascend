# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NPU execution; registered Host memory, exact independent CPU oracle."""

import ctypes
import importlib.util
import mmap
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from .hisparse_reference import gather_reference, resolve_reference
from .test_hisparse_ops import OUTPUTS, fixture_inputs, matrix


def load_module():
    assert torch.npu.is_available() and torch.npu.device_count() == 1
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/triton/hisparse.py"
    spec = importlib.util.spec_from_file_location("hisparse_production_npu", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def module():
    return load_module()


class RegisteredHost:
    """Test-owned live CANN registration; no plain CPU address fallback."""

    device = torch.device("npu:0")

    def __init__(self, source):
        torch.npu.set_device(self.device)
        self.api = ctypes.CDLL("libascendcl.so")
        self.api.aclrtHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.api.aclrtHostRegister.restype = ctypes.c_int
        self.api.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self.api.aclrtHostUnregister.restype = ctypes.c_int
        size = source.numel() * source.element_size()
        self.capacity = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        self.memory = mmap.mmap(-1, self.capacity)
        self.host = torch.frombuffer(self.memory, dtype=source.dtype, count=source.numel()).reshape(source.shape)
        self.host.copy_(source)
        self.pointer = ctypes.c_void_p()
        status = self.api.aclrtHostRegister(self.host.data_ptr(), self.capacity, 0, ctypes.byref(self.pointer))
        assert status == 0 and self.pointer.value, f"aclrtHostRegister failed: {status}"
        self.live = True

    def address_of(self, view):
        assert self.live and view.device.type == "cpu" and view.is_contiguous()
        offset = view.data_ptr() - self.host.data_ptr()
        assert offset >= 0 and offset + view.numel() * view.element_size() <= self.capacity
        return self.pointer.value + offset

    def close(self):
        torch.npu.synchronize()
        assert self.api.aclrtHostUnregister(self.host.data_ptr()) == 0
        self.live = False
        # External tensor views may outlive the mapping in a failed assertion.
        # mmap retains storage until their references are released.


def to_npu(tensor):
    if tensor._base is None:
        return tensor.to("npu")
    base = tensor
    while base._base is not None:
        base = base._base
    storage = base.to("npu")
    return storage.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())


def resolve(module, args, workspace=None):
    values = {
        name: value.tolist()
        for name, value in args.items()
        if isinstance(value, torch.Tensor) and name not in ("host_cache", "hot_cache")
    }
    expected = resolve_reference(
        values,
        args["host_cache"].shape[0],
        args["hot_cache"].shape[1],
        args.get("attention_block_stride", 0),
        args.get("source_block_size", 0),
        args.get("resident_block_size", 0),
        args.get("resident_null_block", 0),
        args["max_union_rows"],
    )
    actual = {
        name: to_npu(value) if isinstance(value, torch.Tensor) and name != "host_cache" else value
        for name, value in args.items()
    }
    rows, k = args["global_indices"].shape
    h = args["region_stride"]
    workspace = workspace or module.HiSparseResolverWorkspace("npu:0", rows, args["lru_slots"].shape[0], k, h, h)
    adapter = module.HiSparseTritonCacheOps(workspace)
    adapter.hisparse_resolve_residency(**actual)
    torch.npu.synchronize()
    for name in OUTPUTS:
        if name in args:
            got = actual[name].cpu()
            assert got.tolist() == expected[name], name
            args[name].copy_(got)
    assert torch.equal(actual["hot_cache"].cpu(), args["hot_cache"])
    return adapter, actual


def test_npu_resolver_smoke(module):
    args = fixture_inputs([[0, 1, 2, 3], [2, 3, 4, 5]], requests=[0, 0], states=[1, 0, 2, 3])
    resolve(module, args)


@pytest.mark.parametrize("optional", range(64))
def test_npu_resolver_optional_outputs(module, optional):
    resolve(module, fixture_inputs([[2, 7, -1, 127]], optional=optional, strided=True))


@pytest.mark.parametrize(
    "scenario",
    [
        "hit",
        "mixed",
        "resident",
        "all_resident",
        "padding",
        "reorder",
        "invalid",
        "bad_lru",
        "different_blocks",
        "resident_zero",
        "stats_overflow",
    ],
)
def test_npu_resolver_contract(module, scenario):
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
    if scenario == "different_blocks":
        args["source_block_size"] = 4
        args.update(resident_block_table=matrix([[0, 0, 23]] * 4), resident_block_size=2)
    if scenario == "resident_zero":
        args.update(resident_block_table=matrix([[0, 9, -1]] * 4), resident_block_size=2, resident_null_block=9)
    if scenario == "stats_overflow":
        args["stats"] = torch.tensor([2**64 - 2, 2**64 - 2], dtype=torch.uint64)
    resolve(module, args)


def upstream_inputs(rows, k, hot_size, *, requests=1, host_blocks=64, width=256):
    """Original CUDA operator geometry, with explicit inputs replacing runtime."""
    args = fixture_inputs([[0] * k] * rows, hot_size=hot_size, union=hot_size)
    blocks_per_request = (hot_size + 63) // 64
    args.update(
        host_cache=((torch.arange((host_blocks + 1) * 64 * width) * 37 + 139) % 256)
        .to(torch.uint8)
        .reshape((host_blocks + 1) * 64, width),
        hot_cache=torch.zeros((requests * blocks_per_request + 1, 64, width), dtype=torch.uint8),
        hot_block_table=torch.arange(requests * blocks_per_request, dtype=torch.int32).reshape(
            requests, blocks_per_request
        ),
        device_global_indices=torch.full((requests, hot_size), -1, dtype=torch.int32),
        lru_slots=torch.arange(hot_size, dtype=torch.int16).repeat(requests, 1),
        request_state_indices=torch.arange(requests, dtype=torch.int32),
        request_ids=torch.zeros((rows,), dtype=torch.int32),
        source_block_table=torch.arange(1, host_blocks + 1, dtype=torch.int32).reshape(requests, -1),
        source_block_size=64,
        attention_block_stride=64,
        num_valid_rows=matrix([rows]),
        stats=torch.zeros((2,), dtype=torch.uint64),
    )
    return args


def resolve_and_gather(module, args, mapping, workspace=None):
    args["host_cache"] = mapping.host
    adapter, actual = resolve(module, args, workspace)
    adapter.host_mapping = mapping
    adapter.hisparse_gather_compact(
        mapping.host,
        actual["hot_cache"],
        actual["swap_host_physical_rows"],
        actual["swap_device_physical_rows"],
        actual["swap_counts"],
    )
    torch.npu.synchronize()
    hot = actual["hot_cache"].cpu().reshape(-1, mapping.host.shape[-1])
    dst, src = args["hot_indices"], args["resolved_global_indices"]
    valid = dst >= 0
    assert torch.equal(hot[dst[valid].to(torch.int64)], mapping.host[src[valid].to(torch.int64)])
    args["hot_cache"].copy_(actual["hot_cache"].cpu())
    return adapter, actual


@pytest.mark.parametrize(
    "rows,stride,k", [(8, 128, 128), (4, 32, 128), (8, 2048, 2048)], ids=["disjoint", "overlap", "long_mtp"]
)
def test_npu_speculative_rows_resolve_host_misses_consistently(module, rows, stride, k):
    blocks = 2 * rows * k // 64
    args = upstream_inputs(rows, k, rows * k, host_blocks=blocks)
    mapping = RegisteredHost(args["host_cache"])
    workspace = None
    try:
        for step in range(4):
            args["global_indices"] = matrix(
                [[(r * stride + c + step * (k // 2)) % (blocks * 64) for c in range(k)] for r in range(rows)]
            )
            adapter, _ = resolve_and_gather(module, args, mapping, workspace)
            workspace = adapter.workspace
    finally:
        mapping.close()


def test_npu_speculative_rows_count_a_shared_load_once(module):
    args = upstream_inputs(2, 128, 256, host_blocks=2)
    args["global_indices"] = torch.arange(128, dtype=torch.int32).repeat(2, 1)
    mapping = RegisteredHost(args["host_cache"])
    try:
        resolve_and_gather(module, args, mapping)
        assert args["stats"].tolist() == [128, 128]
        assert args["swap_counts"].tolist() == [128, 0]
    finally:
        mapping.close()


@pytest.mark.parametrize("k", [128, 8192])
def test_npu_union_table_scales_with_union_bound(module, k):
    args = upstream_inputs(1, k, 32768, host_blocks=k // 64)
    args["max_union_rows"] = k
    args["global_indices"] = torch.arange(k, dtype=torch.int32).reshape(1, k)
    mapping = RegisteredHost(args["host_cache"])
    try:
        resolve_and_gather(module, args, mapping)
    finally:
        mapping.close()


@pytest.mark.parametrize("width", [256, 528, 656])
def test_npu_swap_in_preserves_rows_across_eviction(module, width):
    args = upstream_inputs(4, 128, 256, requests=4, host_blocks=64, width=width)
    args["request_ids"] = torch.arange(4, dtype=torch.int32)
    mapping = RegisteredHost(args["host_cache"])
    workspace = None
    try:
        for step in (0, 0, 1, 2, 0):
            args["global_indices"] = matrix(
                [[(c + step * 128 + r * 17) % 1024 for c in range(127)] + [-1] for r in range(4)]
            )
            adapter, _ = resolve_and_gather(module, args, mapping, workspace)
            workspace = adapter.workspace
        hits, misses = args["stats"].tolist()
        assert hits >= 4 * 127 and misses > 3 * 4 * 127 and hits + misses == 5 * 4 * 127
    finally:
        mapping.close()


def test_npu_resolver_reads_current_request_ids(module):
    args = fixture_inputs([[0, 1, 2, 3]] * 4, requests=[1, 1, 1, 1], states=[2, 0, 1, 3])
    adapter, actual = resolve(module, args)
    actual["request_ids"].copy_(matrix([1, -1, -1, -1]).to("npu"))
    actual["num_valid_rows"] = matrix([1]).to("npu")
    args["request_ids"] = matrix([1, -1, -1, -1])
    args["num_valid_rows"] = matrix([1])
    values = {
        n: t.tolist() for n, t in args.items() if isinstance(t, torch.Tensor) and n not in ("host_cache", "hot_cache")
    }
    expected = resolve_reference(values, 128, 2, 5, 2, union_bound=8)
    adapter.hisparse_resolve_residency(**actual)
    torch.npu.synchronize()
    for name in OUTPUTS:
        if name in args:
            assert actual[name].cpu().tolist() == expected[name], name


def test_npu_multi_step_writes_request_major_output(module):
    args = upstream_inputs(2, 4, 12, requests=2, host_blocks=2, width=32)
    args["request_ids"] = matrix([0, 1])
    physical = torch.full((2, 3, 4), -1, dtype=torch.int32)
    valid_counts = torch.zeros((2, 3), dtype=torch.int32)
    mapping = RegisteredHost(args["host_cache"])
    workspace = None
    try:
        for step in range(3):
            args["global_indices"] = matrix([[step * 4 + c for c in range(4)], [12 + step * 4 + c for c in range(4)]])
            args["attention_indices"] = physical[:, step]
            args["valid_counts"] = valid_counts[:, step]
            adapter, _ = resolve_and_gather(module, args, mapping, workspace)
            workspace = adapter.workspace
        assert valid_counts.tolist() == [[4, 4, 4], [4, 4, 4]]
        hot = args["hot_cache"].reshape(-1, 32)
        expected = mapping.host[matrix([list(range(64, 76)), list(range(140, 152))]).to(torch.int64)]
        assert torch.equal(hot[physical.reshape(2, 12).to(torch.int64)], expected)
    finally:
        mapping.close()


def test_npu_regression_shared_plan_and_rewritten_slot(module):
    args = fixture_inputs([[0, 1, 2, 3], [2, 3, 4, 5]], requests=[0, 0], states=[1, 0, 2, 3])
    mapping = RegisteredHost(args["host_cache"])
    follower = RegisteredHost(args["host_cache"] ^ 213)
    try:
        adapter, actual = resolve_and_gather(module, args, mapping)
        other_hot = torch.zeros_like(actual["hot_cache"])
        other_ops = module.HiSparseTritonCacheOps(adapter.workspace, follower)
        other_ops.hisparse_gather_plan(
            follower.host,
            other_hot,
            actual["resolved_global_indices"],
            actual["hot_indices"],
            actual["miss_mask"],
            matrix([0, 0]).to("npu"),
        )
        torch.npu.synchronize()
        dst = args["hot_indices"].to(torch.int64)
        src = args["resolved_global_indices"].to(torch.int64)
        assert torch.equal(other_hot.cpu().reshape(-1, 16)[dst], follower.host[src])
        mapping.host[4].fill_(231)
        table = actual["device_global_indices"]
        other_table = table.clone()
        ptrs = torch.tensor([table.data_ptr(), other_table.data_ptr()], dtype=torch.int64, device="npu")
        adapter.hisparse_invalidate_written_slots(
            ptrs,
            actual["request_state_indices"],
            matrix([0, 0]).to("npu"),
            torch.tensor([4, 4], dtype=torch.int64, device="npu"),
            4,
            8,
        )
        torch.npu.synchronize()
        assert 4 not in table[1].cpu().tolist() and 4 not in other_table[1].cpu().tolist()
        args["device_global_indices"].copy_(table.cpu())
        _, fresh = resolve_and_gather(module, args, mapping, adapter.workspace)
        target = int(args["hot_indices"][1, 0])
        assert fresh["hot_cache"].cpu().reshape(-1, 16)[target].tolist() == [231] * 16
    finally:
        mapping.close()
        follower.close()


def test_npu_remaps_strided_hma_rows_for_attention(module):
    args = fixture_inputs([[0, 1, 2, 3]])
    storage = torch.full((448,), 79, dtype=torch.uint8)
    args["hot_cache"] = storage.as_strided((2, 4, 16), (192, 16, 1), storage_offset=64)
    args["hot_block_table"] = matrix([[0, 1]])
    args["attention_block_stride"] = 12
    args["lru_slots"][0] = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3], dtype=torch.int16)
    adapter, actual = resolve(module, args)
    assert args["attention_indices"].tolist() == [[12, 13, 14, 15]]
    mapping = RegisteredHost(args["host_cache"])
    try:
        expected, _ = gather_reference(
            storage.tolist(),
            mapping.host.tolist(),
            args["swap_host_physical_rows"].tolist(),
            args["swap_device_physical_rows"].tolist(),
            width=16,
            block=4,
            page_bytes=192,
            view_offset=64,
            hot_rows=8,
            counts=args["swap_counts"].tolist(),
        )
        adapter.host_mapping = mapping
        adapter.hisparse_gather_compact(
            mapping.host,
            actual["hot_cache"],
            actual["swap_host_physical_rows"],
            actual["swap_device_physical_rows"],
            actual["swap_counts"],
        )
        torch.npu.synchronize()
        assert actual["hot_cache"]._base.cpu().tolist() == expected
    finally:
        mapping.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
def test_npu_gather_preserves_payload_bits(module, dtype):
    source = torch.tensor([[0, 0, 192, 127, 255, 255, 255, 255] * 2], dtype=torch.uint8)
    mapping = RegisteredHost(source)
    try:
        hot = torch.zeros((1, 1, 16 // torch.empty((), dtype=dtype).element_size()), dtype=dtype, device="npu")
        workspace = module.HiSparseResolverWorkspace("npu:0", 1, 1, 1, 1, 1)
        module.HiSparseTritonCacheOps(workspace, mapping).hisparse_gather_compact(
            mapping.host, hot, matrix([[0]]).to("npu"), matrix([[0]]).to("npu"), matrix([1]).to("npu")
        )
        assert hot.cpu().view(torch.uint8).reshape(-1).tolist() == source.reshape(-1).tolist()
    finally:
        mapping.close()


def test_npu_gather_zeroes_unaligned_destination(module):
    mapping = RegisteredHost(torch.ones((1, 16), dtype=torch.uint8))
    try:
        storage = torch.full((33,), 7, dtype=torch.uint8, device="npu")
        hot = storage.as_strided((2, 1, 16), (17, 16, 1))
        workspace = module.HiSparseResolverWorkspace("npu:0", 1, 1, 1, 1, 1)
        module.HiSparseTritonCacheOps(workspace, mapping).hisparse_gather_plan(
            mapping.host, hot, matrix([[1]]).to("npu"), matrix([[1]]).to("npu"), matrix([[1]]).to("npu"), None
        )
        assert storage.cpu().tolist() == [7] * 17 + [0] * 16
    finally:
        mapping.close()


def union_overflow_probe():
    module = load_module()
    args = fixture_inputs([[0, 1, 2, -1]], hot_size=4, union=3, optional=0)
    adapter, actual = resolve(module, args)
    print("LEGAL_WARMUP_PASSED", flush=True)
    actual["device_global_indices"].fill_(-1)
    actual["lru_slots"].copy_(torch.arange(4, dtype=torch.int16).repeat(4, 1).to("npu"))
    actual["global_indices"][0, 3] = 3
    torch.npu.synchronize()
    print("OVERFLOW_INPUT_READY", flush=True)
    adapter.hisparse_resolve_residency(**actual)
    torch.npu.synchronize()
    print("UNEXPECTED_OVERFLOW_SUCCESS", flush=True)


def test_npu_expanded_hash_keeps_union_bound(tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.e2e.pull_request.one_card.test_hisparse_ops_npu "
            "import union_overflow_probe; union_overflow_probe()",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    (tmp_path / "child.stdout").write_text(child.stdout)
    (tmp_path / "child.stderr").write_text(child.stderr)
    (tmp_path / "child.exitcode").write_text(str(child.returncode) + "\n")
    assert "LEGAL_WARMUP_PASSED" in child.stdout, child.stdout + child.stderr
    assert "OVERFLOW_INPUT_READY" in child.stdout, child.stdout + child.stderr
    assert child.returncode != 0, child.stdout + child.stderr
    assert "UNEXPECTED_OVERFLOW_SUCCESS" not in child.stdout
    # A3 lowers device_assert to a trap; CANN reports its platform error rather
    # than propagating the source assertion string to the Python exception.
    assert "error code is 507035" in child.stderr, child.stderr
    assert "timeout or trap error" in child.stderr, child.stderr
    assert "union_overflow_probe" in child.stderr, child.stderr


def test_npu_uses_graph_stable_request_state_mapping(module):
    args = upstream_inputs(1, 1, 2, requests=2, host_blocks=2, width=32)
    args.update(
        host_cache=torch.zeros((128, 32), dtype=torch.uint8),
        source_block_table=matrix([[1]]),
        request_state_indices=matrix([1]),
        global_indices=matrix([[0]]),
    )
    resolve(module, args)
    assert args["device_global_indices"][0].tolist() == [-1, -1]
    assert 64 in args["device_global_indices"][1].tolist()


def test_npu_maps_speculative_rows_through_request_state(module):
    args = upstream_inputs(4, 1, 2, host_blocks=1, width=32)
    args.update(
        host_cache=torch.zeros((64, 32), dtype=torch.uint8),
        source_block_table=matrix([[0]]),
        resident_block_table=matrix([[1]]),
        resident_block_size=64,
        global_indices=matrix([[0]] * 4),
    )
    resolve(module, args)
    assert args["hot_indices"].tolist() == [[64]] * 4


def test_npu_resident_rows_bypass_hot_lru(module):
    args = upstream_inputs(1, 4, 5, host_blocks=1, width=32)
    args.update(
        host_cache=torch.zeros((64, 32), dtype=torch.uint8),
        source_block_table=matrix([[5]]),
        resident_block_table=matrix([[1]]),
        resident_block_size=64,
        global_indices=matrix([[1, 2, 3, 4]]),
    )
    before = args["lru_slots"].clone()
    resolve(module, args)
    assert args["hot_indices"].tolist() == [[65, 66, 67, 68]]
    assert args["valid_counts"].tolist() == [4]
    assert torch.equal(before, args["lru_slots"])


def test_npu_gather_rejects_short_attention_block_stride(module):
    w = module.HiSparseResolverWorkspace("npu:0", 1, 1, 1, 1, 1)
    args = [torch.zeros((1, 1), dtype=torch.int32, device="npu") for _ in range(4)]
    with pytest.raises(ValueError, match="shorter than a block"):
        module.HiSparseTritonCacheOps(w).hisparse_gather_plan(
            torch.ones((1, 16), dtype=torch.uint8),
            torch.empty((1, 2, 16), dtype=torch.uint8, device="npu"),
            args[0],
            args[1],
            args[2],
            None,
            args[3],
            1,
        )


def test_npu_multi_step_swaps_match_independent(module):
    args = upstream_inputs(4, 128, 384, requests=4, host_blocks=64)
    args["request_ids"] = torch.arange(4, dtype=torch.int32)
    independent = deepcopy(args)
    mapping = RegisteredHost(args["host_cache"])
    follower_hot = torch.zeros_like(args["hot_cache"], device="npu")
    producer_workspace = independent_workspace = None
    try:
        for iteration in range(2):
            plans = []
            for step in range(2):
                topk = matrix(
                    [[(c + (iteration * 2 + step) * 128 + r * 17) % 1024 for c in range(128)] for r in range(4)]
                )
                args["global_indices"] = topk
                independent["global_indices"] = topk.clone()
                producer, plan = resolve_and_gather(module, args, mapping, producer_workspace)
                own, standalone = resolve_and_gather(module, independent, mapping, independent_workspace)
                producer_workspace, independent_workspace = producer.workspace, own.workspace
                assert torch.equal(plan["hot_indices"].cpu(), standalone["hot_indices"].cpu())
                plans.append(plan)
            for plan in plans:
                producer.hisparse_gather_plan(
                    mapping.host,
                    follower_hot,
                    plan["resolved_global_indices"],
                    plan["hot_indices"],
                    plan["miss_mask"],
                    plan["request_state_indices"],
                )
            torch.npu.synchronize()
            assert torch.equal(follower_hot.cpu(), independent["hot_cache"])
            for plan in plans:
                dst = plan["hot_indices"].cpu().to(torch.int64)
                src = plan["resolved_global_indices"].cpu().to(torch.int64)
                assert torch.equal(follower_hot.cpu().reshape(-1, 256)[dst], mapping.host[src])
    finally:
        mapping.close()


def test_npu_graph_replays_resolver_and_gather_with_current_inputs(module):
    args = fixture_inputs([[0, 1, 2, 3], [2, 3, 4, 5]], requests=[0, 0], states=[1, 0, 2, 3])
    args["num_valid_rows"] = matrix([2])
    mapping = RegisteredHost(args["host_cache"])
    try:
        args["host_cache"] = mapping.host
        adapter, actual = resolve(module, args)
        adapter.host_mapping = mapping

        def run():
            adapter.hisparse_resolve_residency(**actual)
            adapter.hisparse_gather_compact(
                mapping.host,
                actual["hot_cache"],
                actual["swap_host_physical_rows"],
                actual["swap_device_physical_rows"],
                actual["swap_counts"],
            )

        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        for valid in (1, 2):
            args["device_global_indices"].fill_(-1)
            args["lru_slots"].copy_(torch.arange(8, dtype=torch.int16).repeat(4, 1))
            args["global_indices"] = matrix([[4, 5, 6, 7], [6, 7, 8, 9]])
            args["request_ids"] = matrix([0, -1 if valid == 1 else 0])
            args["num_valid_rows"] = matrix([valid])
            args["stats"].zero_()
            for name in OUTPUTS:
                if name in args and name not in ("device_global_indices", "lru_slots", "stats"):
                    args[name].fill_(-77)
            for name, value in args.items():
                if isinstance(value, torch.Tensor) and name not in ("host_cache", "hot_cache"):
                    actual[name].copy_(value.to("npu"))
            mapping.host[8].fill_(246)
            values = {
                n: t.tolist()
                for n, t in args.items()
                if isinstance(t, torch.Tensor) and n not in ("host_cache", "hot_cache")
            }
            expected = resolve_reference(values, 128, 2, 5, 2, union_bound=8)
            graph.replay()
            torch.npu.synchronize()
            for name in OUTPUTS:
                if name in args:
                    assert actual[name].cpu().tolist() == expected[name], name
            dst = torch.tensor(expected["hot_indices"], dtype=torch.int64)[:valid]
            src = torch.tensor(expected["resolved_global_indices"], dtype=torch.int64)[:valid]
            assert torch.equal(actual["hot_cache"].cpu().reshape(-1, 16)[dst], mapping.host[src])
        del graph
    finally:
        mapping.close()


def test_npu_resolver_and_gather_current_stream_event_order(module):
    args = fixture_inputs([[0, 1, 2, 3]])
    mapping = RegisteredHost(args["host_cache"])
    try:
        adapter, actual = resolve(module, args)
        adapter.host_mapping = mapping
        # Warm compilation, then prepare a fresh cold-cache input before the
        # producer stream waits for the default stream's input writes.
        adapter.hisparse_gather_compact(
            mapping.host,
            actual["hot_cache"],
            actual["swap_host_physical_rows"],
            actual["swap_device_physical_rows"],
            actual["swap_counts"],
        )
        torch.npu.synchronize()
        actual["device_global_indices"].fill_(-1)
        actual["lru_slots"].copy_(torch.arange(8, dtype=torch.int16).repeat(4, 1).to("npu"))
        producer = torch.npu.Stream()
        producer.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(producer):
            actual["hot_cache"].fill_(17)
            adapter.hisparse_resolve_residency(**actual)
            adapter.hisparse_gather_compact(
                mapping.host,
                actual["hot_cache"],
                actual["swap_host_physical_rows"],
                actual["swap_device_physical_rows"],
                actual["swap_counts"],
            )
            ready = torch.npu.Event()
            ready.record()
        torch.npu.current_stream().wait_event(ready)
        dst = actual["hot_indices"].cpu().to(torch.int64)
        assert torch.equal(actual["hot_cache"].cpu().reshape(-1, 16)[dst], mapping.host[matrix([[0, 1, 2, 3]])])
    finally:
        mapping.close()


@pytest.mark.parametrize("width,dtype", [(16, torch.float32), (528, torch.uint8), (656, torch.uint8)])
def test_npu_gather_prefill_cache_prefers_resident_rows(module, width, dtype):
    # Operator part of upstream's block tables [[5,2,0],[9,3,0]], lengths
    # [5,8], source block=4, resident block=2, staging block capacity=4.
    generator = torch.Generator().manual_seed(20261102)
    host = torch.randint(0, 128, (40, width), generator=generator, dtype=dtype)
    resident = torch.randint(128, 256, (48, width), generator=generator, dtype=dtype)
    host_rows = [20, 21, 22, 23, 8, -1, -1, -1, 36, 37, 38, 39, 12, 13, 14, 15]
    resident_rows = [22, 23, 24, 25, -1, -1, -1, -1, 42, 43, -1, -1, 44, 45, 46, 47]
    mask = [int(g >= 0 and r < 0) for g, r in zip(host_rows, resident_rows)]
    expected = torch.zeros((16, width), dtype=dtype)
    initial = torch.zeros_like(expected)
    for index, (g, r) in enumerate(zip(host_rows, resident_rows)):
        if r >= 0:
            initial[index].copy_(resident[r])
            expected[index].copy_(resident[r])
        elif g >= 0:
            expected[index].copy_(host[g])
    mapping = RegisteredHost(host)
    try:
        staging = initial.to("npu")
        workspace = module.HiSparseResolverWorkspace("npu:0", 1, 1, 16, 16, 16)
        module.HiSparseTritonCacheOps(workspace, mapping).hisparse_gather_plan(
            mapping.host,
            staging,
            matrix([host_rows]).to("npu"),
            torch.arange(16, dtype=torch.int32).reshape(1, 16).to("npu"),
            matrix([mask]).to("npu"),
            None,
        )
        assert torch.equal(staging.cpu().view(torch.uint8), expected.view(torch.uint8))
    finally:
        mapping.close()


@pytest.mark.parametrize("width", [16, 64, 256, 528, 656])
@pytest.mark.parametrize("layout", ["staging", "paged", "compact"])
def test_npu_gather_bytes_and_guards(module, width, layout):
    block = 1 if layout == "staging" else 2
    pages, offset, gap = 8, 1, 17
    page_bytes = block * width + gap
    storage = torch.full((offset + pages * page_bytes + 19,), 173, dtype=torch.uint8, device="npu")
    shape = (pages, width) if block == 1 else (pages, block, width)
    strides = (page_bytes, 1) if block == 1 else (page_bytes, width, 1)
    hot = storage.as_strided(shape, strides, storage_offset=offset)
    source_storage = ((torch.arange(9 * width, dtype=torch.int32) * 37 + 139) % 256).to(torch.uint8).reshape(9, width)
    mapping = RegisteredHost(source_storage)
    try:
        host = mapping.host[2:7]
        source = matrix([[0, 1, -1, 9], [2, 3, 4, 1], [1, 2, 3, 4], [9, 1, 2, 3]])
        target = matrix([[0, 1, 2, 3], [4, 5, 6, pages * block], [1, 2, 3, 4], [-1, 5, 6, 7]])
        mask = matrix([[1, 0, 1, 2], [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 0, 0]])
        counts = matrix([99, 2, -7, 0]) if layout == "compact" else None
        states = None if counts is not None else matrix([2, 0, -1, 1])
        expected, att = gather_reference(
            storage.cpu().tolist(),
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
        w = module.HiSparseResolverWorkspace("npu:0", 4, 4, 4, 8, 8)
        adapter = module.HiSparseTritonCacheOps(w, mapping)
        attention = torch.full(source.shape, -77, dtype=torch.int32, device="npu")
        if counts is not None:
            adapter.hisparse_gather_compact(host, hot, source.to("npu"), target.to("npu"), counts.to("npu"))
        else:
            adapter.hisparse_gather_plan(
                host, hot, source.to("npu"), target.to("npu"), mask.to("npu"), states.to("npu"), attention, block + 3
            )
        torch.npu.synchronize()
        assert storage.cpu().tolist() == expected
        if counts is None:
            assert attention.cpu().tolist() == att
    finally:
        mapping.close()
