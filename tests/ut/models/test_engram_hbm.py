# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run standalone: pytest --confcutdir=tests/ut/models test_engram_hbm.py."""

import importlib.util
import json
import sys
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"vllm_ascend/models/deepseek_v41/{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hbm = load_module("engram_hbm")
gate_mod = load_module("engram_gate")
hash_mod = load_module("engram_hash")
gate = gate_mod.engram_gate


@contextmanager
def _cpu_offload_runtime():
    """Mock native lookup and pinned allocation; exercise routing and slot lifetime."""
    empty = torch.empty
    pinned_allocations: list[int] = []

    def allocate(*args, **kwargs):
        pin_memory = kwargs.pop("pin_memory", False)
        tensor = empty(*args, **kwargs)
        if pin_memory:
            pinned_allocations.append(tensor.data_ptr())
        return tensor

    def lookup(weight, scale, ids, output):
        output.copy_(hbm.dequantize_engram_rows(weight.index_select(0, ids), scale.index_select(0, ids)))

    with (
        patch.dict(sys.modules, {"vllm_ascend.vllm_ascend_C": ModuleType("vllm_ascend_C")}),
        patch.object(torch.ops._C_ascend, "engram_int8_lookup_cpu", lookup, create=True),
        patch.object(torch, "empty", allocate),
    ):
        yield pinned_allocations


@pytest.fixture(autouse=True)
def cpu_offload_runtime():
    with _cpu_offload_runtime() as allocations:
        yield allocations


@pytest.mark.parametrize("cp", ["none", "v41", "v41_empty_rank"])
def test_engram_history_metadata_uses_full_requests(cp):
    boundaries = torch.tensor([0, 3, 9], dtype=torch.int32)
    pages = torch.tensor([[7, 8, 9], [12, 13, 14]], dtype=torch.int32)
    # Device tensors are intentionally unusable: history must use host mirrors.
    fields = dict(
        query_start_loc=None,
        block_table=None,
        storage_block_size=4,
        query_start_loc_cpu=boundaries,
        block_table_cpu=pages,
    )
    request = SimpleNamespace(**fields)
    if cp.startswith("v41"):
        local = torch.tensor([0, 0, 0] if cp == "v41_empty_rank" else [0, 0, 2])
        metadata = SimpleNamespace(
            global_metadata=request,
            query_start_loc=local,
            query_start_loc_cpu=local,
            block_table=None,
            storage_block_size=4,
        )
    else:
        metadata = request
    actual_boundaries, actual_pages, block_size = hash_mod.engram_history_metadata(metadata)
    assert torch.equal(actual_boundaries, boundaries.long())
    assert torch.equal(actual_pages, pages)
    assert block_size == 4


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    tp = None
    for ranks in ([0, 1], [2, 3]):
        group = dist.new_group(ranks)
        if rank in ranks:
            tp = group
    q = hbm.EngramQueryGroup(dist.group.WORLD, dist.group.WORLD, tp, (rank // 2) * 2)
    # Deliberately not divisible by group size; includes signed zero and NaN bits.
    weights = torch.arange(29 * 8).reshape(29, 8).to(torch.bfloat16)
    weights.view(torch.int16)[0, 0] = -32768
    weights.view(torch.int16)[28, 7] = 32705
    table = hbm.NodeShardedEngram(29, 8, q, device="cpu")
    table.weight.data.copy_(weights[table.start : table.end])
    for a, b in [(0, 0), (1, 0), (0, 17), (3, 7), (31, 1), (1, 1), (0, 0)]:
        n = (a, b)[rank // 2]
        ids = (torch.arange(n * 3).reshape(n, 3) * 7 + rank // 2) % 29
        actual = table(ids)
        expected = weights[ids]
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (rank, a, b)
    assert torch.equal(
        table(torch.tensor([[1, 8, 28]])).view(torch.int16), weights[torch.tensor([[1, 8, 28]])].view(torch.int16)
    )
    ids = torch.tensor([[1, 8, 28]])
    packed = table.forward_many([ids, ids])
    assert torch.equal(packed[0].view(torch.int16), weights[ids].view(torch.int16))
    assert torch.equal(packed[1].view(torch.int16), weights[ids].view(torch.int16))
    single = table.route_many([table], [ids])[0]
    assert torch.equal(single.view(torch.int16), weights[ids].view(torch.int16))
    # Distinct row counts exercise different shard boundaries in one exchange.
    other_weights = -torch.arange(37 * 8).reshape(37, 8).bfloat16()
    other = hbm.NodeShardedEngram(37, 8, q, device="cpu")
    other.weight.data.copy_(other_weights[other.start : other.end])
    for a, b in [(0, 0), (1, 0), (0, 17), (31, 1), (1, 31), (3, 7)]:
        n = (a, b)[rank // 2]
        first_ids = (torch.arange(n * 3).reshape(n, 3) * 7 + rank // 2) % 29
        second_ids = (torch.arange(n * 3).reshape(n, 3) * 11 + rank // 2) % 37
        first, second = table.route_many([table, other], [first_ids, second_ids])
        assert torch.equal(first.view(torch.int16), weights[first_ids].view(torch.int16))
        assert torch.equal(second.view(torch.int16), other_weights[second_ids].view(torch.int16))
    recovered = table.route_many([table, other], [ids, ids])
    assert torch.equal(recovered[1].view(torch.int16), other_weights[ids].view(torch.int16))
    dist.destroy_process_group()


def _compressed_wire_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    for ranks in ([0, 1], [2, 3]):
        group = dist.new_group(ranks)
        if rank in ranks:
            tp = group
    q = hbm.EngramQueryGroup(dist.group.WORLD, dist.group.WORLD, tp, rank // 2 * 2)
    table = hbm.NodeShardedEngram(37, 256, q, device="cpu", storage_format="int8")
    codes = (torch.arange(37 * 256).reshape(37, 256) % 256 - 128).to(torch.int8)
    scales = torch.linspace(0.001, 0.9, 37 * 8).reshape(37, 8)
    codes[0] = 0
    table.weight.data.copy_(codes[table.start : table.end])
    table.weight_scale.copy_(scales[table.start : table.end])
    reference = (codes.float().reshape(37, 8, 32) * scales[..., None]).reshape(37, 256).bfloat16()
    table.compressed_int8_wire = True
    for a, b in ((0, 0), (1, 0), (0, 17), (3, 7), (128, 1), (1, 128)):
        n = (a, b)[rank // 2]
        ids = (torch.arange(n * 24).reshape(n, 24) * 7 + rank // 2) % 37
        if n:
            ids[0, :3] = torch.tensor([0, 36, 0])
        actual = table(ids)
        assert torch.equal(actual.view(torch.int16), reference[ids].view(torch.int16)), (rank, a, b)
    ids = torch.tensor([[0, 36, 1]])
    assert torch.equal(table(ids).view(torch.int16), reference[ids].view(torch.int16))
    dist.destroy_process_group()


def _mixed_storage_worker(rank, rendezvous, cpu_offload):
    with _cpu_offload_runtime():
        torch.set_num_threads(1)
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
        for ranks in ([0, 1], [2, 3]):
            group = dist.new_group(ranks)
            if rank in ranks:
                tp = group
        q = hbm.EngramQueryGroup(dist.group.WORLD, dist.group.WORLD, tp, rank // 2 * 2)
        bf16 = hbm.NodeShardedEngram(41, 256, q, device="cpu", storage_format="bf16", cpu_offload=cpu_offload)
        int8 = hbm.NodeShardedEngram(43, 256, q, device="cpu", storage_format="int8", cpu_offload=cpu_offload)
        bf16_reference = torch.arange(41 * 256, dtype=torch.float32).reshape(41, 256).bfloat16()
        int8_codes = (torch.arange(43 * 256).reshape(43, 256) % 255 - 127).to(torch.int8)
        int8_scales = torch.linspace(0.25, 1.0, 43 * 8).reshape(43, 8)
        int8_reference = (int8_codes.float().reshape(43, 8, 32) * int8_scales[..., None]).reshape(43, 256).bfloat16()
        bf16.weight.data.copy_(bf16_reference[bf16.start : bf16.end])
        int8.weight.data.copy_(int8_codes[int8.start : int8.end])
        int8.weight_scale.copy_(int8_scales[int8.start : int8.end])
        for counts in ((0, 0), (0, 5), (7, 0), (3, 17), (17, 3)):
            count = counts[rank // 2]
            first_ids = (torch.arange(count * 7).reshape(count, 7) * 3 + rank // 2) % 41
            second_ids = (torch.arange(count * 7).reshape(count, 7) * 5 + rank // 2) % 43
            first, second = bf16.route_many([bf16, int8], [first_ids, second_ids])
            assert torch.equal(first.view(torch.int16), bf16_reference[first_ids].view(torch.int16))
            assert torch.equal(second.view(torch.int16), int8_reference[second_ids].view(torch.int16))
        dist.destroy_process_group()


def _fp8_offload_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    for ranks in ([0, 1], [2, 3]):
        group = dist.new_group(ranks)
        if rank in ranks:
            tp = group
    q = hbm.EngramQueryGroup(dist.group.WORLD, dist.group.WORLD, tp, rank // 2 * 2)
    for storage in ("fp8", "mxfp8"):
        table = hbm.NodeShardedEngram(37, 32, q, device="cpu", storage_format=storage)
        codes = (torch.arange(37 * 32).reshape(37, 32) % 240 - 120).to(torch.float8_e4m3fn)
        scales = torch.tensor([[1.0]] * 37, dtype=torch.float8_e8m0fnu)
        table.weight.data.copy_(codes[table.start : table.end])
        table.weight_scale.copy_(scales[table.start : table.end])
        ids = torch.tensor([[0, 36, 1], [7, 7, 19]]) if rank // 2 == 0 else torch.empty((0, 3), dtype=torch.long)
        actual = table(ids)
        reference = codes.float().reshape(37, 1, 32).mul(scales.float().reshape(37, 1, 1)).reshape(37, 32).bfloat16()
        expected = reference[ids] if rank // 2 == 0 else torch.empty((0, 3, 32), dtype=torch.bfloat16)
        assert actual.is_pinned() is False
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (rank, storage)
    dist.destroy_process_group()


def test_cross_dp_variable_idle_and_exact_bf16(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)


def test_int8_compressed_wire_cross_dp(tmp_path):
    mp.spawn(_compressed_wire_worker, args=(f"file://{tmp_path / 'compressed'}",), nprocs=4, join=True)


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_route_many_mixed_storage_and_idle(tmp_path, cpu_offload):
    mp.spawn(_mixed_storage_worker, args=(f"file://{tmp_path / 'mixed'}", cpu_offload), nprocs=4, join=True)


def test_fp8_offload_cross_dp_and_idle(tmp_path):
    mp.spawn(_fp8_offload_worker, args=(f"file://{tmp_path / 'fp8'}",), nprocs=4, join=True)


def test_shard_loader(tmp_path):
    weights = torch.arange(29 * 8).reshape(29, 8).to(torch.bfloat16)
    key = "layers.1.engram.embed.weight"
    save_file({key: weights}, tmp_path / "weights.safetensors")
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors"}})
    )
    pieces = []
    for rank in range(16):
        q = type("QueryGroup", (), {"size": 16, "rank": rank})()
        if rank * 2 >= 29:
            continue
        table = hbm.NodeShardedEngram(29, 8, q, device="cpu")
        table.load_checkpoint(tmp_path, key, chunk_rows=3)
        pieces.append(table.weight)
    assert torch.equal(torch.cat(pieces), weights)


@pytest.mark.parametrize("storage", ["bf16", "int8", "fp8", "mxfp8"])
def test_offload_loader_and_pinned_reuse(tmp_path, storage, cpu_offload_runtime):
    from unittest.mock import Mock

    key, scale_key = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    codes = (torch.arange(19 * 32).reshape(19, 32) % 31 - 15).to(
        torch.bfloat16 if storage == "bf16" else torch.int8 if storage == "int8" else torch.float8_e4m3fn
    )
    if storage == "bf16":
        codes.view(torch.int16)[15, :4] = torch.tensor([0, -32768, 32705, 1], dtype=torch.int16)
    scales = (2.0 ** (torch.arange(19)[:, None] % 5 - 2)).to(
        torch.float32 if storage == "int8" else torch.float8_e8m0fnu
    )
    tensors = {key: codes}
    if storage != "bf16":
        tensors[scale_key] = scales
    save_file(tensors, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"other": "missing"}}))
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors", scale_key: "weights.safetensors"}})
    )
    query = SimpleNamespace(size=4, rank=3)
    with torch.device("meta"):
        table = hbm.NodeShardedEngram(19, 32, query, device="meta", storage_format=storage, cpu_offload=True)
    assert table.weight.device.type == "cpu"
    if storage != "bf16":
        assert table.weight_scale.device.type == "cpu"
    assert not table.weight.is_pinned()
    table.load_checkpoint(tmp_path, key, chunk_rows=2)
    assert torch.equal(table.weight.view(torch.uint8), codes[15:19].view(torch.uint8))
    ids = torch.tensor([0, 3, 0])
    reference = codes[ids + 15] if storage == "bf16" else (codes.float() * scales.float()).bfloat16()[ids + 15]
    first = table.lookup_local(ids, pin_output=True)
    assert first.data_ptr() in cpu_offload_runtime
    torch.testing.assert_close(first.view(torch.int16), reference.view(torch.int16), rtol=0, atol=0)
    event = Mock()
    table._offload_events[first.data_ptr()] = event
    second = table.lookup_local(ids, pin_output=True)
    event.synchronize.assert_not_called()
    third = table.lookup_local(ids, pin_output=True)
    event.synchronize.assert_called_once_with()
    assert first.data_ptr() == third.data_ptr() != second.data_ptr()
    # Force eviction and verify a still-used pinned source is fenced first.
    event = Mock()
    table._offload_events[third.data_ptr()] = event
    table._offload_buffer_bytes_limit = 1
    table.lookup_local(ids[:1], pin_output=True)
    event.synchronize.assert_called_once_with()
    assert len(table._offload_buffers) == 1
    assert table.lookup_local(torch.empty(0, 2, dtype=torch.int64)).shape == (0, 2, 32)


@pytest.mark.parametrize("storage,swap_indexes", [("bf16", True), ("int8", True), ("fp8", True), ("mxfp8", False)])
def test_loader_selects_compatible_index(tmp_path, storage, swap_indexes):
    key, scale_key = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    bf16 = torch.ones(5, 32, dtype=torch.bfloat16)
    fp8 = {key: bf16.to(torch.float8_e4m3fn), scale_key: torch.ones(5, 1).to(torch.float8_e8m0fnu)}
    quant = (
        {key: torch.full((5, 32), 7, dtype=torch.int8), scale_key: torch.ones(5, 1)}
        if storage == "int8"
        else {key: bf16}
    )
    sources = (quant, fp8) if swap_indexes else (fp8, quant)
    for name, tensors in zip(("model", "quant_model_weights"), sources):
        save_file(tensors, tmp_path / f"{name}.safetensors")
        (tmp_path / f"{name}.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: f"{name}.safetensors" for key in tensors}})
        )
    table = hbm.NodeShardedEngram(5, 32, SimpleNamespace(size=1, rank=0), storage_format=storage, cpu_offload=True)
    table.load_checkpoint(tmp_path, key)
    expected = fp8 if storage in ("fp8", "mxfp8") else quant
    assert torch.equal(table.weight.view(torch.uint8), expected[key].view(torch.uint8))
    if scale_key in expected:
        assert torch.equal(table.weight_scale.view(torch.uint8), expected[scale_key].view(torch.uint8))


def test_int8_loads_bf16_source_without_scale(tmp_path):
    key = "layers.1.engram.embed.weight"
    weights = torch.linspace(-12, 12, 19 * 32).reshape(19, 32).bfloat16()
    save_file({key: weights}, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "weights.safetensors"}}))
    table = hbm.NodeShardedEngram(19, 32, SimpleNamespace(size=4, rank=3), storage_format="int8", cpu_offload=True)
    table.load_checkpoint(tmp_path, key, chunk_rows=2)
    codes, scales = hbm.quantize_engram_rows(weights[15:19])
    assert torch.equal(table.weight, codes)
    assert torch.equal(table.weight_scale, scales)


def test_offload_records_h2d_use_before_response_failure(monkeypatch):
    query = SimpleNamespace(size=1, rank=0, is_source=True, group=None)
    table = hbm.NodeShardedEngram(5, 32, query, storage_format="int8", cpu_offload=True)
    table.weight.data.fill_(1)
    table.weight_scale.fill_(1)
    events: list[str] = []

    def exchange(output, source, *args, **kwargs):
        if source.dtype == torch.int64:
            output.copy_(source)
        else:
            assert events == ["source_fenced"]
            raise RuntimeError("response failed")

    monkeypatch.setattr(dist, "get_backend", lambda *args: "gloo")
    monkeypatch.setattr(dist, "all_to_all_single", exchange)
    monkeypatch.setattr(table, "_record_offload_use", lambda *args: events.append("source_fenced"))
    with pytest.raises(RuntimeError, match="response failed"):
        table._forward_with_gathered(torch.tensor([[1]]), [torch.tensor([1, 0])])


def test_cached_metadata_stays_on_cpu_with_device_context():
    q = type("QueryGroup", (), {"size": 4, "rank": 1, "is_source": False})()
    with torch.device("meta"):
        table = hbm.NodeShardedEngram(17, 32, q, device="cpu")
    for ids in (torch.empty((0, 3), dtype=torch.int64), torch.tensor([[1, 2, 3]])):
        flat, order, metadata = table._metadata(ids)
        assert flat.device.type == order.device.type == metadata.device.type == "cpu"
        assert torch.equal(metadata, torch.zeros(5, dtype=torch.int64))


def test_gate_preserves_masked_rows():
    torch.manual_seed(7)
    hidden = torch.randn(3, 4, 32).bfloat16()
    key = torch.randn(3, 4, 32).bfloat16()
    value = torch.randn(3, 32).bfloat16()
    out = gate(hidden, key, value, torch.randn(4, 32), torch.eye(32), torch.tensor([True, False, True]), 1e-5)
    assert torch.equal(out[1], hidden[1])
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("barrier_token", [98, 99])
def test_hash_causal_barrier(barrier_token):
    h = hash_mod.PagedNgramHistory.__new__(hash_mod.PagedNgramHistory)
    h.token_map = torch.arange(100)
    h.pad_id = 2
    h.image_token_id = 99
    h.lookback = 2
    h.image_pad_token_id = 98
    h.primes = torch.tensor([[[101, 103]]])
    h.offsets = torch.tensor([[0, 101]])
    h.multipliers = torch.tensor([[3, 5]])
    h.pages = {}
    values, mask = h.update(
        torch.tensor([0, 5, 9, barrier_token, 13, 17]),
        torch.arange(6),
        torch.zeros(6, dtype=torch.long),
        torch.tensor([[5, 1]]),
        4,
    )
    assert values.shape == (6, 1, 2) and not mask[3]
    # The first token on the next page must hash against padding, not the image.
    assert values[4, 0, 0].item() == ((13 * 3) ^ (h.pad_id * 5)) % 101
