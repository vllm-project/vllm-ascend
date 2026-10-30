# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Focused tests for the Ascend Engram configuration and storage path."""

import ctypes
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip(
    "vllm.models.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

from vllm_ascend.utils import vllm_version_is

# Upstream #59327 (4c2d277643) moved the Engram module from `nvidia.engram` to
# `common.engram` on main; the release lane (v0.31.0) still ships it under
# `nvidia.engram`.
if vllm_version_is("0.31.0"):
    from vllm.models.deepseek_v41.nvidia import engram as upstream_engram  # type: ignore[import-not-found]
else:
    from vllm.models.deepseek_v41.common import engram as upstream_engram

from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod
from vllm_ascend.models.deepseek_v41.engram import npu
from vllm_ascend.models.deepseek_v41.engram.common import engram_gate
from vllm_ascend.models.deepseek_v41.engram.hash_state import DEAD_ID, AscendNgramHashState
from vllm_ascend.models.deepseek_v41.engram.parallel import resolve_dp_shared_memory


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("tokens", [0, 3])
@pytest.mark.parametrize("quantized", [False, True])
def test_lookup_overwrites_dirty_output_before_exchange(monkeypatch, tp, dp, tokens, quantized):
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table.dim, table.block_size = 64, 32
    table.n_hash_cols, table.part_n_hash_cols = 24, 24 // tp
    table.head_start = 24 - table.part_n_hash_cols
    table.vocab_start_idx, table.vocab_end_idx = 5, 8
    table.tp_size, table.dp_size = tp, dp
    table._codes_uva = None
    codes = torch.arange(3 * 64).reshape(3, 64).remainder(31).sub(15)
    table.weight = torch.nn.Parameter(codes.to(torch.int8 if quantized else torch.bfloat16), requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(torch.full((3, 2), 0.125), requires_grad=False) if quantized else None
    # Match the non-contiguous layer slice passed by prepare_engram.
    ids = torch.tensor([-1, 5, 6, 8]).repeat(tokens * dp * 12).reshape(tokens * dp, 2, 24)[:, 1]
    columns = ids[:, table.head_start :]
    owned = (columns >= 5) & (columns < 8)
    local = torch.where(owned, columns - 5, 0)
    expected = (codes[local].float() * (0.125 if quantized else 1)).bfloat16()
    expected = torch.where(owned.unsqueeze(-1), expected, 0)
    original_empty = torch.empty

    def dirty_empty(*args, **kwargs):
        return original_empty(*args, **kwargs).fill_(float("nan"))

    def exchange(out, count):
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        return out[-count:] if count else out[:0]

    def gather(out, dim):
        torch.testing.assert_close(out, expected[:tokens], rtol=0, atol=0)
        return torch.cat([out] * tp, dim=dim)

    monkeypatch.setattr(embedding_mod.torch, "empty", dirty_empty)
    monkeypatch.setattr(embedding_mod, "exchange_engram_rows", exchange)
    monkeypatch.setattr(embedding_mod, "tensor_model_parallel_all_gather", gather)
    result = table.embed_gathered(ids, tokens)
    torch.testing.assert_close(result, torch.cat([expected[:tokens]] * tp, dim=1), rtol=0, atol=0)


@pytest.mark.parametrize("failure", [None, "malloc", "null", "register", "device_pointer"])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("offset", [0, 4096, 65535])
def test_host_uva_aligns_registration_and_frees_original_allocation(monkeypatch, failure, dtype, offset):
    alignment, size = 65536, 4096

    class Library:
        def __init__(self):
            self.storage = ctypes.create_string_buffer(size + 4 * alignment)
            base = ctypes.addressof(self.storage)
            self.original = (base + alignment - 1) // alignment * alignment + offset
            self.registered = self.unregistered = self.freed = None
            self.free_count = self.unregister_count = 0

        def aclrtMallocHost(self, output, count, flags):
            assert count == size + alignment - 1
            if failure == "malloc":
                return 1
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = (
                None if failure == "null" else self.original
            )
            return 0

        def aclrtHostRegisterV2(self, pointer, count, flags):
            assert count == size and pointer.value % alignment == 0
            self.registered = pointer.value
            return 1 if failure == "register" else 0

        def aclrtHostGetDevicePointer(self, pointer, output, flags):
            assert pointer.value == self.registered
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0x100000000
            return 1 if failure == "device_pointer" else 0

        def aclrtHostUnregister(self, pointer):
            self.unregistered = pointer.value
            self.unregister_count += 1
            return 0

        def aclrtFreeHost(self, pointer):
            self.freed = pointer.value
            self.free_count += 1
            return 0

    lib = Library()
    monkeypatch.setattr(npu, "_host_library", lambda: lib)
    shape = (size // torch.empty((), dtype=dtype).element_size(),)
    if failure:
        message = "NULL pointer" if failure == "null" else "failed: rc=1"
        with pytest.raises(RuntimeError, match=message):
            npu.HostUvaBuffer(shape, dtype, "cpu")
    else:
        buffer = npu.HostUvaBuffer(shape, dtype, "cpu")
        assert buffer.tensor.data_ptr() == lib.registered
        assert lib.original <= lib.registered and lib.registered + size <= lib.original + size + alignment - 1
        buffer.tensor.fill_(17)
        value = torch.frombuffer(
            lib.storage, dtype=dtype, count=1, offset=lib.registered - ctypes.addressof(lib.storage)
        )
        assert value.item() == 17
        buffer.close()
        buffer.close()
    if failure in ("malloc", "null"):
        assert lib.registered is lib.unregistered is lib.freed is None
        assert lib.free_count == lib.unregister_count == 0
    else:
        assert lib.unregistered == lib.registered
        assert lib.freed == lib.original
        assert lib.free_count == lib.unregister_count == 1


def test_shared_memory_needs_a_local_dp_peer(monkeypatch):
    from vllm_ascend.models.deepseek_v41.engram import parallel

    monkeypatch.setattr(parallel, "get_engram_dp_size", lambda: 1)
    assert not resolve_dp_shared_memory(True)
    monkeypatch.setattr(parallel, "get_engram_dp_size", lambda: 2)
    assert resolve_dp_shared_memory(True)
    assert not resolve_dp_shared_memory(False)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_loader_preserves_checkpoint_shards_across_dp_pcp_tp(tmp_path, monkeypatch, quantized, shared):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    head_sizes = (1, 2, 3, 4, 5, 6, 7, 8)
    source = torch.linspace(-12, 12, 36 * 64).reshape(36, 64).bfloat16()
    codes, scales = npu.quantize_engram_rows(source)
    save_file({key: source}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "model.safetensors"}}))
    if quantized:
        save_file({key: codes, scale_key: scales}, tmp_path / "quant.safetensors")
        (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "quant.safetensors", scale_key: "quant.safetensors"}})
        )
    expected = codes if quantized else source
    storage: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor | None]] = {}
    writers, barriers = [], []

    def allocate(table):
        rows = table.part_num_embeddings
        return storage.setdefault(
            (table.vocab_start_idx, table.vocab_end_idx),
            (
                torch.zeros(rows, 64, dtype=expected.dtype),
                torch.zeros(rows, 2, dtype=scales.dtype) if quantized else None,
            ),
        )

    original_load = embedding_mod.AscendParallelEngramEmbedding._load_into_storage

    def load(table, *args):
        writers.append((table.vocab_start_idx, table.vocab_end_idx))
        original_load(table, *args)

    def synchronize(errors, error, *, group):
        assert error is None
        barriers.append(group)
        errors[:] = [None] * len(errors)

    monkeypatch.setattr(embedding_mod.AscendParallelEngramEmbedding, "_allocate_weights", allocate)
    monkeypatch.setattr(embedding_mod.AscendParallelEngramEmbedding, "_load_into_storage", load)
    monkeypatch.setattr(embedding_mod.dist, "all_gather_object", synchronize)
    monkeypatch.setattr(embedding_mod, "in_the_same_node_as", lambda group: [True] * 4)
    monkeypatch.setattr(embedding_mod, "get_tp_group", lambda: SimpleNamespace(cpu_group=object()))
    monkeypatch.setattr(embedding_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(embedding_mod, "get_engram_dp_size", lambda: 4)
    shared_ranges = [(0, 10), (10, 36)]
    split_ranges = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 15), (15, 21), (21, 28), (28, 36)]
    loaded_ranges = []
    for dp_rank in range(2):
        for pcp_rank in range(2):
            for tp_rank in range(2):
                group = SimpleNamespace(world_size=4, rank_in_group=dp_rank * 2 + pcp_rank, cpu_group=object())
                monkeypatch.setattr(embedding_mod, "get_engram_dp_group", lambda group=group: group)
                monkeypatch.setattr(embedding_mod, "get_tensor_model_parallel_rank", lambda rank=tp_rank: rank)
                monkeypatch.setattr(upstream_engram, "get_engram_dp_group", lambda group=group: group)
                monkeypatch.setattr(upstream_engram, "get_tensor_model_parallel_rank", lambda rank=tp_rank: rank)
                table = embedding_mod.AscendParallelEngramEmbedding(
                    36, 64, head_sizes, 0, cpu_offload=True, dp_shared_memory=shared, storage_dtype=expected.dtype
                )
                start, end = shared_ranges[tp_rank] if shared else split_ranges[tp_rank * 4 + dp_rank * 2 + pcp_rank]
                assert (table.vocab_start_idx, table.vocab_end_idx) == (start, end)
                assert table.dp_size == (1 if shared else 4)
                assert table._shared_group is (group if shared else None)
                table.load_checkpoint(tmp_path, key, chunk_rows=7)
                assert torch.equal(table.weight, expected[start:end])
                if quantized:
                    assert torch.equal(table.weight_scale_inv, scales[start:end])
                loaded_ranges.append((start, end))
    assert sorted(set(loaded_ranges)) == (shared_ranges if shared else split_ranges)
    assert sorted(writers) == (shared_ranges if shared else split_ranges)
    assert len(barriers) == (8 if shared else 0)


from vllm_ascend.worker.v1.model_runner import NPUModelRunner


@pytest.mark.parametrize(
    "enabled,shared,tp,mode,expected",
    [
        (True, True, 1, "FULL", True),
        (True, True, 1, "NONE", True),
        (False, True, 1, "FULL", False),
        (True, False, 1, "FULL", True),
        (True, True, 2, "FULL", True),
        (True, False, 2, "NONE", True),
        (True, True, 1, "PIECEWISE", False),
    ],
)
def test_preparation_overlap_supports_dp_tp_and_checks_runtime(monkeypatch, enabled, shared, tp, mode, expected):
    from vllm.config import CUDAGraphMode

    from vllm_ascend.models.deepseek_v41 import model as model_module

    model = SimpleNamespace(has_engram=True, _engram_overlap_enabled=enabled, engram_dp_shared_memory=shared)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(model_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        model_module, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode[mode])
    )
    assert model_module.DeepseekV41Model._can_overlap_engram_preparation(model) is expected


def test_bf16_gate_without_rotation():
    hidden = torch.ones(2, 4, 64, dtype=torch.bfloat16)
    value = torch.full((2, 64), 0.25, dtype=torch.bfloat16)
    mask = torch.tensor([True, False])
    result = engram_gate(hidden, hidden * 2, value, torch.ones(4, 64), None, mask, 1e-20)
    # Normalized dot is sqrt(64) = 8; the gate applies signed sqrt, then sigmoid.
    expected = (1 + 0.25 * torch.sigmoid(torch.tensor(8.0).sqrt())).bfloat16()
    assert torch.all(result[0] == expected)
    assert torch.equal(result[1], hidden[1])


def test_loader_applies_mxfp8_checkpoint_scales(tmp_path):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    source = torch.linspace(-0.5, 0.5, 19 * 64).reshape(19, 64)
    checkpoint_scale = torch.full((19, 2), 1 / 128, dtype=torch.float32).to(torch.float8_e8m0fnu)
    checkpoint_weight = (source * 128).to(torch.float8_e4m3fn)
    save_file({key: checkpoint_weight, scale_key: checkpoint_scale}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors", scale_key: "model.safetensors"}})
    )
    embedding_mod.preflight_engram_checkpoint(tmp_path, [1])
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table._shared_group = None
    table.vocab_start_idx = 0
    table.vocab_end_idx = 19
    table.block_size = 32
    table.weight = torch.nn.Parameter(torch.empty((19, 64), dtype=torch.int8), requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(torch.empty((19, 2), dtype=torch.float32), requires_grad=False)
    table.load_checkpoint(tmp_path, key, chunk_rows=7)
    decoded = npu.dequantize_engram_rows(table.weight, table.weight_scale_inv)
    expected = (checkpoint_weight.float().unflatten(-1, (-1, 32)) * checkpoint_scale.float().unsqueeze(-1)).flatten(-2)
    torch.testing.assert_close(decoded.float(), expected, rtol=0, atol=0.02)


def _fake_host_library(device_offset):
    class Library:
        def aclrtHostRegisterV2(self, pointer, size, flags):
            return 0

        def aclrtHostGetDevicePointer(self, pointer, out, flags):
            out._obj.value = pointer.value + device_offset
            return 0

        def aclrtHostUnregister(self, pointer):
            return 0

    return Library()


@pytest.mark.parametrize("leader_rank", [0, 16])
def test_shared_uva_uses_one_python_shared_memory_segment(monkeypatch, leader_rank):
    name_ready = threading.Event()
    attached = threading.Barrier(2)
    names: list[str] = []
    monkeypatch.setattr(npu, "_host_library", lambda: _fake_host_library(1 << 40))
    monkeypatch.setattr(npu.dist, "get_global_rank", lambda group, rank: leader_rank)
    monkeypatch.setattr(npu.dist, "barrier", lambda group: attached.wait(timeout=10))
    monkeypatch.setattr(npu.dist, "all_gather_object", lambda errors, error, group: None)

    def broadcast(payload, src, group):
        assert src == leader_rank
        if payload[0] is None:
            assert name_ready.wait(timeout=10)
            payload[0] = names[0]
        else:
            names.append(payload[0])
            name_ready.set()

    monkeypatch.setattr(npu.dist, "broadcast_object_list", broadcast)

    def create(rank):
        group = SimpleNamespace(cpu_group=None, rank_in_group=rank, world_size=2)
        return npu.SharedUvaBuffer((8, 32), torch.int8, "cpu", group)

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader, follower = list(pool.map(create, (0, 1)))
    leader.tensor.fill_(7)
    assert follower.tensor.tolist() == leader.tensor.tolist()
    assert len(names) == 1
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=names[0])
    leader.close()
    follower.close()


def test_shared_table_skips_per_step_dp_gather(monkeypatch):
    calls = []

    def gather(ids, *, dp_shared_memory=False):
        calls.append(dp_shared_memory)
        return ids if dp_shared_memory else torch.cat((ids + 100, ids))

    monkeypatch.setattr(embedding_mod, "gather_engram_hashes", gather)
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    table.embed_gathered = lambda ids, count: ids[:count]
    table._shared_group = object()
    ids = torch.tensor([[7, 8]])
    assert table.forward(ids).tolist() == [[7, 8]]
    table._shared_group = None
    table.dp_size = 2
    table.embed_gathered = lambda gathered, count: gathered[count : 2 * count]
    assert table.forward(ids).tolist() == [[7, 8]]
    assert calls == [True, False]


@pytest.mark.parametrize("num_tokens", [0, 2])
def test_idle_hashes_have_no_valid_rows(num_tokens):
    state = object.__new__(AscendNgramHashState)
    torch.nn.Module.__init__(state)
    state.multipliers = torch.empty(2, 3, dtype=torch.int64)
    state.primes = torch.empty(2, 2, 4, dtype=torch.int64)
    hashes, keep = state.dummy_hashes(torch.zeros(num_tokens, dtype=torch.int64))
    assert hashes.shape == (num_tokens, 2, 8)
    assert hashes.dtype == torch.int32
    assert (hashes == DEAD_ID).all()
    assert keep.shape == (num_tokens,)
    assert keep.dtype == torch.bool
    assert not keep.any()


def _runner(rows, computed, prompt):
    token_ids = np.full((len(rows), 16), -7, dtype=np.int32)
    for index, row in enumerate(rows):
        token_ids[index, : len(row)] = row
    runner = object.__new__(NPUModelRunner)
    runner.input_batch = SimpleNamespace(
        num_reqs=len(rows),
        token_ids_cpu=token_ids,
        num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
    )
    lookback = np.empty((len(rows), 3), dtype=np.int32)
    runner.lookback_token_ids = SimpleNamespace(
        np=lookback,
        copy_to_gpu=lambda: torch.from_numpy(lookback.copy()),
    )
    runner.is_pooling_model = False
    return runner


@pytest.mark.parametrize(
    "computed,num_reqs,expected",
    [
        (4, None, [13, 12, 11]),
        (6, None, [-1, -1, 13]),
        (0, None, [-1, -1, -1]),
        (4, 0, [-1, -1, -1]),
        (4, 1, [13, 12, 11]),
    ],
)
def test_v1_lookback_uses_prompt_tokens_once(computed, num_reqs, expected):
    runner = _runner([[10, 11, 12, 13, -7, -7]], [computed], [4])
    copy = Mock(wraps=runner.lookback_token_ids.copy_to_gpu)
    runner.lookback_token_ids.copy_to_gpu = copy
    kwargs = runner._init_model_kwargs(num_reqs=num_reqs)
    assert kwargs["lookback_token_ids"][0].tolist() == expected
    copy.assert_called_once_with()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("remote_group", ["tp", "edp"])
def test_engram_rejects_nonlocal_groups_before_allocation(monkeypatch, shared, remote_group):
    monkeypatch.setattr(embedding_mod, "get_tp_group", lambda: SimpleNamespace(cpu_group="tp"))
    monkeypatch.setattr(embedding_mod, "get_engram_dp_group", lambda: SimpleNamespace(cpu_group="edp"))
    monkeypatch.setattr(embedding_mod, "in_the_same_node_as", lambda pg: [True, pg != remote_group])
    error = "TP ranks" if remote_group == "tp" else "same node and shared-memory namespace"
    with pytest.raises(ValueError, match=error):
        embedding_mod.AscendParallelEngramEmbedding(96, 64, (4,) * 24, 0, dp_shared_memory=shared)


@pytest.mark.parametrize("dp_rank,num_tokens", [(2, 3), (3, 2), (3, 0)])
def test_engram_gather_uses_the_local_edp_token_slice(monkeypatch, dp_rank, num_tokens):
    """A replica pads to its own EDP slot, never to another node's prefill."""
    from vllm_ascend.models.deepseek_v41.engram import parallel as parallel_mod

    edp_group = SimpleNamespace(world_size=2, rank_in_group=dp_rank - 2, all_gather=lambda ids, dim=0: ids.repeat(2, 1))
    monkeypatch.setattr(parallel_mod, "get_engram_dp_group", lambda: edp_group)
    monkeypatch.setattr(parallel_mod, "get_dp_group", lambda: SimpleNamespace(rank_in_group=dp_rank))
    monkeypatch.setattr(parallel_mod, "is_pd_decode_recompute_scheduler_enabled", lambda: False)
    # This EDP starts at global DP rank 2, so its slice is (4, 2) and not the
    # 9 tokens another node is prefilling.
    monkeypatch.setattr(
        parallel_mod,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([1, 9, 4, 2]))),
    )
    ids = torch.full((num_tokens, 5), dp_rank + 10, dtype=torch.int32)
    gathered = embedding_mod.gather_engram_hashes(ids)
    assert gathered.shape == (8, 5)
    for replica in gathered.reshape(2, 4, 5):
        torch.testing.assert_close(replica[:num_tokens], ids)
        assert (replica[num_tokens:] == parallel_mod.DEAD_ID).all()


# --- Ascend EDP member-order risk (validation ladder step 2) -----------------
#
# Ascend builds the Engram DP group with ``all_ranks.permute(0, 2, 4, 1, 3)``
# (see ``init_ascend_model_parallel``), which differs from upstream's
# ``permute(0, 2, 3, 4, 1)``. Keeping the TP axis outermost makes every EDP
# group a single TP column, so ``engram_head_shard_rank =
# tp_rank * dp_size + dp_rank`` assigns every rank a unique shard and each EDP
# group owns a contiguous run of checkpoint-order heads. These tests lock that
# invariant so a future refactor back to the upstream permute cannot silently
# reintroduce shard collisions (upstream collides for dp == 1 and scenario D).
_EDP_TOPOLOGIES = [
    # (tp, dp, pp, pcp): every decomposition of an 8-rank world used by the
    # on-device validation, plus the degenerate TP=8 case.
    (2, 2, 1, 2),
    (4, 2, 1, 1),
    (2, 4, 1, 1),
    (2, 1, 1, 4),
    (4, 1, 1, 2),
    (8, 1, 1, 1),
]


def _ascend_edp_groups(tp: int, dp: int, pp: int, pcp: int) -> list[list[int]]:
    """Reconstruct the EDP groups exactly as ``init_ascend_model_parallel`` does."""
    world = tp * dp * pp * pcp
    all_ranks = torch.arange(world).reshape(1, dp, pp, pcp, tp)
    return all_ranks.permute(0, 2, 4, 1, 3).reshape(-1, dp * pcp).tolist()


def _upstream_edp_groups(tp: int, dp: int, pp: int, pcp: int) -> list[list[int]]:
    """Upstream reference ordering: ``all_ranks.permute(0, 2, 3, 4, 1)``."""
    world = tp * dp * pp * pcp
    all_ranks = torch.arange(world).reshape(1, dp, pp, pcp, tp)
    return all_ranks.permute(0, 2, 3, 4, 1).reshape(-1, dp * pcp).tolist()


@pytest.mark.parametrize("tp,dp,pp,pcp", _EDP_TOPOLOGIES)
def test_engram_edp_ordering_gives_unique_consecutive_shards(tp, dp, pp, pcp):
    """Ascend EDP groups are single TP columns, so shards never collide.

    ``engram_head_shard_rank = tp_rank * dp_size + dp_rank`` is collision-free
    under ``permute(0, 2, 4, 1, 3)`` for every world decomposition, and each
    group owns a consecutive run of heads that restores checkpoint order after
    the exchange. This is a structural invariant, not a per-topology accident.
    """
    world = tp * dp * pp * pcp
    edp = _ascend_edp_groups(tp, dp, pp, pcp)

    per_rank_shard: dict[int, int] = {}
    for group in edp:
        for dp_rank, rank in enumerate(group):
            tp_rank = rank % tp
            per_rank_shard[rank] = tp_rank * len(group) + dp_rank

    # The formula must agree with the EDP group structure: each group is a
    # single TP column (tp_rank is constant), and its members map to a
    # contiguous shard run.
    for group in edp:
        assert len({rank % tp for rank in group}) == 1
        shards = sorted(per_rank_shard[rank] for rank in group)
        assert shards == list(range(shards[0], shards[0] + len(group)))
    assert sorted(per_rank_shard.values()) == list(range(world))


@pytest.mark.parametrize("tp,dp,pp,pcp", _EDP_TOPOLOGIES)
def test_engram_edp_ordering_differs_from_upstream_only_when_needed(tp, dp, pp, pcp):
    """Ascend's permute diverges from upstream exactly where upstream collides.

    For ``dp == 1`` (pure PCP) and the PCP>1 scenario D, upstream's
    ``permute(0, 2, 3, 4, 1)`` groups ranks with the same tp_rank *and* same
    dp_rank into one shard, so ``engram_head_shard_rank`` collides. Ascend's
    ordering is what keeps those topologies working, so the divergence is the
    point of this test rather than something to be removed.
    """
    edp_up = _upstream_edp_groups(tp, dp, pp, pcp)
    edp_asc = _ascend_edp_groups(tp, dp, pp, pcp)
    # The two permutes coincide when PCP == 1; with PCP > 1 upstream groups
    # ranks sharing tp_rank and dp_rank into one shard, so it collides and
    # Ascend's single-TP-column ordering is what keeps the topology working.
    assert (edp_up == edp_asc) == (pcp == 1)


@pytest.mark.parametrize("tp,dp,pp,pcp", _EDP_TOPOLOGIES)
@pytest.mark.parametrize("slot", [1, 3])
def test_engram_exchange_restores_checkpoint_head_order(tp, dp, pp, pcp, slot):
    """Faithful all_to_all under Ascend EDP ordering yields checkpoint-order heads.

    Models production end to end: each EDP peer's tokens are gathered rank-major
    (``gather_engram_hashes``), each rank looks up only its own shard
    (``lookup``), and ``exchange_engram_rows``'s all_to_all + view/permute
    reassembles the heads. The all_to_all sends each source's block indexed by
    the destination's ``rank_in_group`` — the real ``dist.all_to_all_single``
    semantics, not a symmetric stand-in.
    """
    world = tp * dp * pp * pcp
    edp = _ascend_edp_groups(tp, dp, pp, pcp)
    per_rank_shard: dict[int, int] = {}
    for group in edp:
        for dp_rank, rank in enumerate(group):
            per_rank_shard[rank] = (rank % tp) * len(group) + dp_rank
    # ``embed_gathered`` skips the exchange when the EDP group is a single
    # rank (``dp_size == 1`` early return), so there is no all_to_all to test.
    if dp * pcp == 1:
        return

    def group_of(rank):
        return next(g for g in edp if rank in g)

    for rank in range(world):
        group = group_of(rank)
        edp_size = len(group)
        rank_in_group = group.index(rank)
        # gathered[peer*slot + t] = token t of EDP peer `peer`, value peer*10 + t.
        token_value = {peer: [peer * 10 + t for t in range(slot)] for peer in group}
        # Each rank looks up only its own shard head, so row i is its embedding
        # of EDP peer i's tokens, tagged with this rank's shard.
        staged = torch.zeros((edp_size * slot, 1, 1))
        for i, peer in enumerate(group):
            for t in range(slot):
                value = token_value[peer][t] * 100 + per_rank_shard[rank]
                staged[i * slot + t, 0, 0] = value
        # Faithful all_to_all: recv chunk src_ig holds src_ig's block destined
        # for me (rank_in_group == rank_in_group); the value is that peer's
        # embedding of *my* tokens under its own shard.
        my_tokens = token_value[group[rank_in_group]]
        recv = torch.zeros((edp_size * slot, 1, 1))
        for src_ig, src in enumerate(group):
            for t in range(slot):
                recv[src_ig * slot + t, 0, 0] = my_tokens[t] * 100 + per_rank_shard[src]
        # exchange_engram_rows body.
        rows = recv.view(edp_size, slot, 1, 1).permute(1, 0, 2, 3)
        out = rows[:slot].reshape(slot, edp_size, 1)
        heads = [per_rank_shard[r] for r in group]
        assert heads == sorted(heads) and len(set(heads)) == len(heads)
        for t in range(slot):
            expected = [my_tokens[t] * 100 + head for head in heads]
            got = out[t].squeeze()
            torch.testing.assert_close(got, torch.tensor(expected, dtype=out.dtype))


def test_native_mxfp8_preserves_shard_bits_and_lookup(tmp_path, monkeypatch):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    codes = torch.linspace(-32, 32, 19 * 64).reshape(19, 64).to(torch.float8_e4m3fn)
    scale_bits = (torch.arange(19 * 2).reshape(19, 2) % 7 + 123).to(torch.uint8)
    scales = scale_bits.view(torch.float8_e8m0fnu)
    save_file({key: codes, scale_key: scales}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors", scale_key: "model.safetensors"}})
    )
    assert embedding_mod.engram_storage_dtype(tmp_path, 1) == torch.float8_e4m3fn
    embedding_mod.preflight_engram_checkpoint(tmp_path, [1])
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table._shared_group = None
    table.vocab_start_idx, table.vocab_end_idx = 5, 17
    table.block_size, table.dim = 32, 64
    table.head_start, table.part_n_hash_cols = 1, 2
    table.weight = torch.nn.Parameter(torch.empty((12, 64), dtype=torch.float8_e4m3fn), requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(torch.empty((12, 2), dtype=torch.uint8), requires_grad=False)
    monkeypatch.setattr(embedding_mod, "quantize_engram_rows", Mock(side_effect=AssertionError("requantization")))
    table.load_checkpoint(tmp_path, key, chunk_rows=5)
    torch.testing.assert_close(table.weight.view(torch.uint8), codes[5:17].view(torch.uint8))
    torch.testing.assert_close(table.weight_scale_inv, scale_bits[5:17])
    ids = torch.tensor([[0, 5, 16], [0, -1, 19]])
    out = torch.empty((2, 2, 64), dtype=torch.bfloat16)
    embedding_mod._torch_lookup(table, ids, out)
    reference = (codes.float().unflatten(-1, (-1, 32)) * scales.float().unsqueeze(-1)).flatten(-2).bfloat16()
    torch.testing.assert_close(out[0], reference[[5, 16]], rtol=0, atol=0)
    assert not out[1].any()


def test_native_mxfp8_requires_e8m0_scales_before_allocation(tmp_path):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    save_file(
        {key: torch.zeros(8, 64).to(torch.float8_e4m3fn), scale_key: torch.ones(8, 2)},
        tmp_path / "model.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors", scale_key: "model.safetensors"}})
    )
    with pytest.raises(ValueError, match="requires E8M0 scales"):
        embedding_mod.preflight_engram_checkpoint(tmp_path, [1])
