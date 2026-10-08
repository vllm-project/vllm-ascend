# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Focused tests for the Ascend Engram configuration and storage path."""

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

from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod
from vllm_ascend.models.deepseek_v41.engram import npu
from vllm_ascend.models.deepseek_v41.engram.common import engram_gate
from vllm_ascend.models.deepseek_v41.engram.hash_state import DEAD_ID, AscendNgramHashState
from vllm_ascend.models.deepseek_v41.engram.parallel import resolve_dp_shared_memory
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_shared_memory_needs_a_local_dp_peer(monkeypatch):
    from vllm_ascend.models.deepseek_v41.engram import parallel

    monkeypatch.setattr(parallel, "get_engram_dp_size", lambda: 1)
    assert not resolve_dp_shared_memory(True)
    monkeypatch.setattr(parallel, "get_engram_dp_size", lambda: 2)
    assert resolve_dp_shared_memory(True)
    assert not resolve_dp_shared_memory(False)


@pytest.mark.parametrize("quantized", [False, True])
def test_loader_preserves_checkpoint_storage(tmp_path, quantized):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    source = torch.linspace(-12, 12, 19 * 64).reshape(19, 64).bfloat16()
    codes, scales = npu.quantize_engram_rows(source)
    save_file({key: source}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "model.safetensors"}}))
    if quantized:
        save_file({key: codes, scale_key: scales}, tmp_path / "quant.safetensors")
        (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "quant.safetensors", scale_key: "quant.safetensors"}})
        )
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table._shared_group = None
    table.vocab_start_idx = 0
    table.vocab_end_idx = 19
    expected = codes if quantized else source
    table.weight = torch.nn.Parameter(torch.empty_like(expected), requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(torch.empty_like(scales), requires_grad=False) if quantized else None
    table.load_checkpoint(tmp_path, key, chunk_rows=7)
    assert torch.equal(table.weight, expected)
    if quantized:
        assert torch.equal(table.weight_scale_inv, scales)


def test_bf16_gate_without_rotation():
    hidden = torch.ones(2, 4, 64, dtype=torch.bfloat16)
    value = torch.full((2, 64), 0.25, dtype=torch.bfloat16)
    mask = torch.tensor([True, False])
    result = engram_gate(hidden, hidden * 2, value, torch.ones(4, 64), None, mask, 1e-20)
    # Normalized dot is sqrt(64) = 8; the gate applies signed sqrt, then sigmoid.
    expected = (1 + 0.25 * torch.sigmoid(torch.tensor(8.0).sqrt())).bfloat16()
    assert torch.all(result[0] == expected)
    assert torch.equal(result[1], hidden[1])


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
    from vllm.models.deepseek_v41.nvidia import engram as parallel_mod

    edp_group = SimpleNamespace(world_size=2, rank_in_group=dp_rank - 2, all_gather=lambda ids, dim=0: ids.repeat(2, 1))
    monkeypatch.setattr(parallel_mod, "get_engram_dp_group", lambda: edp_group)
    monkeypatch.setattr(parallel_mod, "get_dp_group", lambda: SimpleNamespace(rank_in_group=dp_rank))
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


@pytest.mark.parametrize("indices", [[0, 1, 6, 7], [2, 3, 4, 5], []])
def test_pcp_hashes_global_history_but_looks_up_local_rows(monkeypatch, indices):
    from vllm_ascend.models.deepseek_v41 import model as model_mod

    class HashState:
        lookback_depth = 2
        use_slot_cache = False

        def ensure_cache(self):
            return True

        def __call__(self, ids, positions, starts, dead, history, *args):
            # Two-token n-grams cross PCP fragments and use prior-chunk history.
            previous = ids.roll(1)
            previous[starts[:-1].long()] = history[:, 0]
            return (ids * 100 + previous).reshape(-1, 1, 1).int()

    config = SimpleNamespace(engram_layer_ids=[0], image_token_id=999, image_pad_token_id=1000)
    table = SimpleNamespace(
        n_hash_cols=1, dim=1, embed_gathered=Mock(side_effect=lambda ids, count: ids[:count].bfloat16())
    )
    model = SimpleNamespace(
        config=config,
        engram_hash=HashState(),
        engram_dp_shared_memory=False,
        layers=[SimpleNamespace(engram=SimpleNamespace(embed_tokens=table))],
    )
    monkeypatch.setattr(model_mod, "get_engram_dp_size", lambda: 1)
    monkeypatch.setattr(model_mod, "gather_engram_hashes", lambda hashes, **kwargs: hashes)
    ids = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int32)
    selected = torch.tensor(indices, dtype=torch.int64)
    lookups, mask = model_mod.DeepseekV41Model.prepare_engram(
        model,
        ids,
        torch.arange(8),
        torch.tensor([[9, 8], [19, 18]], dtype=torch.int32),
        torch.tensor([0, 4, 8]),
        torch.arange(8),
        torch.zeros(2, 1),
        selected,
    )
    expected = torch.tensor([1009, 1110, 1211, 1312, 2019, 2120, 2221, 2322], dtype=torch.int32)
    table.embed_gathered.assert_called_once()
    gathered, count = table.embed_gathered.call_args.args
    torch.testing.assert_close(gathered[:, 0], expected[selected])
    assert count == len(indices)
    torch.testing.assert_close(lookups[0][:, 0], expected[selected].bfloat16())
    assert mask.tolist() == [True] * len(indices)
