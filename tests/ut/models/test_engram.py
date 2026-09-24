# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for the Ascend Engram configuration and storage path."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod
from vllm_ascend.models.deepseek_v41.engram import npu
from vllm_ascend.patch.platform.patch_engram_config import AscendEngramConfig
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _topology(tp=4, dp=4, **overrides):
    values = dict(
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        data_parallel_size_local=dp,
        data_parallel_external_lb=False,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        nnodes=1,
        enable_elastic_ep=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("tp,dp", [(2, 8), (4, 4), (8, 2)])
@pytest.mark.parametrize("external", [False, True])
def test_dp_shared_memory_config_and_topologies(tp, dp, external):
    config = AscendEngramConfig(cpu_offload=True, dp_shared_memory=True)
    config.verify_parallel_config(
        _topology(tp, dp, data_parallel_external_lb=external, data_parallel_size_local=1 if external else dp)
    )
    config.verify_model_config(
        SimpleNamespace(
            architecture="DeepseekV41ForCausalLM",
            hf_text_config=SimpleNamespace(engram_layer_ids=[1, 14]),
        )
    )
    assert not AscendEngramConfig().dp_shared_memory
    with pytest.raises(ValueError, match="cpu_offload"):
        # vLLM main defaults VLLM_PLE_CPU_OFFLOAD to True, so force it off here
        # to exercise the dp_shared_memory -> cpu_offload validation.
        AscendEngramConfig(cpu_offload=False, dp_shared_memory=True)
    with pytest.raises(ValueError, match="single-node"):
        config.verify_parallel_config(_topology(tp, dp, nnodes=2))


def test_dp_shared_memory_survives_cli_parsing():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    value = {"cpu_offload": True, "dp_shared_memory": True}
    direct = EngineArgs(engram_config=value).engram_config
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = parser.parse_args(["--engram-config", json.dumps(value)]).engram_config
    assert isinstance(direct, AscendEngramConfig) and direct.dp_shared_memory
    assert isinstance(parsed, AscendEngramConfig) and parsed.dp_shared_memory


def test_loader_prefers_quantized_checkpoint(tmp_path):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    source = torch.linspace(-12, 12, 19 * 64).reshape(19, 64).bfloat16()
    codes, scales = npu.quantize_engram_rows(source)
    save_file({key: source}, tmp_path / "model.safetensors")
    save_file({key: codes, scale_key: scales}, tmp_path / "quant.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "model.safetensors"}}))
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "quant.safetensors", scale_key: "quant.safetensors"}})
    )
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table._shared_group = None
    table.vocab_start_idx = 0
    table.vocab_end_idx = 19
    table.weight = torch.nn.Parameter(torch.empty_like(codes), requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(torch.empty_like(scales), requires_grad=False)
    table.load_checkpoint(tmp_path, key)
    assert torch.equal(table.weight, codes)
    assert torch.equal(table.weight_scale_inv, scales)


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


def test_shared_uva_uses_one_python_shared_memory_segment(monkeypatch):
    name_ready = threading.Event()
    attached = threading.Barrier(2)
    names: list[str] = []
    monkeypatch.setattr(npu, "_host_library", lambda: _fake_host_library(1 << 40))
    monkeypatch.setattr(npu.dist, "get_global_rank", lambda group, rank: 0)
    monkeypatch.setattr(npu.dist, "barrier", lambda group: attached.wait(timeout=10))
    monkeypatch.setattr(npu.dist, "all_gather_object", lambda errors, error, group: None)

    def broadcast(payload, src, group):
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


def test_v1_lookback_uses_prompt_tokens_only():
    runner = _runner([[10, 11, 12, 13, -7, -7]], [4], [4])
    prompt = runner._prepare_lookback_token_ids(1).numpy()
    assert prompt[0].tolist() == [13, 12, 11]
    runner.input_batch.num_computed_tokens_cpu[0] = 6
    generated = runner._prepare_lookback_token_ids(1).numpy()
    assert generated[0].tolist() == [-1, -1, 13]


@pytest.mark.parametrize("shared", [False, True])
def test_engram_rejects_dp_outside_shared_node_before_allocation(monkeypatch, shared):
    group = SimpleNamespace(cpu_group=object())
    monkeypatch.setattr(embedding_mod, "get_engram_dp_group", lambda: group)
    monkeypatch.setattr(embedding_mod, "in_the_same_node_as", lambda pg: [True, False])
    with pytest.raises(ValueError, match="same node and shared-memory namespace"):
        embedding_mod.AscendParallelEngramEmbedding(96, 64, (4,) * 24, 0, dp_shared_memory=shared)
