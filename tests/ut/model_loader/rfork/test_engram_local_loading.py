# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise RFork host state with real CPU tensors and checkpoint files."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from .rfork_test_support import RFORK_ROOT, _load_module, _stub


@pytest.fixture
def embedding_runtime(monkeypatch):
    def set_weight_attrs(tensor, attrs):
        for name, value in attrs.items():
            setattr(tensor, name, value)

    prefix = "vllm_ascend.models.deepseek_v41.engram"
    _stub(monkeypatch, "vllm", __path__=[])
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("engram-local-load-test"))
    _stub(
        monkeypatch,
        "vllm.distributed",
        get_tensor_model_parallel_rank=lambda: 1,
        get_tensor_model_parallel_world_size=lambda: 2,
        tensor_model_parallel_all_gather=Mock(),
    )
    _stub(
        monkeypatch,
        "vllm.distributed.parallel_state",
        get_tp_group=lambda: SimpleNamespace(cpu_group="tp"),
        in_the_same_node_as=lambda group: [True, True],
    )
    _stub(monkeypatch, "vllm.model_executor.utils", set_weight_attrs=set_weight_attrs)
    _stub(monkeypatch, "vllm.models.deepseek_v41.common.engram", ParallelEngramEmbedding=torch.nn.Module)
    for name in ("vllm_ascend", "vllm_ascend.models", "vllm_ascend.models.deepseek_v41", prefix):
        _stub(monkeypatch, name, __path__=[])
    _stub(
        monkeypatch,
        f"{prefix}.npu",
        HostUvaBuffer=Mock(),
        SharedUvaBuffer=Mock(),
        gather_dequantize_engram_int8=Mock(),
        gather_dequantize_host_uva=Mock(),
        quantize_engram_rows=Mock(side_effect=AssertionError("native checkpoint must not be requantized")),
    )
    _stub(
        monkeypatch,
        f"{prefix}.parallel",
        engram_head_shard_rank=lambda: 1,
        exchange_engram_rows=Mock(),
        gather_engram_hashes=Mock(),
        get_engram_dp_group=lambda: None,
        get_engram_dp_size=lambda: 1,
    )
    module = _load_module(
        monkeypatch,
        f"{prefix}.embedding",
        str(RFORK_ROOT.parents[1] / "models/deepseek_v41/engram/embedding.py"),
    )

    def allocate(table):
        weight = torch.zeros(table.part_num_embeddings, table.dim, dtype=table.storage_dtype)
        scales = None
        if table.storage_dtype != torch.bfloat16:
            scales = torch.zeros(table.part_num_embeddings, table.dim // table.block_size, dtype=table.scale_dtype)
        return weight, scales

    monkeypatch.setattr(module.AscendParallelEngramEmbedding, "_allocate_weights", allocate)
    return module


def make_table(embedding_runtime, tmp_path, dtype=torch.int8):
    key = "layers.1.engram.embed.weight"
    scale_key = "layers.1.engram.embed.scale"
    weights = torch.arange(8 * 32).reshape(8, 32).to(dtype)
    tensors = {key: weights}
    if dtype != torch.bfloat16:
        tensors[scale_key] = torch.arange(1, 9, dtype=torch.float32).reshape(8, 1)
    save_file(tensors, tmp_path / "table.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "table.safetensors")})
    )
    table = embedding_runtime.AscendParallelEngramEmbedding(8, 32, (4, 4), 0, cpu_offload=True, storage_dtype=dtype)
    table.bind_checkpoint(tmp_path, key)
    return table, tensors


@pytest.mark.parametrize("dtype, entrypoint", [(torch.bfloat16, "rfork"), (torch.int8, "default")])
def test_checkpoint_load_marks_complete_shard_ready(local_weights, embedding_runtime, tmp_path, dtype, entrypoint):
    table, tensors = make_table(embedding_runtime, tmp_path, dtype)
    assert local_weights.get_local_tensor_ids(table) == set()
    if entrypoint == "rfork":
        local_weights.load_local_weights(table)
    else:
        table._weight_loader(table.weight, None)
        # The default loader has no RFork instrumentation.
        assert local_weights.get_local_tensor_ids(table) == set()
        local_weights.record_loaded_local_weights(table)
    torch.testing.assert_close(table.weight, tensors["layers.1.engram.embed.weight"][4:8])
    ready = {id(table.weight)}
    if dtype == torch.int8:
        ready.add(id(table.weight_scale_inv))
        torch.testing.assert_close(table.weight_scale_inv, tensors["layers.1.engram.embed.scale"][4:8])
    assert local_weights.get_local_tensor_ids(table) == ready
    table._load_into_storage = Mock(side_effect=AssertionError("already loaded"))
    local_weights.load_local_weights(table)


@pytest.mark.parametrize("loaded", [False, True])
def test_local_release_closes_both_host_buffers(local_weights, embedding_runtime, tmp_path, loaded):
    table, _ = make_table(embedding_runtime, tmp_path)
    if loaded:
        local_weights.load_local_weights(table)
    else:
        table._load_into_storage = Mock(side_effect=OSError("partial checkpoint read"))
        with pytest.raises(OSError):
            local_weights.load_local_weights(table)
    codes, scales = Mock(), Mock()
    table._codes_uva, table._scales_uva = codes, scales
    local_weights.release_local_weights(table)
    codes.close.assert_called_once()
    scales.close.assert_called_once()
    assert table.weight.numel() == table.weight_scale_inv.numel() == 0
    assert local_weights.get_local_tensor_ids(table) == set()
    local_weights.release_local_weights(table)
    codes.close.assert_called_once()


def test_shared_target_table_is_neither_reloaded_nor_released_by_draft(local_weights, embedding_runtime, tmp_path):
    table, _ = make_table(embedding_runtime, tmp_path)
    target, draft = torch.nn.Module(), torch.nn.Module()
    target.table = draft.table = table
    local_weights.load_local_weights(target)
    table.load_checkpoint = Mock(side_effect=AssertionError("target must not be reloaded"))
    table.close_host_offload = Mock(side_effect=AssertionError("target must not be released"))
    local_weights.load_local_weights(draft)
    local_weights.release_local_weights(draft)
    assert local_weights.get_local_tensor_ids(draft) == {id(table.weight), id(table.weight_scale_inv)}


@pytest.mark.parametrize("entrypoint, processed", [("rfork", False), ("default", True)])
def test_host_table_is_ready_but_only_device_weight_is_registered(
    tensor_runtime, local_weights, embedding_runtime, tmp_path, monkeypatch, entrypoint, processed
):
    table, _ = make_table(embedding_runtime, tmp_path)
    model = torch.nn.Module()
    model.table = table
    model.device_weight = torch.nn.Parameter(torch.ones(4))
    # UVA pointers are process-local and must stay out of the manifest.
    pointers = torch.tensor([12345], dtype=torch.int64)
    table._codes_uva = SimpleNamespace(ptrs=pointers)
    monkeypatch.setattr(
        tensor_runtime.tensor_layout,
        "is_tensor_on_transfer_device",
        lambda tensor: tensor is model.device_weight or tensor is pointers,
    )
    backend = tensor_runtime.RForkTransferBackend()
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=Mock(return_value=SimpleNamespace(is_error=lambda: False))
    )
    backend._memory_registration_cls = lambda *args: args
    monkeypatch.setattr(tensor_runtime.transfer_backend, "read_npu_format", lambda tensor: 0)
    block = {"address": model.device_weight.data_ptr(), "size": 16, "state": "active_allocated"}
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=lambda: [{"blocks": [block]}])),
        raising=False,
    )
    assert not backend.register_memory_region(model, processed)
    backend.transfer_engine.batch_register_memory_ex.assert_not_called()
    if entrypoint == "rfork":
        local_weights.load_local_weights(model)
    else:
        table._weight_loader(table.weight, None)
        local_weights.record_loaded_local_weights(model)
    assert backend.register_memory_region(model, processed)
    assert list(backend.weight_manifest) == ["device_weight"]
    backend.transfer_engine.batch_register_memory_ex.assert_called_once()
