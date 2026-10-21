# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise RFork host state with real CPU tensors and checkpoint files."""

import json
import logging
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from .rfork_test_support import RFORK_ROOT, _load_module, _stub


def test_unknown_cpu_table_cannot_opt_in_with_a_model_hook(tensor_runtime):
    model = torch.nn.Module()
    model.table = torch.nn.Module()
    model.table.weight = torch.nn.Parameter(torch.ones(2))
    model.table.get_rfork_local_state_tensors = lambda: (model.table.weight,)
    assert tensor_runtime.tensor_layout.find_non_npu_state_tensors(model) == ["table.weight"]


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


def make_table(embedding_runtime, tmp_path, dtype=torch.int8, *, offload=True):
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
    table = embedding_runtime.AscendParallelEngramEmbedding(8, 32, (4, 4), 0, cpu_offload=offload, storage_dtype=dtype)
    table.bind_checkpoint(tmp_path, key)
    return table, tensors


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
@pytest.mark.parametrize("entrypoint", ["rfork", "default"])
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


def test_device_table_does_not_opt_into_local_loading(local_weights, embedding_runtime, tmp_path):
    table, _ = make_table(embedding_runtime, tmp_path, offload=False)
    table.load_checkpoint = Mock(side_effect=AssertionError("device weights belong to RFork"))
    local_weights.load_local_weights(table)
    local_weights.record_loaded_local_weights(table)
    assert local_weights.get_local_tensor_ids(table) == set()


@pytest.mark.parametrize("mutation", ["parameter", "storage", "checkpoint", "close", "scale"])
def test_replacing_or_releasing_loaded_storage_revokes_readiness(local_weights, embedding_runtime, tmp_path, mutation):
    table, _ = make_table(embedding_runtime, tmp_path)
    local_weights.load_local_weights(table)
    if mutation == "parameter":
        table.weight = torch.nn.Parameter(torch.zeros_like(table.weight), requires_grad=False)
    elif mutation == "storage":
        table.weight.data = torch.zeros_like(table.weight)
    elif mutation == "checkpoint":
        table.bind_checkpoint(tmp_path, "another.weight")
    elif mutation == "scale":
        table.weight_scale_inv = torch.nn.Parameter(torch.zeros_like(table.weight_scale_inv), requires_grad=False)
    else:
        table._codes_uva = Mock()
        table.close_host_offload()
    assert local_weights.get_local_tensor_ids(table) == set()


def test_local_state_does_not_retain_replaced_parameters(local_weights, embedding_runtime, tmp_path):
    table, _ = make_table(embedding_runtime, tmp_path)
    local_weights.load_local_weights(table)
    original = weakref.ref(table.weight)
    table.weight = torch.nn.Parameter(torch.zeros_like(table.weight), requires_grad=False)
    assert original() is None


def test_failed_rfork_reload_revokes_readiness(local_weights, embedding_runtime, tmp_path):
    table, _ = make_table(embedding_runtime, tmp_path)
    local_weights.load_local_weights(table)
    table.bind_checkpoint(tmp_path, "another.weight")
    table._load_into_storage = Mock(side_effect=OSError("partial checkpoint read"))
    with pytest.raises(OSError, match="partial checkpoint"):
        local_weights.load_local_weights(table)
    assert local_weights.get_local_tensor_ids(table) == set()


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


def test_partial_close_revokes_readiness_and_can_be_retried(local_weights, embedding_runtime, tmp_path):
    table, _ = make_table(embedding_runtime, tmp_path)
    local_weights.load_local_weights(table)
    codes, scales = Mock(), Mock()
    scales.close.side_effect = RuntimeError("host unregister failed")
    table._codes_uva, table._scales_uva = codes, scales
    with pytest.raises(RuntimeError, match="host unregister failed"):
        local_weights.release_local_weights(table)
    assert local_weights.get_local_tensor_ids(table) == set()
    assert table.weight.numel() == 0
    assert table.weight_scale_inv.numel() > 0
    scales.close.side_effect = None
    local_weights.release_local_weights(table)
    codes.close.assert_called_once()
    assert scales.close.call_count == 2
    assert table.weight_scale_inv.numel() == 0


def test_failed_first_table_also_releases_tables_not_yet_loaded(local_weights, embedding_runtime, tmp_path):
    model = torch.nn.Module()
    model.first, _ = make_table(embedding_runtime, tmp_path)
    model.second, _ = make_table(embedding_runtime, tmp_path)
    model.first._load_into_storage = Mock(side_effect=OSError("partial checkpoint read"))
    model.second.load_checkpoint = Mock(side_effect=AssertionError("unreachable"))
    first_codes, second_codes = Mock(), Mock()
    model.first._codes_uva, model.second._codes_uva = first_codes, second_codes
    with pytest.raises(OSError):
        local_weights.load_local_weights(model)
    local_weights.release_local_weights(model)
    first_codes.close.assert_called_once()
    second_codes.close.assert_called_once()
    assert local_weights.get_local_tensor_ids(model) == set()


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_shared_load_synchronizes_success_and_failure(
    local_weights, embedding_runtime, tmp_path, monkeypatch, failure, rank
):
    table, _ = make_table(embedding_runtime, tmp_path)
    table._shared_group = SimpleNamespace(rank_in_group=rank, world_size=2, cpu_group="shared")
    table._load_into_storage = Mock(side_effect=OSError("writer failed") if failure else None)

    def synchronize(errors, error, *, group):
        assert local_weights.get_local_tensor_ids(table) == set()
        assert error == ("OSError: writer failed" if rank == 0 and failure else None)
        assert group == "shared"
        errors[:] = ["writer failed" if failure else None, None]

    monkeypatch.setattr(embedding_runtime.dist, "all_gather_object", synchronize)
    if failure:
        with pytest.raises(RuntimeError, match="writer failed"):
            local_weights.load_local_weights(table)
        assert local_weights.get_local_tensor_ids(table) == set()
    else:
        local_weights.load_local_weights(table)
        assert id(table.weight) in local_weights.get_local_tensor_ids(table)
    assert table._load_into_storage.call_count == (1 if rank == 0 else 0)


@pytest.mark.parametrize("entrypoint", ["rfork", "default"])
@pytest.mark.parametrize("processed", [False, True])
def test_host_table_is_ready_but_only_device_weight_is_registered(
    tensor_runtime, local_weights, embedding_runtime, tmp_path, monkeypatch, entrypoint, processed
):
    table, _ = make_table(embedding_runtime, tmp_path)
    model = torch.nn.Module()
    model.table = table
    model.device_weight = torch.nn.Parameter(torch.ones(4))
    # UVA addresses belong to this process; even the processed-layout scan
    # must leave these private runtime pointers out of the manifest.
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


@pytest.mark.parametrize("processed", [False, True])
@pytest.mark.parametrize(
    ("reuse", "outcome"),
    [
        (False, "transfer"),
        (False, "seed_miss"),
        (False, "transfer_failure"),
        (False, "local_failure"),
        (False, "state_failure"),
        (False, "release_failure"),
        (False, "init_failure"),
        (False, "device_init_failure"),
        (False, "unconfigured_init_failure"),
        (True, "transfer"),
        (True, "local_failure"),
    ],
)
def test_loader_loads_local_state_before_registration_or_shared_reuse(
    rfork_helpers, monkeypatch, processed, reuse, outcome
):
    """Check load/release ordering and prevent unsafe retries after partial construction."""
    runtime = rfork_helpers.loader
    model = torch.nn.Module()
    events = []
    model.table = torch.nn.Module()
    prepare_model_state = runtime._prepare_rfork_model_state

    def prepare_state(model, model_config, load_config):
        events.append("prepare")
        if outcome == "state_failure":
            raise ValueError("checkpoint metadata unavailable")
        prepare_model_state(model, model_config, load_config)

    monkeypatch.setattr(runtime, "_prepare_rfork_model_state", prepare_state)

    def load_local(model):
        events.append("local")
        if outcome == "local_failure":
            raise OSError("checkpoint unavailable")

    monkeypatch.setattr(runtime, "load_local_weights", load_local)

    def release_local(model):
        events.append("release")
        if outcome == "release_failure":
            raise RuntimeError("host unregister failed")

    monkeypatch.setattr(runtime, "release_local_weights", release_local)
    monkeypatch.setattr(runtime, "record_loaded_local_weights", lambda model: events.append("ready"))

    def record(event, result):
        events.append(event)
        return result

    session = SimpleNamespace(
        can_reuse_shared_weights=lambda *args: record("reuse", reuse),
        register_destination=lambda *args: record("register", True),
        acquire_seed=lambda: outcome not in ("seed_miss", "release_failure"),
        transfer_from_seed=lambda *args: outcome != "transfer_failure",
        log_transferred_model_layout=lambda *args: None,
        prepare_for_fallback=lambda: SimpleNamespace(can_schedule_seed=True),
    )
    loader = object.__new__(runtime.RForkModelLoader)
    loader.load_config = SimpleNamespace(device=None)
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda *args: session)
    monkeypatch.setattr(loader, "_get_target_registered_blocks", lambda *args: [(1, 2)] if reuse else [])
    monkeypatch.setattr(loader, "_requires_processed_layout_transfer", lambda *args: processed)
    monkeypatch.setattr(runtime, "mutable_weights_bypass_reason", lambda *args: None)
    monkeypatch.setattr(runtime, "get_ascend_config", lambda: SimpleNamespace(eplb_config=SimpleNamespace()))
    monkeypatch.setattr(runtime, "_is_dynamic_eplb_enabled", lambda *args: False)
    monkeypatch.setattr(runtime, "_snapshot_process_global_model_state", lambda *args: None)
    monkeypatch.setattr(runtime, "_reset_process_global_model_state", lambda *args: events.append("reset"))
    monkeypatch.setattr(runtime, "gc", SimpleNamespace(collect=Mock(return_value=0)))
    fallback_model = torch.nn.Module()
    monkeypatch.setattr(runtime, "_load_with_default_loader", lambda *args: record("default", fallback_model))
    initialization_error = RuntimeError("model construction failed after an earlier table allocation")

    def initialize_model(**kwargs):
        if outcome.endswith("init_failure"):
            events.append("init_failure")
            raise initialization_error
        return model

    monkeypatch.setattr(runtime, "initialize_model", initialize_model)
    monkeypatch.setattr(runtime, "_rfork_skip_unquantized_moe_post_load_processing", lambda *args: nullcontext())
    monkeypatch.setattr(runtime, "_start_rfork_seed_service", lambda *args, **kwargs: events.append("seed"))
    monkeypatch.setattr(
        runtime.torch, "npu", SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None), raising=False
    )
    config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"))
    if outcome != "unconfigured_init_failure":
        config.engram_config = SimpleNamespace(cpu_offload=outcome != "device_init_failure")
    args = (config, SimpleNamespace(dtype=torch.float32))
    if outcome == "init_failure":
        with pytest.raises(RuntimeError, match="cannot safely retry") as error:
            loader.load_model(*args)
        assert error.value.__cause__ is initialization_error
        assert events == ["init_failure"]
        return
    if outcome == "release_failure" and not reuse:
        with pytest.raises(RuntimeError, match="host unregister failed"):
            loader.load_model(*args)
        assert "default" not in events
        return
    result = loader.load_model(*args)
    if outcome.endswith("init_failure"):
        assert result is fallback_model
        assert events == ["init_failure", "reset", "default", "ready", "seed"]
    elif outcome == "state_failure":
        assert result is fallback_model
        assert events == ["prepare", "reset", "release", "default"]
    elif outcome == "local_failure":
        assert result is fallback_model
        assert events == ["prepare", "local", "reset", "release", "default", "ready", "seed"]
    elif reuse:
        assert result is model
        assert events == ["prepare", "local", "reuse"]
    elif outcome != "transfer":
        assert result is fallback_model
        assert events == ["prepare", "local", "register", "reset", "release", "default", "ready", "seed"]
    else:
        assert result is model
        assert events == ["prepare", "local", "register", "seed"]
