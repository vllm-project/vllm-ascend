# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the shared Qwen PLE checkpoint and lifetime contract."""

import ctypes
import json
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

import vllm_ascend.ops  # noqa: F401 - initialize operators before importing PLE
from vllm_ascend.models.host_offload import track_host_offload_owners
from vllm_ascend.models.qwen4_exp import ple_embedding as ple


@pytest.fixture
def checkpoint(tmp_path):
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    data = torch.arange(5 * 160, dtype=torch.float32).reshape(5, 160).to(torch.bfloat16)
    mapping = {}
    for i, shard in enumerate((data[:3], data[3:])):
        key, filename = f"{prefix}.shard_{i}.weight", f"part-{i}.safetensors"
        save_file({key: shard.contiguous()}, str(tmp_path / filename))
        mapping[key] = filename
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"ple_layer_ids": [2], "split_ngram_parts": 2}}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return tmp_path, data


def test_checkpoint_inference_and_padded_bm_publication(checkpoint, monkeypatch):
    root, data = checkpoint
    from vllm_ascend.models.deepseek_v41.engram.shared_pool import has_ple_checkpoint

    assert has_ple_checkpoint(root)
    table = ple.PLECheckpoint(root)
    assert (table.rows, table.width) == (5, 160)
    assert table.rows_per_mapping == (1 << 30) // 320
    assert (1 << 30) % 320 == 64
    out = torch.empty((3, 160), dtype=torch.bfloat16)
    table.copy_rows(out, 2)
    assert torch.equal(out, data[2:])
    assert torch.equal(table.load_rdma()["ple"].view(torch.bfloat16), data)
    # A small stand-in mapping tests the same 64-byte gap without 1 GiB allocations.
    monkeypatch.setattr(ple, "GIB", 1024)
    storage = ctypes.create_string_buffer(table.allocation_bytes)
    address = ctypes.addressof(storage)
    buffer = SimpleNamespace(
        gva=0,
        handle=SimpleNamespace(gva_to_va=lambda offset, _: address + offset),
        runtime=SimpleNamespace(bm=SimpleNamespace(BmMemType=SimpleNamespace(LOCAL_HOST=0))),
    )
    owner = SimpleNamespace(add=lambda key, rows, width, dtype: buffer)
    table.load_bm(owner, SimpleNamespace(is_set=lambda: False))
    assert storage.raw[960:1024] == bytes(64)
    for row in range(5):
        offset = (row // 3) * 1024 + (row % 3) * 320
        assert storage.raw[offset : offset + 320] == data[row].view(torch.uint8).numpy().tobytes()


def test_checkpoint_rejects_multiple_tables_and_missing_shard(checkpoint):
    root, _ = checkpoint
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    item = next(iter(index["weight_map"]))
    del index["weight_map"][item]
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="shard indexes"):
        ple.PLECheckpoint(root)
    index["weight_map"][item.replace("layers.1", "layers.2")] = "part-0.safetensors"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="exactly one checkpoint table"):
        ple.PLECheckpoint(root)


def test_shared_config_fails_closed_and_ignores_auto_dp_sharing(monkeypatch):
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    from vllm_ascend import ascend_config

    shared = SimpleNamespace(backend="a3", endpoint="host:1")
    monkeypatch.setattr(ascend_config, "get_ascend_config", lambda: SimpleNamespace(shared_engram=shared))
    config = SimpleNamespace(
        additional_config={"shared_engram": {"backend": "a3", "endpoint": "host:1"}},
        use_v2_model_runner=True,
        shutdown_timeout=30,
        load_config=SimpleNamespace(load_format="auto"),
        engram_config=SimpleNamespace(cpu_offload=True, dp_shared_memory=True),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(ple_layer_ids=[2])),
    )
    monkeypatch.setattr(DefaultModelLoader, "Source", type("Legacy", (), {"__dataclass_fields__": {}}))
    monkeypatch.setattr(DefaultModelLoader, "_ascend_external_weight_filter", False, raising=False)
    with pytest.raises(RuntimeError, match="per-tensor"):
        ple.shared_ple_config(config)
    monkeypatch.setattr(DefaultModelLoader, "_ascend_external_weight_filter", True)
    assert ple.shared_ple_config(config) is shared
    monkeypatch.setattr(DefaultModelLoader, "_ascend_external_weight_filter", False)
    monkeypatch.setattr(
        DefaultModelLoader,
        "Source",
        type("Filtering", (), {"__dataclass_fields__": {"filter_unused_per_tensor": None}}),
    )
    assert ple.shared_ple_config(config) is shared


def test_registered_owner_closes_after_later_model_failure(monkeypatch):
    monkeypatch.setattr(ple, "get_current_vllm_config", lambda: SimpleNamespace(engram_config=None))
    monkeypatch.setattr(ple, "shared_ple_config", lambda _: SimpleNamespace(backend="a3"))
    monkeypatch.setattr(
        ple.Qwen4ExpPLEEmbedding, "__init__", lambda self, *args, **kwargs: torch.nn.Module.__init__(self)
    )
    client = Mock()
    monkeypatch.setattr(
        ple.AscendPLEHostEmbedding, "_attach_shared", lambda self, *args: setattr(self, "_shared_client", client)
    )
    with track_host_offload_owners() as owners:
        obj = ple.AscendPLEHostEmbedding()
        assert owners == [obj]
        # Runner uses this same tracked cleanup on later model construction failure.
        owners[0].close_host_offload()
    client.close.assert_called_once()


def test_failed_attach_cleanup_uses_tracker_or_standalone_retention(monkeypatch):
    from vllm_ascend.models.deepseek_v41.engram import shared_pool

    retained: list[Callable[[], None]] = []
    monkeypatch.setattr(shared_pool, "_failed_startup_cleanups", retained)
    monkeypatch.setattr(ple, "get_current_vllm_config", lambda: SimpleNamespace(engram_config=None))
    monkeypatch.setattr(ple, "shared_ple_config", lambda _: SimpleNamespace(backend="a3"))
    monkeypatch.setattr(
        ple.Qwen4ExpPLEEmbedding, "__init__", lambda self, *args, **kwargs: torch.nn.Module.__init__(self)
    )
    client = Mock()
    client.close.side_effect = [RuntimeError("close failed"), None, RuntimeError("close failed"), None]

    def fail(self, *args):
        self._shared_client = client
        raise ValueError("attach failed")

    monkeypatch.setattr(ple.AscendPLEHostEmbedding, "_attach_shared", fail)
    with track_host_offload_owners() as owners:
        with pytest.raises(ValueError, match="attach failed"):
            ple.AscendPLEHostEmbedding()
        assert len(owners) == 1 and client.close.call_count == 0 and not retained
        # Model runner retains a failed tracked owner and retries it at shutdown.
        with pytest.raises(RuntimeError, match="close failed"):
            owners[0].close_host_offload()
        owners[0].close_host_offload()
        assert client.close.call_count == 2 and not retained

    with pytest.raises(ValueError, match="attach failed"):
        ple.AscendPLEHostEmbedding()
    assert len(retained) == 1 and client.close.call_count == 3
    retained[0]()
    assert client.close.call_count == 4


def test_bm_global_row_lookup_uses_existing_kernel(monkeypatch):
    from vllm_ascend.ops.triton.qwen4_exp import ple_lookup

    obj = object.__new__(ple.AscendPLEHostEmbedding)
    torch.nn.Module.__init__(obj)
    obj._shared = SimpleNamespace(backend="a3")
    obj._quantized = False
    obj._shared_ptrs = torch.tensor([1, 2])
    obj._shared_checkpoint = SimpleNamespace(rows=12, rows_per_mapping=3)
    obj._closing = False
    obj.embedding_dim = 160
    obj.shard_indices = SimpleNamespace(org_vocab_start_index=4, org_vocab_end_index=16)
    ids = torch.tensor([[4, 8, 12, -1]])
    output = torch.empty((1, 4, 160), dtype=torch.bfloat16)
    lookup = Mock()
    monkeypatch.setattr(ple_lookup, "lookup", lookup, raising=False)
    assert obj._lookup(ids, output) is output
    lookup.assert_called_once_with(obj._shared_ptrs, ids, output, 4, 12, 3, global_rows=True)


def test_external_names_filter_preserves_parent_predicate():
    class Base(torch.nn.Module):
        def is_unused_checkpoint_weight(self, name):
            return name == "parent.weight"

    class Model(ple.ExternalPLEWeightsMixin, Base):
        pass

    model = Model()
    model.embedding = torch.nn.Identity()
    assert not model.filter_unused_checkpoint_weights
    model.embedding.ple_external_checkpoint_names = frozenset({"ple.shard_0.weight"})
    assert model.filter_unused_checkpoint_weights
    assert model.is_unused_checkpoint_weight("ple.shard_0.weight")
    assert model.is_unused_checkpoint_weight("parent.weight")
    assert not model.is_unused_checkpoint_weight("ple.shard_1.weight")


def test_real_loader_skips_mixed_shard_before_materialization(tmp_path, monkeypatch):
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    if "filter_unused_per_tensor" not in DefaultModelLoader.Source.__dataclass_fields__:
        pytest.skip("requires the unmerged vLLM #64395 loader")
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import weight_utils

    class Model(ple.ExternalPLEWeightsMixin, torch.nn.Module):
        pass

    external = "ple.shard_0.weight"
    save_file({external: torch.ones(2), "model.weight": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    model = Model()
    model.embedding = torch.nn.Identity()
    model.embedding.ple_external_checkpoint_names = frozenset({external})
    seen = []
    original = weight_utils.safe_open

    class TrackingOpen:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def keys(self):
            return self.inner.keys()

        def get_tensor(self, name):
            seen.append(name)
            return self.inner.get_tensor(name)

    monkeypatch.setattr(weight_utils, "safe_open", TrackingOpen)
    loader = DefaultModelLoader(LoadConfig(safetensors_load_strategy="lazy"))
    result = dict(loader.get_all_weights(SimpleNamespace(model=str(tmp_path), revision=None), model))
    assert list(result) == ["model.weight"]
    assert seen == ["model.weight"]


def test_remote_model_state_prepares_rows_before_graph(monkeypatch):
    from vllm.models.qwen4_exp.common.ops import ple as ple_ops

    from vllm_ascend.models.qwen4_exp import model_state

    class Remote(ple.AscendPLEHostEmbedding):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.bound = False
            self.prepared = []
            self.cleared = 0

        def bind_preparation(self):
            self.bound = True

        def prepare_rows(self, ids):
            self.prepared.append(ids)

        def clear_prepared_rows(self):
            self.cleared += 1

    monkeypatch.setattr(ple, "shared_ple_config", lambda _: SimpleNamespace(backend="a2"))
    monkeypatch.setattr(model_state, "Qwen4ExpPreparedInputAdapter", Mock())
    monkeypatch.setattr(model_state.Qwen4ExpModelState, "__init__", lambda *args, **kwargs: None)
    inputs = {"query_start_loc": "starts", "ngram_context": "context"}
    monkeypatch.setattr(model_state.Qwen4ExpModelState, "prepare_inputs", lambda *args: inputs)
    monkeypatch.setattr(model_state.Qwen4ExpModelState, "prepare_dummy_inputs", lambda *args: {})
    captured = []

    def make_ids(**kwargs):
        captured.append(kwargs)
        return torch.tensor([[4, 8]])

    monkeypatch.setattr(ple_ops, "ple_ngram_ids", make_ids)
    embedding = Remote()
    layer = torch.nn.Module()
    layer.ngram_embedding = embedding
    layer.layer_multipliers, layer.ngram_heads_vocab_sizes, layer.ngram_heads_offsets = "m", "v", "o"
    layer.eos_token_id, layer.heads_per_ngram = 9, 2
    model = torch.nn.Module()
    model.layer = layer
    state = model_state.AscendQwen4ExpModelState(SimpleNamespace(), model, None, torch.device("cpu"))
    batch = SimpleNamespace(num_reqs=1, input_ids=torch.tensor([[1, 2]]), positions=None, num_tokens_after_padding=1)
    assert state.prepare_inputs(batch, None) is inputs
    assert embedding.bound and torch.equal(embedding.prepared[0], torch.tensor([[4, 8]]))
    assert captured[0]["input_ids"] is batch.input_ids and captured[0]["ngram_context"] == "context"
    state.prepare_inputs(SimpleNamespace(num_reqs=0, positions=None, num_tokens_after_padding=0), None)
    state.prepare_dummy_inputs(1, 1)
    assert embedding.cleared == 2


def test_hixl_owner_uses_common_constructor_with_qwen_tables(checkpoint, monkeypatch):
    from vllm_ascend.models.deepseek_v41.engram import shared_rdma

    root, data = checkpoint
    table = ple.PLECheckpoint(root)
    engine = Mock()
    monkeypatch.setattr(shared_rdma, "HixlEngine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(shared_rdma, "_physical_device_id", lambda device: device)
    owner = shared_rdma.SharedOwner(
        root, "127.0.0.1:0", "127.0.0.1:12345", metadata=table.metadata(), table_loader=table.load_rdma
    )
    try:
        assert owner.metadata == table.metadata()
        assert torch.equal(owner.tables["ple"].view(torch.bfloat16), data)
    finally:
        owner.server_close()
    engine.close.assert_called_once()


@pytest.mark.parametrize("backend", ["a3", "a2"])
def test_existing_owner_cli_selects_qwen_checkpoint(checkpoint, monkeypatch, backend):
    import importlib
    import sys
    import threading
    import types

    from vllm_ascend.models.deepseek_v41.engram import shared_rdma

    root, data = checkpoint
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda _: None), raising=False)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    events = []

    class Event:
        def set(self):
            pass

        def wait(self, *args):
            return True

        def is_set(self):
            return False

    class PoolOwner:
        closed = False
        generation = "g"

        def __init__(self, endpoint, identity, **kwargs):
            assert identity == ple.PLECheckpoint(root).identity
            events.append("construct")

        def ready(self):
            events.append("ready")

        def close(self, timeout):
            self.closed = True
            events.append("close")

    class RDMAOwner:
        generation = "g"
        sessions = {}
        retained = []
        pending_handlers = 0
        lock = threading.Lock()

        def __init__(self, *args, metadata, table_loader, **kwargs):
            assert metadata == ple.PLECheckpoint(root).metadata()
            assert torch.equal(table_loader()["ple"].view(torch.bfloat16), data)
            self.metadata, self.tables = metadata, {"ple": torch.empty(1)}
            events.append("construct")

        def serve_forever(self):
            pass

        def shutdown(self):
            events.append("shutdown")

        def server_close(self):
            events.append("close")

    class Thread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self):
            pass

    if backend == "a3":
        cli = importlib.import_module("vllm_ascend.shared_engram")
        monkeypatch.setattr(cli, "SharedPoolOwner", PoolOwner)
        monkeypatch.setattr(ple.PLECheckpoint, "load_bm", lambda self, owner, stop: events.append("load"))
        extra = []
    else:
        cli = importlib.import_module("vllm_ascend.shared_engram_rdma")
        monkeypatch.setattr(shared_rdma, "SharedOwner", RDMAOwner)
        extra = ["--engine-endpoint", "127.0.0.1:12346"]
    monkeypatch.setattr(cli, "threading", SimpleNamespace(Event=Event, Thread=Thread))
    monkeypatch.setattr(cli, "signal", SimpleNamespace(SIGINT=2, SIGTERM=15, signal=lambda *args: None))
    monkeypatch.setattr(
        sys,
        "argv",
        ["shared_engram", "--model", str(root), "--backend", backend, "--endpoint", "127.0.0.1:12345", *extra],
    )
    cli.main()
    assert events == (
        ["construct", "load", "ready", "close"] if backend == "a3" else ["construct", "shutdown", "close"]
    )


def test_remote_preparation_rejects_capture_before_fence_or_transfer(monkeypatch):
    obj = object.__new__(ple.AscendPLEHostEmbedding)
    torch.nn.Module.__init__(obj)
    obj._preparation_bound, obj._closing = True, False
    obj._shared_client = SimpleNamespace(failed=False)
    obj._shared_checkpoint = SimpleNamespace(rows=12)
    obj.prepared_rows = torch.zeros((2, 2, 160), dtype=torch.bfloat16)
    obj.shard_indices = SimpleNamespace(org_vocab_start_index=4, org_vocab_end_index=14)
    calls = []
    npu = SimpleNamespace(is_current_stream_capturing=lambda: True, synchronize=lambda _: calls.append("fence"))
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(ple, "_fetch_ple_rows", lambda *args: calls.append("transfer"))
    ids = torch.tensor([[4, 11]])
    with pytest.raises(RuntimeError, match="outside graph capture"):
        obj.prepare_rows(ids)
    assert not obj._shared_client.failed and calls == []
    npu.is_current_stream_capturing = lambda: False
    obj.prepare_rows(ids)
    assert calls == ["fence", "transfer"]


def test_hixl_reader_attaches_with_qwen_metadata(checkpoint, monkeypatch):
    import socket
    import threading

    from vllm_ascend.models.deepseek_v41.engram import shared_rdma

    table = ple.PLECheckpoint(checkpoint[0])
    client, server = socket.socketpair()
    monkeypatch.setattr(shared_rdma.socket, "create_connection", lambda *args, **kwargs: client)
    monkeypatch.setattr(shared_rdma, "_configure_socket", lambda _: None)
    monkeypatch.setattr(shared_rdma, "_physical_device_id", lambda _: 7)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False)
    received = []

    def server_thread():
        try:
            received.append(shared_rdma._receive(server))
            shared_rdma._send(server, {"error": "stop after attach"})
        finally:
            server.close()

    thread = threading.Thread(target=server_thread)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="stop after attach"):
            shared_rdma.RemoteReader("unused", "owner:123", capacity=5, metadata=table.metadata())
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert received[0]["metadata"] == table.metadata()
    assert received[0]["physical_device"] == 7


def test_shared_quantized_ple_rejected_before_registering_owner(monkeypatch):
    from vllm_ascend.models.qwen4_exp.ple_quant import AscendPLEQuantizedEmbeddingMethod

    monkeypatch.setattr(ple, "get_etp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(ple, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(ple, "get_current_vllm_config", lambda: SimpleNamespace())
    monkeypatch.setattr(ple, "shared_ple_config", lambda _: SimpleNamespace(backend="a3"))
    register = Mock()
    monkeypatch.setattr(ple, "register_host_offload_owner", register)
    with pytest.raises(ValueError, match="Shared PLE requires BF16"):
        ple.AscendPLEHostEmbedding(
            embedding_method=AscendPLEQuantizedEmbeddingMethod(mxfp8=False), params_dtype=torch.bfloat16
        )
    register.assert_not_called()
