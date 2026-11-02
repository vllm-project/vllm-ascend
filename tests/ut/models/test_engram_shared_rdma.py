# SPDX-License-Identifier: Apache-2.0
"""Protocol and packed-row checks, including the real native packer when built."""

import ctypes
import importlib.util
import json
import os
import socket
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def rdma():
    package_spec = importlib.util.find_spec("vllm_ascend")
    assert package_spec is not None and package_spec.origin is not None
    package = Path(package_spec.origin).parent
    source = Path(os.getenv("ENGRAM_TEST_SOURCE", package / "models/deepseek_v41/engram/shared_rdma.py"))
    spec = importlib.util.spec_from_file_location("engram_shared_rdma_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_round_trip_and_closed_connection(rdma):
    left, right = socket.socketpair()
    try:
        rdma._send(left, {"generation": "g", "count": 17})
        assert rdma._receive(right) == {"generation": "g", "count": 17}
        left.close()
        with pytest.raises(ConnectionError):
            rdma._receive(right)
    finally:
        left.close()
        right.close()


def test_oversized_control_is_rejected_before_payload(rdma):
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!I", rdma.MAX_CONTROL_BYTES + 1))
        with pytest.raises(ValueError, match="message size"):
            rdma._receive(right)
    finally:
        left.close()
        right.close()


def test_same_nic_reader_uses_another_owner_device(rdma, monkeypatch):
    owner = object.__new__(rdma.SharedOwner)
    owner.server_address = ("10.0.0.1", 29500)
    owner.device = 0
    owner.engine = object()
    owner.engine_endpoint = "10.0.0.1:29501"
    owner.library_path = None
    owner.alternate_engine = owner.alternate_endpoint = None
    owner.lock = threading.Lock()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device_count=lambda: 2), raising=False)
    created = []

    def engine(endpoint, device, library_path):
        created.append((endpoint, device))
        return object()

    monkeypatch.setattr(rdma, "HixlEngine", engine)
    assert owner.engine_for_reader("10.0.0.2", 0) == (owner.engine, owner.engine_endpoint)
    assert owner.engine_for_reader("10.0.0.1", 1) == (owner.engine, owner.engine_endpoint)
    alternate = owner.engine_for_reader("10.0.0.1", 0)
    assert alternate[1] == "10.0.0.1:29502"
    assert owner.engine_for_reader("10.0.0.1", 0) == alternate
    assert created == [("10.0.0.1:29502", 1)]


def test_both_layers_keep_token_and_head_order_in_one_copy(rdma):
    hashes = torch.tensor(
        [[[0, 10, -1, 13], [20, 21, 22, 23]], [[30, 31, 32, 33], [40, -1, 42, 43]]], dtype=torch.int32
    )
    copied = rdma.RemoteReader.copy_hashes(hashes, 1, 2)
    assert copied.dtype == np.int64
    np.testing.assert_array_equal(copied[:, 0], [[10, -1], [31, 32]])
    np.testing.assert_array_equal(copied[:, 1], [[21, 22], [-1, 42]])


def test_batch_reader_falls_back_without_touching_transport(rdma):
    reader = object.__new__(rdma.RemoteReader)
    reader.supports_batch = False
    tables = [SimpleNamespace(part_n_hash_cols=6)] * 2
    hashes = torch.empty((4, 2, 24), dtype=torch.int64)
    assert reader.lookup_layers(tables, hashes, None) is None
    reader.supports_batch = True
    reader.capacity = 47
    assert reader.lookup_layers(tables, hashes, None) is None


def test_close_fences_only_owned_buffer_before_retiring_lease(rdma, monkeypatch):
    order = []
    reader = object.__new__(rdma.RemoteReader)
    reader.closed = False
    reader._last_use = SimpleNamespace(synchronize=lambda: order.append("last_use"))
    reader.engine = SimpleNamespace(close=lambda: order.append("engine"))
    reader.wire = SimpleNamespace(close=lambda: order.append("wire"))
    reader.generation = "generation"
    reader.buffer = object()

    def no_device_fence():
        raise AssertionError("unrelated device work must not gate lease retirement")

    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=no_device_fence), raising=False)
    monkeypatch.setattr(rdma, "_send", lambda *args: order.append("owner_close"))
    monkeypatch.setattr(rdma, "_receive", lambda *args: {"closed": True})
    reader.close()
    reader.close()
    assert order == ["last_use", "engine", "owner_close", "wire"]
    assert reader.closed and reader.buffer is None


@pytest.mark.parametrize(
    "lookups",
    [
        [],
        [{"layer": "missing", "count": 1}],
        [{"layer": "1", "count": True}],
        [{"layer": "1", "count": 2}, {"layer": "1", "count": 1}],
        [{"layer": "1", "count": 4}, {"layer": "14", "count": 3}],
    ],
)
def test_batch_plan_rejects_invalid_layout_before_ids(rdma, lookups):
    tables = {"1": torch.empty((4, 264), dtype=torch.uint8), "14": torch.empty((5, 288), dtype=torch.uint8)}
    with pytest.raises(ValueError):
        rdma._lookup_plan(dict(op="lookup_batch", lookups=lookups), tables, 6)


def test_batch_native_packing_keeps_layer_offsets_and_rejects_partial_writes(rdma):
    path = os.getenv("ENGRAM_TEST_LIBRARY")
    if not path:
        pytest.skip("requires the actual native row packer")
    lib = ctypes.CDLL(path)
    lib.engram_pack_rows.argtypes = [ctypes.c_size_t] * 3 + [ctypes.c_void_p] + [ctypes.c_size_t] * 2

    def check(code):
        assert code == 0

    engine = SimpleNamespace(lib=lib, check=check)
    tables = {
        "1": torch.arange(4 * 264).to(torch.uint8).reshape(4, 264),
        "14": torch.arange(5 * 288).to(torch.uint8).reshape(5, 288),
    }
    request = dict(op="lookup_batch", lookups=[dict(layer="1", count=3), dict(layer="14", count=2)])
    plan, count = rdma._lookup_plan(request, tables, 6)
    assert count == 5 and plan == [("1", 3, 0, 0), ("14", 2, 3, 792)]
    stage = torch.full((6 * 288,), 33, dtype=torch.uint8)
    ids = np.array([3, -1, 0, 4, 1], dtype=np.int64)
    rdma._pack_lookup_rows(engine, stage, tables, plan, ids)
    expected = torch.cat(
        [tables["1"][3], torch.zeros(264, dtype=torch.uint8), tables["1"][0], tables["14"][4], tables["14"][1]]
    )
    torch.testing.assert_close(stage[: expected.numel()], expected)
    assert torch.all(stage[expected.numel() :] == 33)
    before = stage.clone()
    ids[-1] = 5
    with pytest.raises(ValueError, match="row ID"):
        rdma._pack_lookup_rows(engine, stage, tables, plan, ids)
    torch.testing.assert_close(stage, before)


@pytest.mark.skipif(not Path("/dev/davinci_manager").exists(), reason="requires the installed NPU worker runtime")
def test_worker_retires_reader_before_network_shutdown(monkeypatch):
    from vllm_ascend.worker import worker as worker_module

    order = []
    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(close_shared_engram=lambda: order.append("reader")),
        shutdown=lambda: order.append("runner"),
    )
    worker.profiler = None
    worker.weight_transfer_engine = None
    monkeypatch.setattr(worker_module, "ensure_kv_transfer_shutdown", lambda: order.append("kv"))
    worker.shutdown()
    assert order == ["reader", "kv", "runner"]


@pytest.mark.skipif(not Path("/dev/davinci_manager").exists(), reason="requires NPU kernels")
@pytest.mark.parametrize("scale_dtype,row_bytes", [("F32", 288), ("F8_E8M0", 264)])
def test_packed_dequant_preserves_signed_codes_and_scales(scale_dtype, row_bytes):
    import torch_npu  # noqa: F401

    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.ops.triton.engram_lookup import dequantize_packed_engram

    torch.npu.set_device(1 if torch.npu.device_count() > 1 else 0)
    codes = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).repeat(3, 1)
    scales = torch.tensor([[0.125, 0.25, 0.5, 1, 2, 4, 8, 16]], dtype=torch.float32).repeat(3, 1)
    raw = (
        scales.view(torch.uint8)
        if scale_dtype == "F32"
        else torch.tensor([[124, 125, 126, 127, 128, 129, 130, 131]], dtype=torch.uint8).repeat(3, 1)
    )
    packed = torch.empty((3, row_bytes), dtype=torch.uint8)
    packed[:, :256].copy_(codes.view(torch.uint8))
    packed[:, 256:].copy_(raw)
    actual = torch.empty((3, 256), dtype=torch.bfloat16, device="npu")
    dequantize_packed_engram(packed.to("npu"), actual, 3, row_bytes, scale_dtype)
    torch.npu.synchronize()
    expected = (codes.float().unflatten(-1, (8, 32)) * scales.unsqueeze(-1)).flatten(-2).bfloat16()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_checkpoint_samples_detect_same_metadata_changed_weights(rdma, tmp_path):
    config = {"text_config": {"engram_head_dim": 256, "engram_layer_ids": [1], "engram_num_embeddings": [4]}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    weight_key, scale_key = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    index = {"weight_map": {weight_key: "weights.safetensors", scale_key: "weights.safetensors"}}
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(json.dumps(index))
    weights = torch.arange(4 * 256).reshape(4, 256).to(torch.int8)
    scales = torch.ones((4, 8), dtype=torch.float32)
    save_file({weight_key: weights, scale_key: scales}, tmp_path / "weights.safetensors")
    # Worker initialization supplies a non-CPU default device. Sampling must
    # not materialize the table there before selecting the requested rows.
    with torch.device("meta"):
        first = rdma.checkpoint_metadata(tmp_path)
    weights[0, 0] += 1
    save_file({weight_key: weights, scale_key: scales}, tmp_path / "weights.safetensors")
    second = rdma.checkpoint_metadata(tmp_path)
    assert first["identity"] == second["identity"]
    assert first["tables"]["1"]["samples"] != second["tables"]["1"]["samples"]
    assert second["tables"]["1"]["row_bytes"] == 288


@pytest.mark.parametrize("width", [264, 288])
def test_native_packer_mask_order_and_no_partial_write(width):
    path = os.getenv("ENGRAM_TEST_LIBRARY")
    if not path:
        pytest.skip("build the CANN native library to test its actual packer")
    lib = ctypes.CDLL(path)
    lib.engram_pack_rows.argtypes = [ctypes.c_size_t] * 3 + [ctypes.c_void_p] + [ctypes.c_size_t] * 2
    source = np.arange(4 * width, dtype=np.uint8).reshape(4, width)
    output = np.full((4, width), 33, dtype=np.uint8)
    ids = np.array([3, -1, 0, 3], dtype=np.int64)
    args = (output.ctypes.data, source.ctypes.data, width, ids.ctypes.data, len(ids), len(source))
    assert lib.engram_pack_rows(*args) == 0
    np.testing.assert_array_equal(output, np.stack([source[3], np.zeros(width, dtype=np.uint8), source[0], source[3]]))
    before = output.copy()
    ids[-1] = 4
    assert lib.engram_pack_rows(*args) != 0
    np.testing.assert_array_equal(output, before)
