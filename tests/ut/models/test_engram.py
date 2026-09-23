# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram unit tests: hashing, sharded lookup, host offload and graph inputs."""

import ctypes
import importlib.util
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]


def load_module(module_name, path):
    """Load one module by path, without importing the whole vllm_ascend package."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ENGRAM = ROOT / "vllm_ascend/models/deepseek_v41/engram"
npu = load_module("engram_npu", ENGRAM / "npu.py")
_missing_upstream = ""
try:
    common = load_module("engram_common", ENGRAM / "common.py")
except ImportError as error:  # the hashing itself comes from upstream's V4.1 module
    common = None
    _missing_upstream = str(error)

requires_upstream_hash = pytest.mark.skipif(
    common is None, reason=f"engram hashing needs vLLM with DeepSeek V4.1: {_missing_upstream}"
)


def test_engram_config_patch_keeps_the_checkpoint_contract(monkeypatch):
    """Upstream rejects non-CUDA platforms; Ascend hashes the same checkpoint."""

    class EngramConfig:
        dp_shared_memory = False
        embedding_across_dp = False

        def verify_model_config(self, model_config):
            raise AssertionError("Engram config patch was not applied")

    fake = ModuleType("vllm.config.engram")
    monkeypatch.setattr(fake, "EngramConfig", EngramConfig, raising=False)
    fake.__spec__ = importlib.util.spec_from_loader("vllm.config.engram", loader=None)
    monkeypatch.setitem(sys.modules, "vllm.config.engram", fake)
    load_module("engram_config_patch", ROOT / "vllm_ascend/patch/platform/patch_engram_config.py")

    v41 = SimpleNamespace(hf_text_config=SimpleNamespace(engram_layer_ids=[1, 14]))
    EngramConfig().verify_model_config(v41)  # accepted, unlike upstream on Ascend
    with pytest.raises(ValueError):
        EngramConfig().verify_model_config(SimpleNamespace(hf_text_config=SimpleNamespace(engram_layer_ids=[])))
    dp_sharded = EngramConfig()
    dp_sharded.dp_shared_memory = True
    with pytest.raises(ValueError):
        dp_sharded.verify_model_config(v41)


def _npu_available() -> bool:
    try:
        subprocess.run(["npu-smi", "info"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


requires_npu = pytest.mark.skipif(not _npu_available(), reason="host registration needs an NPU")


def _reference_rows(codes, scales, width):
    """What the group-32 INT8 rows decode to, computed on CPU."""
    groups = scales.shape[-1]
    return (codes.float().reshape(-1, groups, width // groups) * scales[..., None]).reshape(-1, width).bfloat16()


def _history():
    h = common.PagedNgramHistory.__new__(common.PagedNgramHistory)
    h.token_map = torch.arange(100)
    h.pad_id = 2
    h.image_token_id = 99
    h.image_pad_token_id = 98
    h.lookback = 2
    h.n_hash_cols = 2
    h.primes = torch.tensor([[[101, 103]]])
    h.offsets = torch.tensor([[0, 101]])
    h.multipliers = torch.tensor([[3, 5]])
    h.pages = {}
    return h


@requires_upstream_hash
@pytest.mark.parametrize("barrier_token", [98, 99])
def test_hash_stops_at_image_and_unwritten_pages(barrier_token):
    """A masked image token and a page this replica never wrote both end the n-gram."""
    masked = _history()
    values, mask = masked.update(
        torch.tensor([0, 5, 9, barrier_token, 13, 17]),
        torch.arange(6),
        torch.zeros(6, dtype=torch.long),
        torch.tensor([[5, 1]]),
        4,
    )
    # The first token on the next page must hash against padding, not the image.
    assert not mask[3]
    assert values[4, 0, 0].item() == ((13 * 3) ^ (masked.pad_id * 5)) % 101

    # Position 4 owns page 11; its look-back reaches page 10, never written here.
    # 4 tokens use the row path, 32 the slab path.
    from_prefix = _history()
    values, mask = from_prefix.update(
        torch.arange(32) % 50,
        torch.arange(4, 36),
        torch.zeros(32, dtype=torch.long),
        torch.arange(10, 26).reshape(1, -1),
        4,
    )
    assert values.shape == (32, 1, 2) and mask.all()


@requires_upstream_hash
def test_gate_preserves_masked_rows():
    torch.manual_seed(7)
    hidden = torch.randn(3, 4, 32).bfloat16()
    out = common.engram_gate(
        hidden,
        torch.randn(3, 4, 32).bfloat16(),
        torch.randn(3, 32).bfloat16(),
        torch.randn(4, 32),
        torch.eye(32),
        torch.tensor([True, False, True]),
        1e-5,
    )
    assert torch.equal(out[1], hidden[1]) and torch.isfinite(out.float()).all()


def _lookup_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    tp = None
    for ranks in ([0, 1], [2, 3]):
        group = dist.new_group(ranks)
        if rank in ranks:
            tp = group
    q = npu.EngramQueryGroup(dist.group.WORLD, dist.group.WORLD, tp, (rank // 2) * 2)
    width, groups = 64, 2
    codes = (torch.arange(41 * width).reshape(41, width) % 251 - 125).to(torch.int8)
    scales = torch.linspace(0.25, 1.0, 41 * groups).reshape(41, groups)
    table = npu.NodeShardedEngram(41, width, q, device="cpu")
    table.weight.data.copy_(codes[table.start : table.end])
    table.weight_scale.copy_(scales[table.start : table.end])
    reference = _reference_rows(codes, scales, width)
    # Row counts that do not divide the shard size, idle replicas, and ids that
    # cross a shard boundary all restore requester order bit for bit.
    for a, b in ((0, 0), (1, 0), (0, 17), (3, 7), (31, 1), (1, 1), (0, 0)):
        count = (a, b)[rank // 2]
        ids = (torch.arange(count * 3).reshape(count, 3) * 7 + rank // 2) % 41
        for result in (table(ids), table.forward_many([ids])[0], table.route_many([table], [ids])[0]):
            assert torch.equal(result.view(torch.int16), reference[ids].view(torch.int16)), (rank, a, b)
    dist.destroy_process_group()


def test_cross_dp_lookup_restores_order(tmp_path):
    mp.spawn(_lookup_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)


@pytest.mark.parametrize("quantized", [True, False])
def test_loader_quantizes_bf16_source_and_prefers_int8(tmp_path, quantized):
    key, scale_key = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    weights = torch.linspace(-12, 12, 19 * 64).reshape(19, 64).bfloat16()
    expected_codes, expected_scales = npu.quantize_engram_rows(weights)
    save_file({key: weights}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "model.safetensors"}}))
    if quantized:
        save_file({key: expected_codes, scale_key: expected_scales}, tmp_path / "quant_model_weights.safetensors")
        (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: "quant_model_weights.safetensors" for name in (key, scale_key)}})
        )
    table = npu.NodeShardedEngram(19, 64, SimpleNamespace(size=1, rank=0), device="cpu")
    table.load_checkpoint(tmp_path, key, chunk_rows=4)
    assert torch.equal(table.weight, expected_codes)
    assert torch.equal(table.weight_scale, expected_scales)


def _fake_host_library(host_address, device_offset):
    """Enough of libascendcl.so to publish a test buffer without a device."""

    class FakeHostLibrary:
        def aclrtMallocHost(self, out, size, flags):
            out._obj.value = host_address
            return 0

        def aclrtHostGetDevicePointer(self, pointer, out, flags):
            out._obj.value = pointer.value + device_offset
            return 0

        def aclrtHostRegisterV2(self, pointer, size, flags):
            return 0

        def aclrtHostUnregister(self, pointer):
            return 0

        def aclrtFreeHost(self, pointer):
            return 0

    return FakeHostLibrary()


def test_host_uva_address_table_steps_by_chunk(monkeypatch):
    """Rows past the first chunk need their own 64 bit device address."""
    rows, width, chunk = 20, 8, 8
    backing = ctypes.create_string_buffer(rows * width)
    host_address, device_offset = ctypes.addressof(backing), 1 << 40
    monkeypatch.setattr(npu, "CHUNK_ROWS", chunk)
    monkeypatch.setattr(npu, "_host_library", lambda: _fake_host_library(host_address, device_offset))
    table = npu.HostUvaBuffer((rows, width), torch.int8, device="cpu")
    assert table.tensor.shape == (rows, width)
    assert table.ptrs.tolist() == [device_offset + host_address + start * width for start in range(0, rows, chunk)]
    table.close()
    assert isinstance(table.pointer, ctypes.c_void_p)
    assert table.pointer.value is None


@requires_npu
def test_host_offloaded_shard_serves_the_lookup(monkeypatch):
    monkeypatch.setattr(npu, "CHUNK_ROWS", 8)
    torch.npu.set_device(0)
    rows, width, groups = 20, 64, 2
    table = npu.NodeShardedEngram(rows, width, SimpleNamespace(size=1, rank=0), cpu_offload=True)
    codes = (torch.arange(rows * width).reshape(rows, width) % 251 - 125).to(torch.int8)
    scales = torch.linspace(0.25, 1.0, rows * groups).reshape(rows, groups)
    table.weight.data.copy_(codes)
    table.weight_scale.copy_(scales)
    # 7/8 and 15/16 straddle the chunk boundary; 19 is the last row.
    ids = torch.tensor([0, 7, 8, 15, 16, 19, 0])
    actual = table.lookup_local(ids)
    expected = _reference_rows(codes, scales, width)[ids]
    assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
    assert table.weight.device.type == "cpu" and table.weight_scale.device.type == "cpu"


@pytest.fixture
def engram_model(monkeypatch):
    try:
        from vllm_ascend.models.deepseek_v41 import model as implementation
    except ImportError as error:  # the V4.1 model needs a newer vLLM
        pytest.skip(f"DeepSeek V4.1 is not importable here: {error}")
    monkeypatch.setattr(implementation, "engram_enabled", lambda config: True)
    cls = implementation.DeepseekV41Model
    shell = SimpleNamespace(
        config=SimpleNamespace(engram_layer_ids=[1, 14], engram_max_ngram_size=4, engram_n_heads=8),
        layers=[SimpleNamespace(engram=SimpleNamespace(embed=SimpleNamespace(width=32))) for _ in range(15)],
        engram_rotation=torch.eye(32),
        _engram_max_tokens=16,
        _engram_input_buffers=None,
    )
    shell.prepare_engram_graph_inputs = cls.prepare_engram_graph_inputs.__get__(shell)
    shell.prepare_engram_inputs = cls.prepare_engram_inputs.__get__(shell)
    shell.engram_module = implementation
    return shell


def test_graph_buffers_are_reused_and_refreshed(engram_model):
    captured = engram_model.prepare_engram_graph_inputs(4)
    pointers = {layer: value.data_ptr() for layer, value in captured["engram_lookups"].items()}
    for count, padded in ((9, 12), (1, 4), (0, 4)):
        values = {layer: torch.full((count, 768), float(count), dtype=torch.bfloat16) for layer in (1, 14)}
        engram_model.prepare_engram = lambda *args, values=values, count=count: (
            values,
            torch.ones(count, dtype=torch.bool),
        )
        # A small decode must refresh its rows without clearing the capacity.
        for buffer in captured["engram_lookups"].values():
            buffer[padded:].fill_(123)
        actual = engram_model.prepare_engram_inputs(torch.arange(count), torch.arange(count), padded)
        assert actual["engram_mask"][:count].all() and not actual["engram_mask"][count:padded].any()
        for layer, buffer in actual["engram_lookups"].items():
            assert buffer.shape == (16, 768) and buffer.data_ptr() == pointers[layer]
            assert torch.equal(buffer[:count], values[layer]) and not buffer[count:padded].any()
            assert (buffer[padded:] == 123).all()


def test_disabled_engram_capture_skips_layers(engram_model, monkeypatch):
    monkeypatch.setattr(engram_model.engram_module, "engram_enabled", lambda config: False)
    engram_model.layers = [SimpleNamespace(engram=None) for _ in range(15)]
    for result in (
        engram_model.prepare_engram_graph_inputs(4),
        engram_model.prepare_engram_inputs(None, torch.arange(4), 4),
    ):
        assert result["engram_lookups"] == {} and result["engram_mask"].numel() == 0


@pytest.mark.parametrize("num_reqs", [0, 2])
def test_runner_history_inputs_use_full_swa_requests(monkeypatch, num_reqs):
    from vllm_ascend.worker import model_runner_v1 as runner_module

    pages = torch.tensor([[7, 8, 9], [12, 13, 14], [99, 99, 99]], dtype=torch.int32)
    boundaries = torch.tensor([0, 3, 9, 99], dtype=torch.int32)
    groups = [
        SimpleNamespace(layer_names=["long_kv"], kv_cache_spec=object()),
        SimpleNamespace(layer_names=["swa"], kv_cache_spec=object()),
    ]
    monkeypatch.setattr(runner_module, "get_forward_context", lambda: SimpleNamespace(attn_metadata={}))
    monkeypatch.setattr(
        runner_module, "get_storage_block_size", lambda spec: 4 if spec is groups[1].kv_cache_spec else 8
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(engram_cache_layer_name="swa"),
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        query_start_loc=SimpleNamespace(cpu=boundaries),
        input_batch=SimpleNamespace(
            num_reqs=num_reqs,
            block_table=[None, SimpleNamespace(get_cpu_tensor=lambda: pages)],
        ),
    )
    actual_boundaries, actual_pages, block_size = runner_module.NPUModelRunner._get_engram_history_inputs(runner)
    torch.testing.assert_close(actual_boundaries, boundaries[: num_reqs + 1])
    torch.testing.assert_close(actual_pages, pages[:num_reqs])
    assert actual_boundaries.data_ptr() == boundaries.data_ptr()
    if num_reqs:
        assert actual_pages.data_ptr() == pages.data_ptr()
    assert block_size == 4


def test_runner_history_without_attention_metadata_routes_empty_inputs(monkeypatch):
    from unittest.mock import Mock

    from vllm_ascend.worker import model_runner_v1 as runner_module

    model = Mock(return_value=42)
    model.prepare_engram_inputs.return_value = {}
    model.engram_cache_layer_name = "swa"
    runner = SimpleNamespace(model=model, enable_enpu=False, _update_full_graph_params_if_needed=lambda *args: None)
    runner._get_engram_history_inputs = runner_module.NPUModelRunner._get_engram_history_inputs.__get__(runner)
    monkeypatch.setattr(
        runner_module,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=runner_module.CUDAGraphMode.NONE, attn_metadata=None),
    )
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    assert runner_module.NPUModelRunner._model_forward(runner, 4) == 42
    model.prepare_engram_inputs.assert_called_once_with(None, None, 4, None)
