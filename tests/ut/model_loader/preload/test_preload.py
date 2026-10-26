# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
import weakref
from collections import OrderedDict
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.weight_cache.protocol import TensorEntry, WeightCacheState

from vllm_ascend.model_loader.preload import config, transport
from vllm_ascend.model_loader.preload.loader import AscendIpcModelLoader


def _config():
    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            quantization=None,
            hf_config=SimpleNamespace(),
            is_moe=False,
            use_mla=False,
            is_multimodal_model=False,
            runner_type="generate",
            enable_sleep_mode=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        speculative_config=None,
        lora_config=None,
        cache_config=SimpleNamespace(),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0), prefetch=SimpleNamespace(offload_group_size=0)
        ),
    )


@pytest.fixture
def valid_config(monkeypatch):
    monkeypatch.setattr(config, "init_ascend_config", lambda _: SimpleNamespace(weight_nz_mode=0))
    monkeypatch.setattr(config, "get_current_hardware_profile", lambda: SimpleNamespace(weight_layout_policy=None))
    return _config()


def test_supported_dense_config_retains_tp_placement(valid_config):
    valid_config.parallel_config.tensor_parallel_size = 2
    config.validate_preload_config(valid_config)


@pytest.mark.parametrize(
    "owner,field,value,message",
    [
        ("root", "use_v2_model_runner", False, "V2"),
        ("model_config", "dtype", torch.float32, "BF16"),
        ("model_config", "quantization", "ascend", "quantized"),
        ("model_config", "is_moe", True, "Dense"),
        ("model_config", "use_mla", True, "Dense"),
        ("model_config", "is_multimodal_model", True, "Dense"),
        ("model_config", "is_hybrid", True, "hybrid"),
        ("model_config", "is_attention_free", True, "attention-free"),
        ("model_config", "runner_type", "pooling", "generate"),
        ("model_config", "enable_sleep_mode", True, "sleep"),
        ("root", "speculative_config", object(), "speculative"),
        ("root", "lora_config", object(), "LoRA"),
        ("root", "weight_transfer_config", object(), "weight transfer"),
        ("root", "kv_transfer_config", object(), "cache transfer"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP"),
        ("parallel_config", "data_parallel_size", 2, "DP"),
        ("parallel_config", "enable_expert_parallel", True, "EP"),
        ("parallel_config", "prefill_context_parallel_size", 2, "context"),
    ],
)
def test_invalid_config_fails_before_loading(valid_config, owner, field, value, message):
    setattr(valid_config if owner == "root" else getattr(valid_config, owner), field, value)
    with pytest.raises(ValueError, match=message):
        config.validate_preload_config(valid_config)


def test_nz_config_requires_explicit_nd(valid_config, monkeypatch):
    monkeypatch.setattr(config, "init_ascend_config", lambda _: SimpleNamespace(weight_nz_mode=2))
    with pytest.raises(ValueError, match="weight_nz_mode"):
        config.validate_preload_config(valid_config)


def test_nonstandard_weight_sharding_is_rejected(valid_config, monkeypatch):
    ascend = SimpleNamespace(weight_nz_mode=0, finegrained_tp_config=SimpleNamespace(mlp_tensor_parallel_size=2))
    monkeypatch.setattr(config, "init_ascend_config", lambda _: ascend)
    with pytest.raises(ValueError, match="fine-grained"):
        config.validate_preload_config(valid_config)


def test_cpu_offload_rejected(valid_config):
    valid_config.offload_config.uva.cpu_offload_gb = 1
    with pytest.raises(ValueError, match="offloading"):
        config.validate_preload_config(valid_config)


def test_cpu_entry_keeps_upstream_behavior():
    tensor = torch.arange(8)
    entry = transport.NPUTensorEntry.from_tensor(tensor, "buffer")
    assert type(entry) is TensorEntry
    torch.testing.assert_close(entry.rebuild(0), tensor)


def test_npu_entry_retargets_device_without_changing_original_args(monkeypatch):
    module = ModuleType("torch_npu.multiprocessing.reductions")
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
    module.rebuild_npu_tensor = Mock(return_value=tensor)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(transport.torch_npu, "get_npu_format", lambda _: transport.ND_FORMAT, raising=False)
    args = tuple(range(15))
    entry = transport.NPUTensorEntry("param", ipc_args=args)
    assert entry.rebuild(3) is tensor
    received = module.rebuild_npu_tensor.call_args.args
    assert received[transport.IPC_DEVICE_INDEX] == 3
    assert entry.ipc_args == args
    assert received[:6] == args[:6] and received[7:] == args[7:]


def test_nz_export_is_rejected_before_reduction(monkeypatch):
    reducer = Mock()
    monkeypatch.setattr(torch.multiprocessing.reductions, "reduce_tensor", reducer)
    monkeypatch.setattr(transport.torch_npu, "get_npu_format", lambda _: 29, raising=False)
    tensor = Mock(device=SimpleNamespace(type="npu"))
    tensor.detach.return_value = tensor
    with pytest.raises(ValueError, match="NZ IPC"):
        transport.NPUTensorEntry.from_tensor(tensor, "param")
    reducer.assert_not_called()


def test_external_bytes_deduplicates_storage_and_excludes_cpu():
    storage = SimpleNamespace(data_ptr=lambda: 123, nbytes=lambda: 4096)
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu"), untyped_storage=lambda: storage)
    cpu = torch.ones(100)
    model = SimpleNamespace(parameters=lambda: iter([tensor, tensor]), buffers=lambda: iter([tensor, cpu]))
    assert transport.exported_npu_weight_bytes(model) == 4096
    assert transport.exported_npu_weight_bytes(None) == 0


def _loader(mode="zero_copy"):
    return AscendIpcModelLoader(LoadConfig(load_format="ipc_cache", model_loader_extra_config={"mode": mode}))


@pytest.mark.parametrize("mode", ["zero_copy", "copy"])
def test_binding_preserves_tied_parameter_identity(mode):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(4, device="meta"))
    model.tied = model.weight
    model.register_buffer("cache", torch.empty(2, device="meta"), persistent=False)
    source = torch.arange(4, dtype=torch.float32)
    state = WeightCacheState(
        entries={
            "weight": TensorEntry.from_tensor(source, "param"),
            "cache": TensorEntry.from_tensor(source[:2], "buffer"),
        },
        aliases={"tied": "weight"},
        attrs={},
    )
    _loader(mode)._apply_entries(model, state, 0)
    assert model.weight is model.tied
    torch.testing.assert_close(model.weight, source)
    assert (model.weight.data_ptr() == source.data_ptr()) == (mode == "zero_copy")


def test_missing_weight_is_not_materialized_empty():
    model = torch.nn.Linear(4, 4, device="meta")
    with pytest.raises(ValueError, match="missing"):
        _loader()._apply_entries(model, WeightCacheState({}, {}, {}), 0)
    assert model.weight.is_meta


def test_wrong_shape_is_rejected():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(4, device="meta"))
    state = WeightCacheState({"weight": TensorEntry.from_tensor(torch.ones(8), "param")}, {}, {})
    with pytest.raises(ValueError, match="shape or dtype"):
        _loader()._apply_entries(model, state, 0)


def test_zero_copy_reload_does_not_write_shared_weights():
    with pytest.raises(ValueError, match="daemon-owned"):
        _loader().load_weights(torch.nn.Module(), Mock())


def test_patch_is_idempotent_and_updates_cli_binding():
    from vllm.entrypoints.cli import preload
    from vllm.model_executor.model_loader.weight_cache import daemon

    from vllm_ascend.patch.platform.patch_preload import apply_preload_patch, check_ipc_platform_support

    apply_preload_patch()
    original = daemon._run_daemon_original
    apply_preload_patch()
    assert daemon._run_daemon_original is original
    assert preload.check_ipc_platform_support is check_ipc_platform_support


def test_nd_export_uses_torch_npu_patched_reducer(monkeypatch):
    tensor = Mock(device=SimpleNamespace(type="npu"))
    tensor.detach.return_value = tensor
    reducer = Mock(return_value=(object(), tuple(range(15))))
    monkeypatch.setattr(torch.multiprocessing.reductions, "reduce_tensor", reducer)
    monkeypatch.setattr(transport.torch_npu, "get_npu_format", lambda _: transport.ND_FORMAT, raising=False)
    entry = transport.NPUTensorEntry.from_tensor(tensor, "param")
    assert entry.ipc_args == tuple(range(15))
    reducer.assert_called_once_with(tensor)


def test_failed_constructor_restores_rope_and_attention_registry(monkeypatch):
    from vllm.model_executor.layers import rotary_embedding as upstream_rope

    from vllm_ascend import ops
    from vllm_ascend.model_loader.preload.loader import _model_construction_state

    fields = ("_cos_sin_cache", "_cos_cache", "_sin_cache", "_cos_sin_flat")
    previous = {name: object() for name in fields}
    rope = SimpleNamespace(**previous)
    monkeypatch.setattr(ops, "rotary_embedding", rope, raising=False)
    original_rope = {"existing": object()}
    monkeypatch.setattr(upstream_rope, "_ROPE_DICT", dict(original_rope))
    original_context = {"existing": object()}
    context = dict(original_context)
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=context))
    with pytest.raises(RuntimeError, match="constructor failure"), _model_construction_state(cfg):
        assert not upstream_rope._ROPE_DICT
        assert all(getattr(rope, name) is None for name in fields)
        context["discarded"] = object()
        upstream_rope._ROPE_DICT["discarded"] = object()
        raise RuntimeError("constructor failure")
    assert context == original_context
    assert original_rope == upstream_rope._ROPE_DICT
    assert all(getattr(rope, name) is value for name, value in previous.items())


def test_failed_constructor_removes_only_its_new_model_hooks(monkeypatch):
    from torch.utils.hooks import RemovableHandle

    from vllm_ascend.model_loader.preload.loader import _model_construction_state

    class HookedModel(torch.nn.Module):
        def __init__(self, hooks):
            super().__init__()
            self._bytecode_hook_handle = RemovableHandle(hooks)
            hooks[self._bytecode_hook_handle.id] = self.callback

        def callback(self):
            pass

    hooks = OrderedDict()
    existing = HookedModel(hooks)
    dynamo = ModuleType("torch._dynamo.convert_frame")
    dynamo._bytecode_hooks = hooks
    monkeypatch.setitem(sys.modules, dynamo.__name__, dynamo)
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
    with pytest.raises(RuntimeError, match="discard"), _model_construction_state(cfg):
        discarded = HookedModel(hooks)
        unrelated = RemovableHandle(hooks)
        hooks[unrelated.id] = lambda: None
        raise RuntimeError("discard")
    assert existing._bytecode_hook_handle.id in hooks
    assert unrelated.id in hooks
    assert discarded._bytecode_hook_handle.id not in hooks


@pytest.mark.parametrize("failure", ["unavailable", "mismatch"])
@pytest.mark.parametrize("fallback", [False, True])
def test_cache_failure_obeys_upstream_fallback_policy(valid_config, monkeypatch, failure, fallback):
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        CacheConfigMismatchError,
        WeightCacheUnavailableError,
    )

    loader = _loader()
    loader.fallback = fallback
    exception = (
        WeightCacheUnavailableError("unavailable") if failure == "unavailable" else CacheConfigMismatchError("mismatch")
    )
    monkeypatch.setattr(loader, "_fetch_entries", Mock(side_effect=exception))
    disk_model = object()
    disk_load = Mock(return_value=disk_model)
    monkeypatch.setattr(loader, "_fallback_load", disk_load)
    if fallback:
        assert loader.load_model(valid_config, valid_config.model_config) is disk_model
        disk_load.assert_called_once()
    else:
        with pytest.raises(type(exception)):
            loader.load_model(valid_config, valid_config.model_config)
        disk_load.assert_not_called()


def test_copy_has_no_external_weight_budget(valid_config, monkeypatch):
    loader = _loader("copy")
    query = Mock(return_value=4096)
    monkeypatch.setattr(loader, "_daemon_memory", query)
    assert loader.get_external_weight_memory(valid_config) == 0
    assert loader.resident_cache_memory == 4096
    query.assert_called_once()


def test_external_memory_budget_avoids_false_startup_oom():
    from vllm.v1.worker.utils import request_memory

    snapshot = SimpleNamespace(total_memory=1000, free_memory=650, device_="npu:0")
    cache = SimpleNamespace(gpu_memory_utilization=0.9)
    assert request_memory(snapshot, cache, external_weight_memory=300) == 600
    with pytest.raises(ValueError, match="Free memory"):
        request_memory(snapshot, cache)
    with pytest.raises(ValueError, match="exceed"):
        request_memory(snapshot, cache, external_weight_memory=900)


def test_preprocessed_linear_does_not_transform_shared_weight(monkeypatch):
    from vllm.model_executor.utils import weights_already_processed

    from vllm_ascend.ops import linear

    layer = torch.nn.Linear(4, 4)
    weight = layer.weight
    transform = Mock(side_effect=AssertionError("shared weights must not be transformed"))
    monkeypatch.setattr(linear, "maybe_trans_nz", transform)
    with weights_already_processed():
        linear.AscendUnquantizedLinearMethod().process_weights_after_loading(layer)
    assert layer.weight is weight
    transform.assert_not_called()


@pytest.mark.parametrize("operation", ["sleep", "reload_weights", "update_weights"])
def test_mutating_operations_reject_shared_weights(operation):
    with pytest.raises(ValueError, match="daemon-owned"):
        config.require_private_weights(_loader().load_config, operation)
    config.require_private_weights(_loader("copy").load_config, operation)


@pytest.mark.parametrize("copy", [False, True])
def test_rope_globals_bind_imported_buffers_and_preserve_fast_path(monkeypatch, copy):
    from vllm_ascend.model_loader.preload.loader import _rebind_rope_caches
    from vllm_ascend.ops import rotary_embedding as rope

    model = object.__new__(rope.AscendRotaryEmbedding)
    torch.nn.Module.__init__(model)
    interleaved = torch.arange(48, dtype=torch.float32).reshape(6, 2, 4)
    cos, sin = interleaved[:, 0], interleaved[:, 1]
    model.register_buffer("cos_sin_cache", torch.ones(6, 8))
    model.register_buffer("_rope_derived_cos_cache", cos.clone() if copy else cos)
    model.register_buffer("_rope_derived_sin_cache", sin.clone() if copy else sin)
    for name in ("_cos_sin_cache", "_cos_cache", "_sin_cache", "_cos_sin_flat"):
        monkeypatch.setattr(rope, name, None)
    _rebind_rope_caches(model)
    assert rope._cos_sin_cache is model.cos_sin_cache
    torch.testing.assert_close(rope._cos_sin_flat, interleaved.flatten(0, 1))
    if not copy:
        assert rope._cos_sin_flat.data_ptr() == interleaved.data_ptr()


@pytest.mark.parametrize("rotary_class", ["AscendYaRNRotaryEmbedding", "AscendGemma4RotaryEmbedding"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["zero_copy", "copy"])
def test_rope_without_derived_buffers_rebinds_meta_cache(monkeypatch, rotary_class, dtype, mode):
    from vllm_ascend.model_loader.preload.loader import _rebind_rope_caches
    from vllm_ascend.ops import rotary_embedding as rope

    # Isolate the real class's buffer replacement contract from op dispatch.
    model = object.__new__(getattr(rope, rotary_class))
    torch.nn.Module.__init__(model)
    model.register_buffer("cos_sin_cache", torch.empty(16, 8, dtype=dtype, device="meta"), persistent=False)
    for name in ("_cos_sin_cache", "_cos_cache", "_sin_cache", "_cos_sin_flat", "_cos_slice", "_sin_slice"):
        monkeypatch.setattr(rope, name, None)
    rope._record_cos_sin_cache(model.cos_sin_cache)
    cache = torch.arange(128, dtype=torch.float32).reshape(16, 8).to(dtype)
    state = WeightCacheState({"cos_sin_cache": TensorEntry.from_tensor(cache, "buffer")}, {}, {})
    _loader(mode)._apply_entries(model, state, 0)
    assert rope._cos_sin_cache.is_meta
    _rebind_rope_caches(model)
    assert rope._cos_sin_cache is model.cos_sin_cache
    assert not rope._cos_sin_cache.is_meta
    assert rope._cos_cache is rope._sin_cache is rope._cos_sin_flat is None
    assert (model.cos_sin_cache.data_ptr() == cache.data_ptr()) == (mode == "zero_copy")

    monkeypatch.setattr(rope, "_cos", torch.ones(1, 2, 1, 8, dtype=dtype))
    monkeypatch.setattr(rope, "_sin", torch.zeros(1, 2, 1, 8, dtype=dtype))
    positions = torch.tensor([1, 3])
    rope.update_cos_sin(positions)
    expected_cos = cache[positions, :4].repeat(1, 2).reshape(1, 2, 1, 8)
    expected_sin = cache[positions, 4:].repeat(1, 2).reshape(1, 2, 1, 8)
    torch.testing.assert_close(rope._cos_slice, expected_cos)
    torch.testing.assert_close(rope._sin_slice, expected_sin)


def test_fp32_weight_reload_preserves_graph_storage(monkeypatch):
    from vllm_ascend.ops import linear

    layer = torch.nn.Linear(4, 4)
    layer.precast_fp32_weight = True
    layer.skip_weight_nz_conversion = True
    layer.prefix = "test"
    layer.weight_fp32 = torch.nn.Parameter(torch.zeros_like(layer.weight))
    previous = layer.weight_fp32
    monkeypatch.setattr(linear, "_should_keep_nd_for_compatibility_weight", lambda _: False)
    monkeypatch.setattr(linear, "_should_reshape_wo_a_to_3d", lambda *_: False)
    linear.AscendUnquantizedLinearMethod().process_weights_after_loading(layer)
    assert layer.weight_fp32 is previous
    torch.testing.assert_close(layer.weight_fp32, layer.weight)


def test_daemon_world_group_does_not_require_engine_config(monkeypatch):
    from vllm_ascend import utils
    from vllm_ascend.platform import NPUPlatform

    calculation = Mock(side_effect=AssertionError("world group must retain the daemon rendezvous context"))
    monkeypatch.setattr(utils, "calculate_dp_buffer_size", calculation)
    assert utils.get_hccl_config_for_pg_options("world") == utils.get_default_buffer_config()
    assert NPUPlatform.dist_backend == "hccl"
    calculation.assert_not_called()


def test_daemon_memory_logging_uses_native_npu_allocator():
    import vllm_ascend.patch.platform.patch_torch_accelerator  # noqa: F401

    assert torch.accelerator.memory_allocated is torch.npu.memory_allocated


def test_device_uuid_is_a_protocol_string(monkeypatch):
    from vllm_ascend.platform import NPUPlatform

    class NativeUUID:
        def __str__(self):
            return "00cc0800-d000-0000-0000-fc1803570b75"

    monkeypatch.setattr(torch.npu, "get_device_properties", lambda _: SimpleNamespace(uuid=NativeUUID()))
    assert NPUPlatform.get_device_uuid(0).encode() == b"00cc0800-d000-0000-0000-fc1803570b75"


def test_registration_keeps_old_vllm_loaders_untouched(monkeypatch):
    from vllm.model_executor import model_loader
    from vllm.model_executor.model_loader.weight_cache import daemon

    from vllm_ascend.model_loader.preload import register_preload_loader

    register = Mock()
    monkeypatch.setattr(model_loader, "register_model_loader", register)
    monkeypatch.delattr(daemon, "exported_weight_bytes")
    register_preload_loader()
    register.assert_not_called()


def test_spawn_entry_bootstraps_plugins_before_upstream_daemon(monkeypatch):
    import vllm_ascend
    from vllm_ascend.model_loader.preload import daemon
    from vllm_ascend.patch.platform import patch_preload

    order = []
    monkeypatch.setattr(vllm_ascend, "_ensure_global_patch", lambda: order.append("plugins"))
    monkeypatch.setattr(patch_preload, "apply_preload_patch", lambda: order.append("device adapters"))
    run = Mock(side_effect=lambda *_, **__: order.append("upstream"))
    monkeypatch.setattr(daemon.upstream, "_run_daemon_original", run)
    daemon.run_ascend_daemon(0, 0, "config", "rendezvous", "socket", "queue")
    assert order == ["plugins", "device adapters", "upstream"]
    run.assert_called_once_with(0, 0, "config", "rendezvous", "socket", "queue")


def test_copy_finishes_imported_storage_reads_before_release(monkeypatch):
    from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader

    order = []
    monkeypatch.setattr(torch.npu, "synchronize", lambda: order.append("complete copies"))
    monkeypatch.setattr(torch.npu, "mem_get_info", Mock(side_effect=[(100, 1000), (300, 1000)]))
    monkeypatch.setattr(IpcModelLoader, "_send_release", lambda _: order.append("release daemon weights"))
    loader = _loader("copy")
    loader._send_release()
    assert order == ["complete copies", "release daemon weights"]
    assert loader._released_cache_memory == 200


@pytest.mark.parametrize("resident,released", [(4096, 0), (4096, 2048), (4096, 8192), (0, 4096)])
def test_copy_release_is_not_counted_as_negative_engine_memory(monkeypatch, resident, released):
    from contextlib import nullcontext

    from vllm_ascend.worker import worker

    instance = worker.NPUWorker.__new__(worker.NPUWorker)
    instance.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=False), weight_transfer_config=None
    )
    model = torch.nn.Module()
    model._ascend_preload_released_cache_bytes = released
    instance.model_runner = Mock(get_model=Mock(return_value=model))
    instance._preload_copy_cache_memory = resident
    instance.init_snapshot = SimpleNamespace(free_memory=1000, cuda_memory=9000, non_torch_memory=8000)
    monkeypatch.setattr(worker, "set_current_vllm_config", lambda _: nullcontext())
    instance.load_model()
    correction = min(resident, released)
    assert instance.init_snapshot.free_memory == 1000 + correction
    assert instance.init_snapshot.cuda_memory == 9000 - correction
    assert instance.init_snapshot.non_torch_memory == 8000 - correction
    instance.load_model()
    assert instance.init_snapshot.free_memory == 1000 + correction


def test_daemon_copy_release_collects_cycles_and_hooks_before_acknowledging(monkeypatch):
    from torch.utils.hooks import RemovableHandle

    from vllm_ascend.model_loader.preload import daemon, loader

    class CyclicModel(torch.nn.Module):
        def callback(self):
            pass

    model = CyclicModel()
    model.weight = torch.nn.Parameter(torch.ones(4))
    model.weight.weight_loader = model.callback
    hooks = OrderedDict()
    model._bytecode_hook_handle = RemovableHandle(hooks)
    hooks[model._bytecode_hook_handle.id] = model.callback
    reference = weakref.ref(model)
    instance = daemon.AscendWeightCacheDaemon.__new__(daemon.AscendWeightCacheDaemon)
    instance.model = model
    context = {"layer": model}
    instance.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=context))
    del model
    order = []
    monkeypatch.setattr(loader, "_clear_rope_caches", lambda: order.append("clear caches"))
    monkeypatch.setattr(torch.npu, "ipc_collect", lambda: order.append("collect IPC"))

    def acknowledge(self, connection):
        assert self.model is None and reference() is None
        assert not hooks and not context
        order.append("upstream release")

    monkeypatch.setattr(daemon.AscendWeightCacheDaemon.__bases__[0], "_handle_release", acknowledge)
    # The patched upstream name points to the Ascend subclass; patch its
    # actual upstream base so super() retains the production dispatch path.
    instance._handle_release(None)
    assert order == ["clear caches", "collect IPC", "upstream release"]
