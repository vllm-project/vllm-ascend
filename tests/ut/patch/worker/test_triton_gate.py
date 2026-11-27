# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import builtins
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("triton_supported", [False, True])
@pytest.mark.parametrize("copy_supported", [False, True])
@pytest.mark.parametrize("legacy_vllm", [False, True])
def test_mamba_patch_keeps_upstream_kernels_when_capability_is_missing(triton_supported, copy_supported, legacy_vllm):
    imported = []
    original_copy = object()
    original_postprocess = object()
    mamba_utils = SimpleNamespace(batch_memcpy=original_copy, postprocess_mamba_fused_kernel=original_postprocess)
    buffers = SimpleNamespace(create=MagicMock())
    kernel = MagicMock()
    postprocess = object()

    def capture_import(name, *args, **kwargs):
        if name == "vllm.v1.worker":
            return SimpleNamespace(mamba_utils=mamba_utils)
        if name == "vllm.v1.worker.mamba_utils":
            return SimpleNamespace(MambaCopyBuffers=buffers)
        if name == "vllm_ascend.utils":
            return SimpleNamespace(
                supports_triton=lambda: triton_supported,
                vllm_version_is=lambda version: legacy_vllm,
            )
        if name == "vllm_ascend.device.hardware_profile":
            return SimpleNamespace(
                HardwareCapability=SimpleNamespace(TRITON_BATCH_MEMCPY=object()),
                get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda cap: copy_supported),
            )
        if name.startswith("vllm_ascend.ops.triton"):
            assert triton_supported and copy_supported
            imported.append(name)
            return SimpleNamespace(batch_memcpy_kernel=kernel, postprocess_mamba_fused_kernel=postprocess)
        return builtins.__import__(name, *args, **kwargs)

    source = REPO_ROOT / "vllm_ascend/patch/worker/patch_mamba_utils.py"
    namespace: dict[str, Any] = {"__builtins__": {**vars(builtins), "__import__": capture_import}}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    if triton_supported and copy_supported:
        assert len(imported) == 2
        assert mamba_utils.batch_memcpy is namespace["_batch_memcpy_triton"]
        assert mamba_utils.postprocess_mamba_fused_kernel is postprocess
    else:
        assert not imported
        assert mamba_utils.batch_memcpy is original_copy
        assert mamba_utils.postprocess_mamba_fused_kernel is original_postprocess


@pytest.mark.parametrize("standard_patches", [False, True])
@pytest.mark.parametrize("triton_supported", [False, True])
def test_worker_patch_selection_and_order(standard_patches, triton_supported):
    """Execute bundle registration without applying process-wide patches."""
    imported = []
    profile = SimpleNamespace(supports=lambda capability: standard_patches)
    hardware = SimpleNamespace(
        HardwareCapability=SimpleNamespace(STANDARD_WORKER_PATCHES=object()),
        get_current_hardware_profile=lambda: profile,
    )
    utils = SimpleNamespace(supports_triton=lambda: triton_supported, vllm_version_is=lambda *args, **kwargs: False)

    def capture_import(name, *args, **kwargs):
        imported.append(name)
        if name == "vllm_ascend.device.hardware_profile":
            return hardware
        if name == "vllm_ascend.utils":
            return utils
        return MagicMock()

    source = REPO_ROOT / "vllm_ascend/patch/worker/__init__.py"
    namespace: dict[str, Any] = {"__builtins__": {**vars(builtins), "__import__": capture_import}}
    exec(compile(source.read_text(), str(source), "exec"), namespace)

    prefix = "vllm_ascend.patch.worker."
    assert (prefix + "patch_triton" in imported) is triton_supported
    assert (prefix + "patch_ascend_ops" in imported) is triton_supported
    if triton_supported:
        assert imported.index(prefix + "patch_ascend_ops") < imported.index(prefix + "patch_triton")
    assert (prefix + "patch_qwen3_5" in imported) is standard_patches
    assert (prefix + "patch_idex_310" in imported) is not standard_patches
    assert (prefix + "patch_spec_decode_310" in imported) is not standard_patches
    assert prefix + "patch_mamba_utils" in imported


@pytest.mark.parametrize("triton_supported", [False, True])
def test_cache_binding_does_not_load_step5_on_unsupported_workers(triton_supported):
    """Import the real patch and bind caches across its model import boundary."""
    imported = []

    class CacheLayer:
        def __init__(self):
            self.bind_kv_cache = MagicMock()

    class Step5Ring(CacheLayer):
        pass

    class Step5Summary(CacheLayer):
        pass

    worker_utils = SimpleNamespace(defaultdict=defaultdict, extract_layer_index=lambda *args: 0)

    def capture_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.v1.worker.utils":
            if fromlist:
                return worker_utils
            return SimpleNamespace(v1=SimpleNamespace(worker=SimpleNamespace(utils=worker_utils)))
        if name == "vllm.model_executor.layers.attention":
            return SimpleNamespace(Attention=object)
        if name == "vllm.v1.kv_cache_interface":
            return SimpleNamespace(KVCacheGroupSpec=object)
        if name == "vllm_ascend.attention.dsa_v41":
            return SimpleNamespace(DeepseekV41CacheLayer=CacheLayer)
        if name == "vllm_ascend.utils":
            return SimpleNamespace(supports_triton=lambda: triton_supported, vllm_version_is=lambda version: False)
        if name == "vllm_ascend.models.step5.sparse_attention":
            assert triton_supported, "Cache binding imported Step5's Triton dependency on an unsupported worker"
            imported.append(name)
            return SimpleNamespace(AscendStep5CSATailRing=Step5Ring, AscendStep5SparseSummaryCacheLayer=Step5Summary)
        return builtins.__import__(name, globals, locals, fromlist, level)

    source = REPO_ROOT / "vllm_ascend/patch/worker/patch_bind_kv_cache.py"
    namespace: dict[str, Any] = {"__builtins__": {**vars(builtins), "__import__": capture_import}}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    assert not imported

    ordinary = SimpleNamespace(kv_cache=None, bind_kv_cache=MagicMock())
    layers: dict[str, SimpleNamespace | CacheLayer] = {
        "model.layers.0.attn": ordinary,
        "model.layers.1.attn": CacheLayer(),
    }
    if triton_supported:
        layers.update({"model.layers.2.ring": Step5Ring(), "model.layers.3.summary": Step5Summary()})
    caches = {name: object() for name in layers}
    namespace["bind_kv_cache_to_layers"](caches, layers)

    assert ordinary.kv_cache is caches["model.layers.0.attn"]
    ordinary.bind_kv_cache.assert_not_called()
    for name, layer in layers.items():
        if layer is not ordinary:
            layer.bind_kv_cache.assert_called_once_with(caches[name])
    assert bool(imported) is triton_supported


@pytest.mark.parametrize("device_type", ["_310P", None])
def test_plugin_310p_build_does_not_import_installed_triton(device_type):
    """Fail on a Triton import even if package discovery says it exists."""
    program = f"""
import builtins
import importlib.util
import sys
from types import ModuleType

build = ModuleType('vllm_ascend._build_info')
build.__device_type__ = {device_type!r}
build.__soc_version__ = 'ascend310p3'
sys.modules[build.__name__] = build
original_import = builtins.__import__
original_find_spec = importlib.util.find_spec

def guarded_import(name, *args, **kwargs):
    if name == 'triton' or name.startswith('triton.'):
        raise AssertionError('310P imported ' + name)
    if name == 'vllm_ascend.logger':
        return ModuleType(name)
    return original_import(name, *args, **kwargs)

def find_spec(name, *args, **kwargs):
    if name == 'triton':
        return importlib.util.spec_from_loader(name, loader=None)
    return original_find_spec(name, *args, **kwargs)

builtins.__import__ = guarded_import
importlib.util.find_spec = find_spec
import vllm_ascend
assert vllm_ascend._triton_available is False
"""
    subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT, check=True, timeout=30)


@pytest.mark.parametrize("triton_supported", [False, True])
def test_triton_package_boundary_rejects_unsupported_implementation(triton_supported):
    """A direct Triton-only model import must fail before model construction."""
    source = REPO_ROOT / "vllm_ascend/ops/triton/__init__.py"

    def capture_import(name, *args, **kwargs):
        if name == "vllm_ascend.device.hardware_profile":
            return SimpleNamespace(
                HardwareCapability=SimpleNamespace(TRITON_KERNELS=object()),
                get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda _: triton_supported),
            )
        return builtins.__import__(name, *args, **kwargs)

    namespace: dict[str, Any] = {"__builtins__": {**vars(builtins), "__import__": capture_import}}
    if triton_supported:
        exec(compile(source.read_text(), str(source), "exec"), namespace)
    else:
        with pytest.raises(RuntimeError, match="Triton-Ascend implementations are unavailable"):
            exec(compile(source.read_text(), str(source), "exec"), namespace)


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen4ExpForCausalLM",
        "Step5ForCausalLM",
        "Glm5NextForCausalLM",
        "DeepseekV41ForCausalLM",
        "MiniMaxM3SparseForCausalLM",
    ],
)
def test_model_level_gate_rejects_triton_only_architecture(monkeypatch, architecture):
    profile = SimpleNamespace(supports=lambda capability: False)
    hardware = SimpleNamespace(
        HardwareCapability=SimpleNamespace(TRITON_KERNELS=object()),
        get_current_hardware_profile=lambda: profile,
    )
    source = REPO_ROOT / "vllm_ascend/device/triton_gate.py"

    def capture_import(name, *args, **kwargs):
        if name == "vllm_ascend.device.hardware_profile":
            return hardware
        return builtins.__import__(name, *args, **kwargs)

    namespace: dict[str, Any] = {"__builtins__": {**vars(builtins), "__import__": capture_import}}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]), hf_text_config=None)

    with pytest.raises(ValueError, match="requires Triton-Ascend"):
        namespace["validate_model_triton_compatibility"](config)


@pytest.mark.parametrize("architecture", ["LlamaForCausalLM", "DeepseekV4ForCausalLM"])
@pytest.mark.parametrize("triton_supported", [False, True])
def test_model_level_gate_allows_native_or_supported_architecture(monkeypatch, architecture, triton_supported):
    profile = SimpleNamespace(supports=lambda capability: triton_supported)
    hardware = SimpleNamespace(
        HardwareCapability=SimpleNamespace(TRITON_KERNELS=object()),
        get_current_hardware_profile=lambda: profile,
    )
    source = REPO_ROOT / "vllm_ascend/device/triton_gate.py"

    def capture_import(name, *args, **kwargs):
        if name == "vllm_ascend.device.hardware_profile":
            return hardware
        return builtins.__import__(name, *args, **kwargs)

    namespace: dict[str, Any] = {"__builtins__": {**vars(builtins), "__import__": capture_import}}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]), hf_text_config=None)

    namespace["validate_model_triton_compatibility"](config)


@pytest.mark.parametrize("triton_supported", [False, True])
def test_cold_registry_gate_runs_before_lazy_model_import(triton_supported):
    """A cold ModelConfig registry lookup must gate before loading the model class."""
    imported = []

    class FakeRegistry:
        def inspect_model_cls(self, architectures, model_config):
            imported.append(architectures)
            return "model-info", architectures[0]

    profile = SimpleNamespace(supports=lambda capability: triton_supported)
    hardware = SimpleNamespace(
        HardwareCapability=SimpleNamespace(TRITON_KERNELS=object()),
        get_current_hardware_profile=lambda: profile,
    )
    source = REPO_ROOT / "vllm_ascend/device/triton_gate.py"

    def capture_import(name, *args, **kwargs):
        if name == "vllm_ascend.device.hardware_profile":
            return hardware
        return builtins.__import__(name, *args, **kwargs)

    namespace: dict[str, Any] = {"__builtins__": {**vars(builtins), "__import__": capture_import}}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    registry = FakeRegistry()
    namespace["install_model_registry_gate"](registry)
    wrapped = FakeRegistry.inspect_model_cls
    namespace["install_model_registry_gate"](registry)
    assert FakeRegistry.inspect_model_cls is wrapped

    if triton_supported:
        assert registry.inspect_model_cls(["Qwen4ExpForCausalLM"], SimpleNamespace()) == (
            "model-info",
            "Qwen4ExpForCausalLM",
        )
        assert imported == [["Qwen4ExpForCausalLM"]]
    else:
        with pytest.raises(ValueError, match="requires Triton-Ascend"):
            registry.inspect_model_cls(["Qwen4ExpForCausalLM"], SimpleNamespace())
        assert imported == []

    native_architectures = ["LlamaForCausalLM"]
    assert registry.inspect_model_cls(native_architectures, SimpleNamespace()) == ("model-info", "LlamaForCausalLM")
    assert imported[-1] is native_architectures
