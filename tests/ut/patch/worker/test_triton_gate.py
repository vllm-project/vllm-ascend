# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import builtins
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("triton_supported", [False, True])
@pytest.mark.parametrize("copy_supported", [False, True])
def test_mamba_patch_keeps_upstream_kernels_when_capability_is_missing(triton_supported, copy_supported):
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
            return SimpleNamespace(supports_triton=lambda: triton_supported)
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
    namespace = {"__builtins__": {**vars(builtins), "__import__": capture_import}}
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
    utils = SimpleNamespace(supports_triton=lambda: triton_supported)

    def capture_import(name, *args, **kwargs):
        imported.append(name)
        if name == "vllm_ascend.device.hardware_profile":
            return hardware
        if name == "vllm_ascend.utils":
            return utils
        return MagicMock()

    source = REPO_ROOT / "vllm_ascend/patch/worker/__init__.py"
    namespace = {"__builtins__": {**vars(builtins), "__import__": capture_import}}
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
