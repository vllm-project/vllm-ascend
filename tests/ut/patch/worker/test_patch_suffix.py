# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import builtins
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.ops.triton.spec_decode import suffix as ops
from vllm_ascend.patch.worker import patch_triton


@pytest.fixture
def mock_suffix_import(monkeypatch):
    module_name = "vllm.v1.worker.gpu.spec_decode.suffix.speculator"
    original_import = builtins.__import__

    def install(load_suffix):
        def import_with_suffix(name, *args, **kwargs):
            if name != module_name:
                return original_import(name, *args, **kwargs)
            # A dotted import without a from-list returns the top-level module.
            module = load_suffix()
            for part in reversed(module_name.split(".")[1:]):
                module = SimpleNamespace(**{part: module})
            return module

        monkeypatch.setattr(builtins, "__import__", import_with_suffix)

    return install


@pytest.mark.parametrize("repeat", [False, True])
def test_suffix_patch_installs_persistent_launch_adapters(monkeypatch, mock_suffix_import, repeat):
    module = SimpleNamespace()
    mock_suffix_import(lambda: module)
    launch = MagicMock()
    monkeypatch.setattr(ops, "launch_suffix_observe", launch)
    patch_triton.patch_suffix_kernels()
    if repeat:
        patch_triton.patch_suffix_kernels()

    module._suffix_observe_kernel[(7,)]("request state", K=3)
    launch.assert_called_once_with((7,), "request state", K=3)
    assert module._suffix_local_scan_kernel.launch is ops.launch_suffix_local_scan
    assert module._suffix_finalize_kernel.launch is ops.launch_suffix_finalize
    assert module._suffix_ingest_kernel.launch is ops.launch_suffix_ingest


def test_suffix_bindings_remain_installed_after_launch_failure(monkeypatch, mock_suffix_import):
    module = SimpleNamespace()
    mock_suffix_import(lambda: module)
    launch = MagicMock(side_effect=RuntimeError("launch failed"))
    monkeypatch.setattr(ops, "launch_suffix_observe", launch)
    patch_triton.patch_suffix_kernels()
    adapter = module._suffix_observe_kernel
    with pytest.raises(RuntimeError, match="launch failed"):
        adapter[(1,)]()
    assert module._suffix_observe_kernel is adapter


@pytest.mark.parametrize(
    "missing",
    ["vllm.v1.worker.gpu.spec_decode.suffix", "vllm.v1.worker.gpu.spec_decode.suffix.speculator"],
)
def test_older_runtime_without_suffix_does_not_block_other_methods(mock_suffix_import, missing):
    def import_missing():
        raise ModuleNotFoundError(name=missing)

    mock_suffix_import(import_missing)
    patch_triton.patch_suffix_kernels()


def test_suffix_patch_does_not_hide_unrelated_dependency_failure(mock_suffix_import):
    def import_missing():
        raise ModuleNotFoundError("Missing runtime dependency", name="unrelated_dependency")

    mock_suffix_import(import_missing)
    with pytest.raises(ModuleNotFoundError, match="Missing runtime dependency"):
        patch_triton.patch_suffix_kernels()
