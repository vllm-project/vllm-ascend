# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.ops.triton.spec_decode import suffix as ops
from vllm_ascend.patch.worker import patch_triton


@pytest.mark.parametrize("repeat", [False, True])
def test_suffix_patch_installs_persistent_launch_adapters(monkeypatch, repeat):
    module = SimpleNamespace()
    import_module = MagicMock(return_value=module)
    monkeypatch.setattr(patch_triton, "import_module", import_module)
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


def test_suffix_bindings_remain_installed_after_launch_failure(monkeypatch):
    module = SimpleNamespace()
    monkeypatch.setattr(patch_triton, "import_module", lambda name: module)
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
def test_older_runtime_without_suffix_does_not_block_other_methods(monkeypatch, missing):
    def import_missing(name):
        raise ModuleNotFoundError(name=missing)

    monkeypatch.setattr(patch_triton, "import_module", import_missing)
    patch_triton.patch_suffix_kernels()


def test_suffix_patch_does_not_hide_unrelated_dependency_failure(monkeypatch):
    def import_missing(name):
        raise ModuleNotFoundError("Missing runtime dependency", name="unrelated_dependency")

    monkeypatch.setattr(patch_triton, "import_module", import_missing)
    with pytest.raises(ModuleNotFoundError, match="Missing runtime dependency"):
        patch_triton.patch_suffix_kernels()
