# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import builtins
import importlib

import pytest

from vllm_ascend.patch.worker import patch_triton
from vllm_ascend.utils import vllm_version_is


@pytest.mark.skipif(vllm_version_is("0.33.0"), reason="vLLM 0.33.0 has no MRV2 suffix")
def test_worker_patch_binds_suffix_launch_adapters():
    upstream = importlib.import_module("vllm.v1.worker.gpu.spec_decode.suffix.speculator")
    ops = importlib.import_module("vllm_ascend.ops.triton.spec_decode.suffix")

    bindings = (
        (upstream._suffix_local_scan_kernel, ops.launch_suffix_local_scan),
        (upstream._suffix_finalize_kernel, ops.launch_suffix_finalize),
        (upstream._suffix_observe_kernel, ops.launch_suffix_observe),
        (upstream._suffix_ingest_kernel, ops.launch_suffix_ingest),
    )
    for adapter, launch in bindings:
        assert isinstance(adapter, ops.SuffixKernelLaunchAdapter)
        assert adapter.launch is launch


@pytest.mark.skipif(not vllm_version_is("0.33.0"), reason="requires the vLLM 0.33.0 release baseline")
def test_vllm_0330_worker_patch_skips_suffix_imports(monkeypatch):
    original_import = builtins.__import__

    def import_without_suffix(name, *args, **kwargs):
        assert not name.startswith("vllm.v1.worker.gpu.spec_decode.suffix")
        assert name != "vllm_ascend.ops.triton.spec_decode.suffix"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_suffix)
    importlib.reload(patch_triton)
