# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch
from vllm.config import cache as cache_mod
from vllm.utils import torch_utils

from vllm_ascend.patch.platform import patch_kv_cache_dtype as platform_patch
from vllm_ascend.patch.worker import patch_kv_cache_dtype as worker_patch


@pytest.mark.skipif(not hasattr(cache_mod, "KV_CACHE_DTYPES"), reason="Requires the vLLM dtype registry")
@pytest.mark.parametrize(
    "apply_patch",
    [platform_patch._apply_kv_cache_dtype_int8_patch, worker_patch._widen_cache_dtype_literal_for_worker],
)
@pytest.mark.parametrize("dtype", ["int8", "mxfp8", "nvfp4_ds_mla", "late_custom_dtype"])
def test_registry_validation_preserves_upstream_and_late_dtypes(monkeypatch, apply_patch, dtype):
    registry = [d for d in cache_mod.KV_CACHE_DTYPES if d not in ("int8", "mxfp8")]
    monkeypatch.setattr(cache_mod, "KV_CACHE_DTYPES", registry)
    annotation = cache_mod.CacheConfig.__annotations__["cache_dtype"]
    apply_patch()
    apply_patch()
    registry.append("late_custom_dtype")

    assert cache_mod.KV_CACHE_DTYPES is registry
    assert registry.count("int8") == registry.count("mxfp8") == 1
    assert cache_mod.CacheConfig.__annotations__["cache_dtype"] is annotation
    assert cache_mod.CacheConfig(cache_dtype=dtype).cache_dtype == dtype


@pytest.mark.parametrize("has_api", [False, True])
def test_fp8_fallback_without_a_registered_handler(monkeypatch, has_api):
    monkeypatch.setitem(torch_utils.STR_DTYPE_TO_TORCH_DTYPE, "fp8", torch.uint8)
    if has_api:
        get_handler = Mock(return_value=None)
        monkeypatch.setattr(cache_mod, "get_kv_cache_dtype_handler", get_handler, raising=False)
    else:
        monkeypatch.delattr(cache_mod, "get_kv_cache_dtype_handler", raising=False)

    worker_patch._apply_fp8_kv_cache_dtype_flip()

    assert torch_utils.kv_cache_dtype_str_to_dtype("fp8", None) == torch.float8_e4m3fn


def test_registered_fp8_handler_remains_authoritative(monkeypatch):
    monkeypatch.setitem(torch_utils.STR_DTYPE_TO_TORCH_DTYPE, "fp8", torch.uint8)
    get_handler = Mock(return_value=object())
    monkeypatch.setattr(cache_mod, "get_kv_cache_dtype_handler", get_handler, raising=False)

    worker_patch._apply_fp8_kv_cache_dtype_flip()

    assert torch_utils.kv_cache_dtype_str_to_dtype("fp8", None) == torch.uint8
