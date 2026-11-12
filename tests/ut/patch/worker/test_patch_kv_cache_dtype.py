# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import vllm.config.cache as cache_config
import vllm.utils.torch_utils as torch_utils

from vllm_ascend.patch.worker import patch_kv_cache_dtype


@pytest.mark.parametrize("has_registration_api", [False, True])
def test_fp8_raw_byte_fallback(monkeypatch, has_registration_api):
    if has_registration_api:
        monkeypatch.setattr(cache_config, "register_kv_cache_dtype", object(), raising=False)
    else:
        monkeypatch.delattr(cache_config, "register_kv_cache_dtype", raising=False)
    mapping = {"fp8": torch.uint8, "int8": torch.int8, "fp8_e5m2": torch.uint8}
    monkeypatch.setattr(torch_utils, "STR_DTYPE_TO_TORCH_DTYPE", mapping)

    patch_kv_cache_dtype._apply_fp8_kv_cache_dtype_flip()
    assert torch_utils.kv_cache_dtype_str_to_dtype("fp8", None) == torch.float8_e4m3fn
    assert mapping == {"fp8": torch.float8_e4m3fn, "int8": torch.int8, "fp8_e5m2": torch.uint8}
    # Reapplying the worker patch must not change a resolved mapping.
    patch_kv_cache_dtype._apply_fp8_kv_cache_dtype_flip()
    assert torch_utils.kv_cache_dtype_str_to_dtype("fp8", None) == torch.float8_e4m3fn


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2, None])
def test_registration_mapping_is_preserved(monkeypatch, dtype):
    monkeypatch.setattr(cache_config, "register_kv_cache_dtype", object(), raising=False)
    mapping = {"int8": torch.int8}
    if dtype is not None:
        mapping["fp8"] = dtype
    expected = mapping.copy()
    monkeypatch.setattr(torch_utils, "STR_DTYPE_TO_TORCH_DTYPE", mapping)

    patch_kv_cache_dtype._apply_fp8_kv_cache_dtype_flip()

    assert mapping == expected
