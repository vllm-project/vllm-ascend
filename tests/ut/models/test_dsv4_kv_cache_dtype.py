# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.models.deepseek_v4.indexer import (
    AscendDeepseekV4IndexerCache,
    _dsv4_indexer_kv_dtype,
)
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
from vllm_ascend.models.layer.attention.layer import (
    dsv4_resolve_attn_kv_dtype,
    get_dsv4_block_sizes,
)
from vllm_ascend.utils import AscendDeviceType


def _vllm_config(cache_dtype: str, model_dtype: torch.dtype = torch.bfloat16):
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype, block_size=128),
        model_config=SimpleNamespace(dtype=model_dtype),
    )


@pytest.mark.parametrize(
    ("device_type", "cache_dtype", "expected_attn", "expected_indexer"),
    [
        (AscendDeviceType.A5, "bfloat16", torch.bfloat16, torch.float8_e4m3fn),
        # CLI / STR_DTYPE uses "fp8" / "fp8_e4m3", not "float8_e4m3fn".
        (AscendDeviceType.A5, "fp8", torch.float8_e4m3fn, torch.float8_e4m3fn),
        (AscendDeviceType.A5, "fp8_e4m3", torch.float8_e4m3fn, torch.float8_e4m3fn),
        # A5 "auto" must not follow model BF16 into SparseFlashMla.
        (AscendDeviceType.A5, "auto", torch.float8_e4m3fn, torch.float8_e4m3fn),
        (AscendDeviceType.A2, "bfloat16", torch.bfloat16, torch.int8),
        (AscendDeviceType.A2, "auto", torch.bfloat16, torch.int8),
    ],
)
def test_dsv4_a5_attn_kv_dtype_respects_request(
    device_type,
    cache_dtype,
    expected_attn,
    expected_indexer,
):
    vllm_config = _vllm_config(cache_dtype)
    with (
        patch(
            "vllm_ascend.models.layer.attention.layer.get_ascend_device_type",
            return_value=device_type,
        ),
        patch(
            "vllm_ascend.models.deepseek_v4.indexer.get_ascend_device_type",
            return_value=device_type,
        ),
    ):
        assert dsv4_resolve_attn_kv_dtype(vllm_config, torch.bfloat16) == expected_attn
        assert _dsv4_indexer_kv_dtype() == expected_indexer


@pytest.mark.parametrize(
    ("attn_kv_dtype", "expected_pads"),
    [
        (torch.float8_e4m3fn, [16896, 81920]),
        (None, [16896, 81920]),
        (torch.bfloat16, [16896, 131072]),
    ],
)
def test_a5_block_sizes_bf16_uses_bf16_attn_pad(attn_kv_dtype, expected_pads):
    with patch(
        "vllm_ascend.models.layer.attention.layer.get_ascend_device_type",
        return_value=AscendDeviceType.A5,
    ):
        sizes = get_dsv4_block_sizes(attn_kv_dtype)
    assert sizes[128][0] == [128, 128, 8, 16]
    assert sizes[128][1] == expected_pads


def test_a5_auto_swa_spec_stays_fp8_not_sparse_flash_mla():
    vllm_config = _vllm_config("auto")
    swa = AscendDeepseekV4SWACache.__new__(AscendDeepseekV4SWACache)
    swa.head_dim = 512
    swa.dtype = torch.bfloat16
    swa.window_size = 128
    swa.block_size = 128

    with (
        patch(
            "vllm_ascend.models.deepseek_v4.model.get_ascend_device_type",
            return_value=AscendDeviceType.A5,
        ),
        patch(
            "vllm_ascend.models.layer.attention.layer.get_ascend_device_type",
            return_value=AscendDeviceType.A5,
        ),
    ):
        spec = swa.get_kv_cache_spec(vllm_config)

    assert spec.dtype == torch.float8_e4m3fn
    assert spec.head_size == 512 + 128
    assert vllm_config.cache_config.cache_dtype == "auto"


def test_swa_get_kv_cache_spec_keeps_a5_bf16_and_does_not_mutate_cache_config():
    vllm_config = _vllm_config("bfloat16")
    swa = AscendDeepseekV4SWACache.__new__(AscendDeepseekV4SWACache)
    swa.head_dim = 512
    swa.dtype = torch.bfloat16
    swa.window_size = 128
    swa.block_size = 128

    with (
        patch(
            "vllm_ascend.models.deepseek_v4.model.get_ascend_device_type",
            return_value=AscendDeviceType.A5,
        ),
        patch(
            "vllm_ascend.models.layer.attention.layer.get_ascend_device_type",
            return_value=AscendDeviceType.A5,
        ),
    ):
        spec = swa.get_kv_cache_spec(vllm_config)

    assert spec.dtype == torch.bfloat16
    assert spec.head_size == 512
    assert vllm_config.cache_config.cache_dtype == "bfloat16"


def test_indexer_get_kv_cache_spec_stays_fp8_on_a5_without_mutating_cache_config():
    vllm_config = _vllm_config("bfloat16")
    indexer = AscendDeepseekV4IndexerCache.__new__(AscendDeepseekV4IndexerCache)
    indexer.head_dim = 128
    indexer.dtype = torch.bfloat16  # stale value must not win on A5
    indexer.compress_ratio = 4

    with (
        patch(
            "vllm_ascend.models.deepseek_v4.indexer.get_ascend_device_type",
            return_value=AscendDeviceType.A5,
        ),
        patch(
            "vllm_ascend.models.layer.attention.layer.get_ascend_device_type",
            return_value=AscendDeviceType.A5,
        ),
    ):
        spec = indexer.get_kv_cache_spec(vllm_config)

    assert indexer.dtype == torch.float8_e4m3fn
    assert spec.dtype == torch.float8_e4m3fn
    assert vllm_config.cache_config.cache_dtype == "bfloat16"
