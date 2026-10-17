# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config.attention import AttentionConfig
from vllm.config.cache import CacheConfig
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

from vllm_ascend.attention.dsa_attn_kv_plan import is_a5_bf16_kv_enabled
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v4.indexer import get_indexer_k_dtype
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache, VllmDeepseekV4SWACache
from vllm_ascend.models.layer.attention.layer import DSAAttention


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("main", ["auto", "fp8", "bfloat16"])
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_dsv4_cache_specs_resolve_auto_without_changing_config(device_type, main, compress_ratio):
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=CacheConfig(cache_dtype=main, block_size=128),
        attention_config=AttentionConfig(),
    )
    is_a5 = device_type == AscendDeviceType.A5
    expected = (
        torch.float8_e4m3fn if is_a5 and main == "auto" else kv_cache_dtype_str_to_dtype(main, config.model_config)
    )
    profile = get_hardware_profile(device_type)

    # Only stub upstream layer registration; exercise the real dtype and specs.
    def init_swa(self, head_dim, window_size, dtype, prefix, cache_config):
        torch.nn.Module.__init__(self)
        self.head_dim, self.window_size, self.cache_config = head_dim, window_size, cache_config

    with (
        patch.object(VllmDeepseekV4SWACache, "__init__", init_swa),
        patch("vllm_ascend.models.deepseek_v4.model.get_current_hardware_profile", return_value=profile),
        patch("vllm_ascend.models.layer.attention.layer.get_current_hardware_profile", return_value=profile),
        patch("vllm_ascend.models.deepseek_v4.indexer.get_current_hardware_profile", return_value=profile),
        patch("vllm_ascend.attention.dsa_attn_kv_plan.get_current_hardware_profile", return_value=profile),
    ):
        swa = AscendDeepseekV4SWACache(
            512, 128, kv_cache_dtype_str_to_dtype(main, config.model_config), "swa", config.cache_config
        )
        attn = DSAAttention.__new__(DSAAttention)
        torch.nn.Module.__init__(attn)
        attn.kv_cache_dtype, attn.head_size, attn.compress_ratio = main, 512, compress_ratio
        swa_spec, compressed_spec = swa.get_kv_cache_spec(config), attn.get_kv_cache_spec(config)
        assert swa.dtype == swa_spec.dtype == compressed_spec.dtype == expected
        if is_a5:
            assert swa_spec.head_size == compressed_spec.head_size == (512 if main == "bfloat16" else 640)
            assert swa_spec.page_size_bytes == compressed_spec.page_size_bytes
        assert is_a5_bf16_kv_enabled(config) == (is_a5 and main == "bfloat16")
        assert get_indexer_k_dtype(config) == (torch.float8_e4m3fn if is_a5 else torch.int8)
        assert config.cache_config.cache_dtype == main
        assert config.attention_config.indexer_kv_dtype == "auto"
