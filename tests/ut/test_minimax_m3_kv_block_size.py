# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend import utils
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.platform import NPUPlatform


def make_config(architecture, dtype):
    model = MagicMock()
    model.hf_config.architectures = [architecture]
    model.hf_config.model_type = "minimax_m3"
    model.is_hybrid = False
    model.dtype = torch.bfloat16
    model.get_num_kv_heads.return_value = 1
    model.get_head_size.return_value = 128
    return SimpleNamespace(
        model_config=model,
        parallel_config=MagicMock(),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True, enable_chunked_prefill=True),
        speculative_config=None,
        kv_transfer_config=None,
        cache_config=SimpleNamespace(
            block_size=128,
            cache_dtype=dtype,
            user_specified_block_size=False,
            enable_prefix_caching=True,
            kv_cache_dtype_skip_layers=["0", "1", "2"],
            skip_page_size_padded=None,
            mamba_page_size_padded=None,
        ),
    )


@pytest.mark.parametrize("architecture", ["MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"])
@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3"])
@pytest.mark.parametrize("user_specified", [False, True])
def test_m3_fp8_keeps_upstream_alignment_at_128(architecture, dtype, user_specified):
    config = make_config(architecture, dtype)
    config.cache_config.user_specified_block_size = user_specified
    with (
        patch.object(
            utils, "get_current_hardware_profile", return_value=get_hardware_profile(utils.AscendDeviceType.A5)
        ),
        patch("vllm_ascend.attention.attention_v1.get_current_vllm_config_or_none", return_value=config),
        patch("vllm.config.vllm.set_current_vllm_config", return_value=nullcontext()),
        patch.object(NPUPlatform, "_find_non_ssm_backend", return_value=AscendAttentionBackend),
    ):
        utils.refresh_block_size(config)
        # The runner snapshots this size before model loading. Keep it at the
        # final size; only the backend kernel minimum should become 64.
        assert config.cache_config.block_size == 128
        # Exercise the full hook repeatedly, including upstream dtype alignment.
        for _ in range(2):
            NPUPlatform.update_block_size_for_backend(config)
            assert config.cache_config.block_size == 128
            assert config.cache_config.skip_page_size_padded == 32768
        assert AscendAttentionBackend.get_supported_kernel_block_sizes() == [64, 128]


@pytest.mark.parametrize(
    "architecture,dtype,hardware",
    [
        ("OtherForCausalLM", "fp8", "A5"),
        ("MiniMaxM3SparseForCausalLM", "auto", "A5"),
        ("MiniMaxM3SparseForCausalLM", "bfloat16", "A5"),
        ("MiniMaxM3SparseForCausalLM", "fp8", "A3"),
        ("MiniMaxM3SparseForCausalLM", "fp8", "A2"),
    ],
)
def test_other_layouts_keep_kernel_block_sizes(architecture, dtype, hardware):
    config = make_config(architecture, dtype)
    with (
        patch.object(
            utils, "get_current_hardware_profile", return_value=get_hardware_profile(utils.AscendDeviceType[hardware])
        ),
        patch("vllm_ascend.attention.attention_v1.get_current_vllm_config_or_none", return_value=config),
    ):
        utils.refresh_block_size(config)
        assert config.cache_config.block_size == 128
        assert AscendAttentionBackend.get_supported_kernel_block_sizes() == [128]


def test_no_engine_context_keeps_default_backend():
    with patch("vllm_ascend.attention.attention_v1.get_current_vllm_config_or_none", return_value=None):
        assert AscendAttentionBackend.get_supported_kernel_block_sizes() == [128]


def test_without_skip_layers_keeps_128_for_indexer():
    config = make_config("MiniMaxM3SparseForCausalLM", "fp8")
    config.cache_config.kv_cache_dtype_skip_layers = []
    with (
        patch.object(
            utils, "get_current_hardware_profile", return_value=get_hardware_profile(utils.AscendDeviceType.A5)
        ),
        patch("vllm_ascend.attention.attention_v1.get_current_vllm_config_or_none", return_value=config),
        patch("vllm.config.vllm.set_current_vllm_config", return_value=nullcontext()),
        patch.object(NPUPlatform, "_find_non_ssm_backend", return_value=AscendAttentionBackend),
    ):
        NPUPlatform.update_block_size_for_backend(config)
        assert AscendAttentionBackend.get_supported_kernel_block_sizes() == [128]
        assert config.cache_config.block_size == 128
        assert config.cache_config.skip_page_size_padded is None


def test_separate_draft_does_not_reset_shared_cache():
    config = make_config("MiniMaxM3SparseForCausalLM", "fp8")
    config.speculative_config = SimpleNamespace(draft_model_config=config.model_config, target_model_config=object())
    utils.refresh_block_size(config)
    assert config.cache_config.block_size == 128
