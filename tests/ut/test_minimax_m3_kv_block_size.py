# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.platforms import Platform

from vllm_ascend import utils
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.minimax_m3.msa_m3 import AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3SparseBackend
from vllm_ascend.platform import NPUPlatform


def _non_ssm_backend_patch(backends=None):
    if backends is None:
        backends = [AscendAttentionBackend]
    return patch.object(NPUPlatform, "_find_non_ssm_backends", return_value=backends)


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
        _non_ssm_backend_patch(),
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


@pytest.mark.parametrize("architecture", ["MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"])
@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3"])
@pytest.mark.parametrize("user_specified", [False, True])
@pytest.mark.parametrize(
    "backends",
    [
        [AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3SparseBackend],
        [AscendMiniMaxM3SparseBackend, AscendMiniMaxM3IndexerBackend],
        [AscendAttentionBackend, AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3SparseBackend],
    ],
    ids=["indexer-first-pp-stage", "sparse-first-pp-stage", "dense-first-pp-stage"],
)
def test_m3_fp8_pp_stages_keep_128_without_dense_layers(architecture, dtype, user_specified, backends):
    config = make_config(architecture, dtype)
    config.parallel_config.pipeline_parallel_size = 2
    config.cache_config.user_specified_block_size = user_specified
    with (
        patch.object(
            utils, "get_current_hardware_profile", return_value=get_hardware_profile(utils.AscendDeviceType.A5)
        ),
        patch("vllm_ascend.attention.attention_v1.get_current_vllm_config_or_none", return_value=config),
        patch("vllm.config.vllm.set_current_vllm_config", return_value=nullcontext()),
        _non_ssm_backend_patch(backends),
    ):
        utils.refresh_block_size(config)
        assert config.cache_config.block_size == 128
        for _ in range(2):
            NPUPlatform.update_block_size_for_backend(config)
            assert config.cache_config.block_size == 128
            assert config.cache_config.skip_page_size_padded == 32768
        # Page alignment must not claim that sparse/indexer kernels support 64.
        assert AscendMiniMaxM3IndexerBackend.get_supported_kernel_block_sizes() == [128]
        assert AscendMiniMaxM3SparseBackend.get_supported_kernel_block_sizes() == [128]


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
        with patch.object(Platform, "_align_heterogeneous_kv_block_size") as align:
            NPUPlatform._align_heterogeneous_kv_block_size(config, AscendMiniMaxM3IndexerBackend)
            align.assert_called_once_with(config, AscendMiniMaxM3IndexerBackend)


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
        _non_ssm_backend_patch(),
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
