# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend import utils
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.minimax_m3 import MiniMaxM3Attention
from vllm_ascend.models.minimax_m3 import msa_m3 as msa_m3_module
from vllm_ascend.models.minimax_m3.msa_m3 import (
    AscendMiniMaxM3DenseBackend,
    AscendMiniMaxM3IndexerBackend,
    AscendMiniMaxM3SparseBackend,
)
from vllm_ascend.platform import NPUPlatform


def _non_ssm_backend_patch(backends=None):
    if backends is None:
        backends = [AscendMiniMaxM3DenseBackend]
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
            msa_m3_module, "get_current_hardware_profile", return_value=get_hardware_profile(utils.AscendDeviceType.A5)
        ),
        patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config_or_none", return_value=config),
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
        assert AscendMiniMaxM3DenseBackend.get_supported_kernel_block_sizes() == [64, 128]


@pytest.mark.parametrize("architecture", ["MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"])
@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3"])
@pytest.mark.parametrize("user_specified", [False, True])
@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize(
    "backends",
    [
        [AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3SparseBackend],
        [AscendMiniMaxM3SparseBackend, AscendMiniMaxM3IndexerBackend],
        [AscendMiniMaxM3DenseBackend, AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3SparseBackend],
        [AscendMiniMaxM3IndexerBackend],
        [AscendMiniMaxM3SparseBackend],
    ],
    ids=["indexer-first-pp-stage", "sparse-first-pp-stage", "dense-first-pp-stage", "indexer-only", "sparse-only"],
)
def test_m3_fp8_pp_stages_keep_128_without_dense_layers(architecture, dtype, user_specified, pp_size, backends):
    config = make_config(architecture, dtype)
    config.parallel_config.pipeline_parallel_size = pp_size
    config.cache_config.user_specified_block_size = user_specified
    with (
        patch.object(
            msa_m3_module, "get_current_hardware_profile", return_value=get_hardware_profile(utils.AscendDeviceType.A5)
        ),
        patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config_or_none", return_value=config),
        patch("vllm.config.vllm.set_current_vllm_config", return_value=nullcontext()),
        _non_ssm_backend_patch(backends),
    ):
        utils.refresh_block_size(config)
        assert config.cache_config.block_size == 128
        for _ in range(2):
            NPUPlatform.update_block_size_for_backend(config)
            assert config.cache_config.block_size == 128
            assert config.cache_config.skip_page_size_padded == 32768
        for backend in backends:
            assert backend.get_supported_kernel_block_sizes() == [64, 128]
            # The alignment minimum must not switch FP8 execution to 64-token
            # sparse/indexer kernels. Exercise the real kernel-size selector.
            assert select_common_block_size(config.cache_config.block_size, [backend]) == 128
        assert select_common_block_size(config.cache_config.block_size, backends) == 128


@pytest.mark.parametrize(
    "architecture,dtype,hardware",
    [
        ("OtherForCausalLM", "fp8", "A5"),
        ("MiniMaxM3SparseForCausalLM", "auto", "A5"),
        ("MiniMaxM3SparseForCausalLM", "bfloat16", "A5"),
        ("MiniMaxM3SparseForCausalLM", "fp8_e5m2", "A5"),
        ("MiniMaxM3SparseForCausalLM", "fp8", "A3"),
        ("MiniMaxM3SparseForCausalLM", "fp8", "A2"),
    ],
)
@pytest.mark.parametrize(
    "backend", [AscendMiniMaxM3DenseBackend, AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3SparseBackend]
)
def test_other_layouts_keep_kernel_block_sizes(architecture, dtype, hardware, backend):
    config = make_config(architecture, dtype)
    with (
        patch.object(
            msa_m3_module,
            "get_current_hardware_profile",
            return_value=get_hardware_profile(utils.AscendDeviceType[hardware]),
        ),
        patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config_or_none", return_value=config),
    ):
        utils.refresh_block_size(config)
        assert config.cache_config.block_size == 128
        assert backend.get_supported_kernel_block_sizes() == [128]


@pytest.mark.parametrize(
    "backend", [AscendMiniMaxM3DenseBackend, AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3SparseBackend]
)
def test_no_engine_context_keeps_default_backend(backend):
    with (
        patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config_or_none", return_value=None),
    ):
        assert backend.get_supported_kernel_block_sizes() == [128]


@pytest.mark.parametrize(
    "backend", [AscendMiniMaxM3DenseBackend, AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3SparseBackend]
)
def test_without_skip_layers_keeps_128_for_indexer(backend):
    config = make_config("MiniMaxM3SparseForCausalLM", "fp8")
    config.cache_config.kv_cache_dtype_skip_layers = []
    with (
        patch.object(
            msa_m3_module, "get_current_hardware_profile", return_value=get_hardware_profile(utils.AscendDeviceType.A5)
        ),
        patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config_or_none", return_value=config),
        patch("vllm.config.vllm.set_current_vllm_config", return_value=nullcontext()),
        _non_ssm_backend_patch([backend]),
    ):
        NPUPlatform.update_block_size_for_backend(config)
        assert backend.get_supported_kernel_block_sizes() == [128]
        assert config.cache_config.block_size == 128
        assert config.cache_config.skip_page_size_padded is None


def test_separate_draft_does_not_reset_shared_cache():
    config = make_config("MiniMaxM3SparseForCausalLM", "fp8")
    config.speculative_config = SimpleNamespace(draft_model_config=config.model_config, target_model_config=object())
    utils.refresh_block_size(config)
    assert config.cache_config.block_size == 128


@pytest.mark.parametrize("hardware", ["A2", "A3", "A5"])
@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("skip_layers", [False, True])
@pytest.mark.parametrize("standard_backend", [False, True])
def test_dense_layer_uses_model_backend_only_for_affected_config(hardware, dtype, skip_layers, standard_backend):
    config = make_config("MiniMaxM3SparseForCausalLM", dtype)
    if not skip_layers:
        config.cache_config.kv_cache_dtype_skip_layers = []
    model_module = "vllm_ascend.models.minimax_m3.minimax_m3"
    original_backend = AscendAttentionBackend if standard_backend else type("OtherBackend", (), {})
    original_attention = torch.nn.Identity()
    original_attention.attn_backend = original_backend
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                msa_m3_module,
                "get_current_hardware_profile",
                return_value=get_hardware_profile(utils.AscendDeviceType[hardware]),
            )
        )
        stack.enter_context(patch.object(msa_m3_module, "get_current_vllm_config_or_none", return_value=config))
        stack.enter_context(patch(f"{model_module}.get_tensor_model_parallel_world_size", return_value=1))
        for name in ("QKVParallelLinear", "RowParallelLinear", "GemmaRMSNorm", "get_rope"):
            stack.enter_context(patch(f"{model_module}.{name}", return_value=torch.nn.Identity()))
        attention = stack.enter_context(patch(f"{model_module}.Attention", return_value=original_attention))
        MiniMaxM3Attention(
            hidden_size=256,
            num_heads=2,
            num_kv_heads=1,
            rotary_dim=128,
            cache_config=config.cache_config,
            prefix="model.layers.0.self_attn",
        )
    expected = (
        AscendMiniMaxM3DenseBackend
        if standard_backend and hardware == "A5" and dtype in ("fp8", "fp8_e4m3") and skip_layers
        else original_backend
    )
    assert "attn_backend" not in attention.call_args.kwargs
    assert original_attention.attn_backend is expected


@pytest.mark.parametrize("hardware", ["A2", "A3", "A5"])
def test_generic_dense_backend_keeps_128_for_m3(hardware):
    config = make_config("MiniMaxM3SparseForCausalLM", "fp8")
    with (
        patch.object(
            msa_m3_module,
            "get_current_hardware_profile",
            return_value=get_hardware_profile(utils.AscendDeviceType[hardware]),
        ),
        patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config_or_none", return_value=config),
    ):
        assert AscendAttentionBackend.get_supported_kernel_block_sizes() == [128]


@pytest.mark.parametrize("is_dcp_enabled", [False, True])
def test_model_dense_backend_reuses_generic_implementation(is_dcp_enabled):
    with patch("vllm_ascend.attention.attention_v1.is_dcp_enabled", return_value=is_dcp_enabled):
        assert AscendMiniMaxM3DenseBackend.get_impl_cls() is AscendAttentionBackend.get_impl_cls()
        assert AscendMiniMaxM3DenseBackend.get_builder_cls() is AscendAttentionBackend.get_builder_cls()
        assert AscendMiniMaxM3DenseBackend.supports_pcp() == AscendAttentionBackend.supports_pcp()
    assert AscendMiniMaxM3DenseBackend.get_name() == AscendAttentionBackend.get_name()
    assert AscendMiniMaxM3DenseBackend.get_kv_cache_shape(3, 128, 2, 128) == AscendAttentionBackend.get_kv_cache_shape(
        3, 128, 2, 128
    )
