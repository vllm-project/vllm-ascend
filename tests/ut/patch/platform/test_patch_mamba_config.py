# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.models.config import (
    HybridAttentionMambaModelConfig,
    MambaModelConfig,
)
from vllm.platforms.interface import Platform

from vllm_ascend.attention.utils import get_sfa_qsfa_packed_head_dim
from vllm_ascend.models.kimi_k3.kimi_k3 import AscendKimiK3ForConditionalGeneration
from vllm_ascend.patch.platform.patch_mamba_config import (
    _get_sparse_index_kpool,
    _using_kv_store,
)
from vllm_ascend.platform import NPUPlatform


def _model_config(**text_config):
    return SimpleNamespace(
        hf_text_config=SimpleNamespace(**text_config),
        hf_config=SimpleNamespace(),
    )


def test_sparse_index_kpool_detection_is_model_agnostic():
    model_config = _model_config(
        model_type="another_hybrid_model",
        index_topk=2048,
        index_kpool=4,
    )

    assert _get_sparse_index_kpool(model_config) == 4


def test_dense_model_with_index_kpool_field_uses_generic_layout():
    model_config = _model_config(
        model_type="glm5_next",
        index_topk=None,
        index_kpool=4,
    )

    assert _get_sparse_index_kpool(model_config) is None


def test_sparse_indexer_without_kpool_uses_generic_layout():
    model_config = _model_config(index_topk=2048)

    assert _get_sparse_index_kpool(model_config) is None


@pytest.mark.parametrize("index_kpool", [None, 0, 1, "4"])
def test_active_sparse_index_kpool_requires_valid_ratio(index_kpool):
    model_config = _model_config(
        index_topk=2048,
        index_kpool=index_kpool,
    )

    with pytest.raises(ValueError, match="integer greater than 1"):
        _get_sparse_index_kpool(model_config)


def _config(
    *,
    connector=None,
    disable_hybrid=False,
    prefix_caching=False,
    mamba_cache_mode="none",
    speculative_method=None,
):
    return SimpleNamespace(
        use_flashinfer_replayssm=False,
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=disable_hybrid,
        ),
        kv_transfer_config=connector,
        speculative_config=(None if speculative_method is None else SimpleNamespace(method=speculative_method)),
        cache_config=SimpleNamespace(
            block_size=128,
            mamba_page_size_padded=8192,
            mamba_cache_mode=mamba_cache_mode,
            enable_prefix_caching=prefix_caching,
            mamba_block_size=None,
        ),
        model_config=SimpleNamespace(max_model_len=4096, hf_config=SimpleNamespace(model_type="dummy")),
    )


def _run(config):
    with patch.object(MambaModelConfig, "verify_and_update_config") as upstream:
        HybridAttentionMambaModelConfig.verify_and_update_config(config)
    upstream.assert_called_once_with(config)


def test_hybrid_config_preserves_noncontiguous_page_sizes():
    config = _config()

    _run(config)

    assert config.cache_config.block_size == 128
    assert config.cache_config.mamba_page_size_padded == 8192
    assert config.cache_config.mamba_block_size == 4096


@pytest.mark.parametrize("cache_dtype", ["fp8", "int8", "auto"])
@pytest.mark.parametrize("block_size", [None, 128, 2048])
@pytest.mark.parametrize("use_mla", [True, False])
def test_kpool_page_alignment_preserves_packed_and_regular_geometry(cache_dtype, block_size, use_mla):
    config = _config(prefix_caching=True, mamba_cache_mode="align")
    config.cache_config.cache_dtype = cache_dtype
    config.cache_config.block_size = block_size
    config.cache_config.mamba_page_size_padded = None
    config.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    config.model_config = SimpleNamespace(
        architecture="DummyHybridModel",
        use_mla=use_mla,
        dtype=torch.bfloat16,
        max_model_len=4096,
        hf_text_config=SimpleNamespace(index_topk=2048, index_kpool=4, kv_lora_rank=512, qk_rope_head_dim=64),
        hf_config=SimpleNamespace(),
        get_num_kv_heads=lambda _parallel_config: 1 if use_mla else 2,
        get_head_size=lambda: 128,
    )
    dummy_model_cls = SimpleNamespace(
        get_mamba_state_shape_from_config=lambda _config: ((256, 256), (256, 441)),
        get_mamba_state_dtype_from_config=lambda _config: (torch.bfloat16, torch.bfloat16),
    )
    packed_token_page_size = get_sfa_qsfa_packed_head_dim(512, 64)
    assert packed_token_page_size == 512 + 64 * 2 + (512 // 128) * 4

    with patch(
        "vllm_ascend.patch.platform.patch_mamba_config.ModelRegistry.resolve_model_cls",
        return_value=(dummy_model_cls, None),
    ) as resolve:
        _run(config)

    resolve.assert_called_once_with(config.model_config.architecture, model_config=config.model_config)
    if use_mla and cache_dtype in ("fp8", "int8"):
        expected_token_page_size = packed_token_page_size
        expected_block_size = 640
    else:
        expected_token_page_size = 1152 if use_mla else 1024 if cache_dtype == "auto" else 512
        minimum_block_size = 384 if cache_dtype == "auto" else 768
        expected_block_size = max(block_size or 128, minimum_block_size)
    assert config.cache_config.block_size == expected_block_size
    padded_page_size = config.cache_config.mamba_page_size_padded
    assert isinstance(padded_page_size, int)
    assert padded_page_size == expected_block_size * expected_token_page_size
    assert padded_page_size >= 356864
    assert config.cache_config.mamba_block_size == expected_block_size


@pytest.mark.parametrize("cache_dtype", ["int8", "fp8"])
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_backend_alignment_preserves_kpool_c8_page_geometry(cache_dtype, prefix_caching):
    config = _config(prefix_caching=prefix_caching, mamba_cache_mode="align" if prefix_caching else "none")
    config.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    config.cache_config.cache_dtype = cache_dtype
    config.cache_config.user_specified_block_size = False
    config.cache_config.user_specified_mamba_block_size = False
    config.cache_config.kv_cache_dtype_skip_layers = []
    config.cache_config.mamba_page_size_padded = None
    config.model_config = SimpleNamespace(
        architecture="DummyHybridModel",
        is_hybrid=True,
        use_mla=True,
        dtype=torch.bfloat16,
        max_model_len=133120,
        hf_text_config=SimpleNamespace(index_topk=2048, index_kpool=4, kv_lora_rank=512, qk_rope_head_dim=0),
        hf_config=SimpleNamespace(),
        get_num_kv_heads=lambda _parallel_config: 1,
        get_head_size=lambda: 512,
    )
    # GLM-5.3-Flash's convolution and recurrent-state pages occupy 4,587,520 bytes.
    model_cls = SimpleNamespace(
        get_mamba_state_shape_from_config=lambda _config: ((8, 24576), (64, 128, 128)),
        get_mamba_state_dtype_from_config=lambda _config: (torch.bfloat16, torch.float32),
    )
    backend = SimpleNamespace(
        get_preferred_block_size=lambda _default: 128,
        get_supported_kernel_block_sizes=lambda: [128],
    )

    with (
        patch.object(MambaModelConfig, "verify_and_update_config"),
        patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls", return_value=(model_cls, None)),
        patch("vllm.config.vllm.set_current_vllm_config", return_value=nullcontext()),
    ):
        HybridAttentionMambaModelConfig.verify_and_update_config(config)
        runner_block_size = config.cache_config.block_size
        assert runner_block_size == 8704

        # The backend pass resets the preferred block to 128. Generic MLA
        # sizing omits the 16 scale bytes and changes it to 8960 instead.
        for _ in range(2):
            config.cache_config.block_size = backend.get_preferred_block_size(128)
            NPUPlatform._align_hybrid_block_size(config, backend)
            assert config.cache_config.block_size == runner_block_size
            assert config.cache_config.mamba_page_size_padded == runner_block_size * 528
            assert config.cache_config.block_size // 4 % 16 == 0
            expected_mamba_block_size = runner_block_size if prefix_caching else config.model_config.max_model_len
            assert config.cache_config.mamba_block_size == expected_mamba_block_size


@pytest.mark.parametrize(
    "use_mla,cache_dtype,kpool",
    [(True, "auto", True), (True, "int8", False), (False, "int8", True)],
)
def test_backend_alignment_delegates_regular_cache_geometry(use_mla, cache_dtype, kpool):
    from vllm.v1.attention.backend import AttentionBackend

    config = _config()
    config.cache_config.cache_dtype = cache_dtype
    config.model_config.use_mla = use_mla
    config.model_config.hf_text_config = SimpleNamespace(**({"index_kpool": 4} if kpool else {}))
    backend = AttentionBackend

    with patch.object(Platform, "_align_hybrid_block_size") as upstream:
        NPUPlatform._align_hybrid_block_size(config, backend)

    upstream.assert_called_once_with(config, backend)


@pytest.mark.parametrize("tp_size", [8, 16])
@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "int8", "fp8", "turboquant_4bit_nc"])
def test_kimi_k31_platform_alignment_preserves_native_pages(tp_size, mamba_cache_mode, cache_dtype):
    config = _config(prefix_caching=mamba_cache_mode == "align", mamba_cache_mode=mamba_cache_mode)
    config.parallel_config = SimpleNamespace(tensor_parallel_size=tp_size)
    config.cache_config.cache_dtype = cache_dtype
    config.cache_config.mamba_block_size = 131072
    config.cache_config.mamba_page_size_padded = 66502656
    config.attention_config = SimpleNamespace(resolve_indexer_kv_dtype=lambda _default: "fp8")
    config.model_config = SimpleNamespace(
        architecture="KimiK3ForConditionalGeneration",
        use_mla=True,
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(model_type="kimi_k31"),
        hf_text_config=SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            dsa_config=SimpleNamespace(index_head_dim=128),
            linear_attn_config={"full_attn_layers": [4, 5], "kda_layers": [1, 2, 3]},
        ),
        get_num_kv_heads=lambda _parallel: 1,
        get_head_size=lambda: 576,
    )
    local_heads = 96 // tp_size

    class KimiGeometry(AscendKimiK3ForConditionalGeneration):
        # Keep the real model's inherited cache-layout hook. A namespace with
        # only state getters hides #66037's new whole-row packed selection.
        @classmethod
        def get_mamba_state_shape_from_config(cls, _config):
            return ((3, local_heads * 384), (local_heads, 128, 128))

        @classmethod
        def get_mamba_state_dtype_from_config(cls, _config):
            return (torch.bfloat16, torch.float32)

    from vllm_ascend.attention.sfa_v1 import AscendSFABackend

    model_cls = KimiGeometry
    backend = AscendSFABackend
    if cache_dtype in ("int8", "fp8"):
        expected_block_size, token_bytes = 12288 // tp_size, 656
    elif cache_dtype == "turboquant_4bit_nc":
        # A large requested block covers the recurrent state even with TQ4.
        expected_block_size, token_bytes = 4096, 386
    else:
        expected_block_size, token_bytes = 6144 // tp_size, 1152

    with (
        patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls", return_value=(model_cls, None)),
        patch("vllm.model_executor.layers.mamba.abstract.get_mamba_prefill_checkpoint_alignment", return_value=128),
        patch.object(NPUPlatform, "_find_non_ssm_backends", return_value=[backend]),
    ):
        for _ in range(2):
            config.cache_config.block_size = 4096 if cache_dtype == "turboquant_4bit_nc" else 128
            NPUPlatform._align_hybrid_block_size(config, backend)
            assert config.cache_config.block_size == expected_block_size
            assert config.cache_config.mamba_page_size_padded == expected_block_size * token_bytes
            assert config.cache_config.mamba_block_size == (
                expected_block_size if mamba_cache_mode == "align" else 131072
            )


@pytest.mark.parametrize(
    "connector",
    [
        SimpleNamespace(kv_connector="AscendStoreConnector"),
        SimpleNamespace(
            kv_connector="MultiConnector",
            kv_connector_extra_config={
                "connectors": [{"kv_connector": "AscendStoreConnector"}],
            },
        ),
    ],
)
def test_using_kv_store_recognizes_supported_connectors(connector):
    assert _using_kv_store(_config(connector=connector))


@pytest.mark.parametrize(
    "connector",
    [
        None,
        SimpleNamespace(kv_connector="OtherConnector"),
        SimpleNamespace(
            kv_connector="MultiConnector",
            kv_connector_extra_config=None,
        ),
    ],
)
def test_using_kv_store_rejects_unrelated_connectors(connector):
    assert not _using_kv_store(_config(connector=connector))


def test_kv_store_aligns_mamba_cache_for_prefix_caching():
    config = _config(
        connector=SimpleNamespace(kv_connector="AscendStoreConnector"),
        prefix_caching=True,
    )

    _run(config)

    assert config.cache_config.mamba_cache_mode == "align"
    assert config.cache_config.mamba_block_size == config.cache_config.block_size


def test_disabled_hybrid_manager_does_not_force_align_mode():
    config = _config(
        connector=SimpleNamespace(kv_connector="AscendStoreConnector"),
        disable_hybrid=True,
        prefix_caching=True,
    )

    _run(config)

    assert config.cache_config.mamba_cache_mode == "none"
    assert config.cache_config.mamba_block_size == config.model_config.max_model_len


def test_extract_hidden_states_does_not_force_align_mode():
    config = _config(
        connector=SimpleNamespace(kv_connector="AscendStoreConnector"),
        prefix_caching=True,
        speculative_method="extract_hidden_states",
    )

    _run(config)

    assert config.cache_config.mamba_cache_mode == "none"
    assert config.cache_config.mamba_block_size == config.model_config.max_model_len


def test_kv_store_rejects_non_align_explicit_mode():
    config = _config(
        connector=SimpleNamespace(kv_connector="AscendStoreConnector"),
        mamba_cache_mode="all",
    )

    with (
        patch.object(MambaModelConfig, "verify_and_update_config"),
        pytest.raises(AssertionError, match="only support 'align'"),
    ):
        HybridAttentionMambaModelConfig.verify_and_update_config(config)


@pytest.mark.parametrize(
    "case_name,model_type,use_mla,cache_dtype,backend_name,expected",
    [
        ("qwen35_gdn_bf16", "qwen3_next", False, "auto", "dense", (1152, 1179648, 1152)),
        ("kimi_linear_bf16", "kimi_linear", False, "auto", "dense", (896, 917504, 896)),
        ("kimi_k31_int8", "kimi_k31", True, "int8", "sfa", (1536, 1536 * 656, 1536)),
        ("kimi_k31_fp8", "kimi_k31", True, "fp8", "sfa", (1536, 1536 * 656, 1536)),
        ("kimi_k31_tq4", "kimi_k31", True, "turboquant_4bit_nc", "sfa", (1536, 1536 * 386, 1536)),
        ("glm5_kpool_int8", "glm_moe_dsa", True, "int8", "sfa", (8704, 8704 * 528, 8704)),
        ("glm5_kpool_fp8", "glm_moe_dsa", True, "fp8", "sfa", (8704, 8704 * 528, 8704)),
    ],
)
def test_hybrid_platform_alignment_baseline_characterization(
    case_name, model_type, use_mla, cache_dtype, backend_name, expected
):
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
    from vllm_ascend.attention.sfa_v1 import AscendSFABackend

    config = _config(prefix_caching=True, mamba_cache_mode="align")
    config.scheduler_config.enable_chunked_prefill = True
    config.parallel_config = SimpleNamespace(tensor_parallel_size=8 if model_type == "kimi_k31" else 1)
    config.cache_config.cache_dtype = cache_dtype
    config.cache_config.block_size = 128
    config.cache_config.user_specified_block_size = False
    config.cache_config.user_specified_mamba_block_size = False
    config.cache_config.kv_cache_dtype_skip_layers = []
    config.cache_config.mamba_page_size_padded = None
    config.cache_config.mamba_block_size = 131072 if model_type == "kimi_k31" else None
    config.compilation_config = SimpleNamespace(static_forward_context={})
    config.attention_config = SimpleNamespace(
        hisparse_config=None,
        resolve_indexer_kv_dtype=lambda _default: "fp8",
    )

    if model_type == "qwen3_next":
        text_cfg = SimpleNamespace(model_type=model_type)
        mamba_shapes, mamba_dtypes = ((3, 1024), (16, 128, 128)), (torch.bfloat16, torch.float32)
        num_heads, head_dim = 2, 128
        max_len = 32768
        model_cls_base = SimpleNamespace
    elif model_type == "kimi_linear":
        text_cfg = SimpleNamespace(model_type=model_type)
        mamba_shapes, mamba_dtypes = ((3, 1024), (12, 128, 128)), (torch.bfloat16, torch.float32)
        num_heads, head_dim = 2, 128
        max_len = 32768
        model_cls_base = SimpleNamespace
    elif model_type == "kimi_k31":
        text_cfg = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            dsa_config=SimpleNamespace(index_head_dim=128),
            linear_attn_config={"full_attn_layers": [4, 5], "kda_layers": [1, 2, 3]},
        )
        local_heads = 12
        mamba_shapes, mamba_dtypes = ((3, local_heads * 384), (local_heads, 128, 128)), (torch.bfloat16, torch.float32)
        num_heads, head_dim = 1, 576
        max_len = 131072
        model_cls_base = AscendKimiK3ForConditionalGeneration
    else:
        text_cfg = SimpleNamespace(index_topk=2048, index_kpool=4, kv_lora_rank=512, qk_rope_head_dim=0)
        mamba_shapes, mamba_dtypes = ((8, 24576), (64, 128, 128)), (torch.bfloat16, torch.float32)
        num_heads, head_dim = 1, 512
        max_len = 133120
        model_cls_base = SimpleNamespace

    config.model_config = SimpleNamespace(
        architecture=f"Dummy_{case_name}",
        is_hybrid=True,
        use_mla=use_mla,
        dtype=torch.bfloat16,
        max_model_len=max_len,
        hf_config=SimpleNamespace(model_type=model_type),
        hf_text_config=text_cfg,
        get_num_kv_heads=lambda _parallel: num_heads,
        get_head_size=lambda: head_dim,
    )

    model_cls: Any
    if model_cls_base is SimpleNamespace:
        model_cls = SimpleNamespace(
            get_mamba_state_shape_from_config=lambda _config: mamba_shapes,
            get_mamba_state_dtype_from_config=lambda _config: mamba_dtypes,
        )
    else:

        class KimiCharGeometry(AscendKimiK3ForConditionalGeneration):
            @classmethod
            def get_mamba_state_shape_from_config(cls, _config):
                return mamba_shapes

            @classmethod
            def get_mamba_state_dtype_from_config(cls, _config):
                return mamba_dtypes

        model_cls = KimiCharGeometry

    backend = AscendSFABackend if backend_name == "sfa" else AscendAttentionBackend
    with (
        patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls", return_value=(model_cls, None)),
        patch("vllm.model_executor.layers.mamba.abstract.get_mamba_prefill_checkpoint_alignment", return_value=128),
        patch.object(NPUPlatform, "_find_non_ssm_backends", return_value=[backend]),
        patch.object(NPUPlatform, "_check_aligned_block_size"),
    ):
        if model_type == "glm_moe_dsa":
            with patch.object(MambaModelConfig, "verify_and_update_config"):
                HybridAttentionMambaModelConfig.verify_and_update_config(config)
        NPUPlatform.update_block_size_for_backend(config)

    assert (
        config.cache_config.block_size,
        config.cache_config.mamba_page_size_padded,
        config.cache_config.mamba_block_size,
    ) == expected


def test_non_sfa_mla_backend_leaves_hybrid_page_unoverridden():
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole

    from vllm_ascend.attention.mla_v1 import AscendMLABackend
    from vllm_ascend.attention.sfa_v1 import AscendSFABackend
    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

    config = _config(prefix_caching=True, mamba_cache_mode="align")
    config.scheduler_config.enable_chunked_prefill = True
    config.parallel_config = SimpleNamespace(tensor_parallel_size=8)
    config.cache_config.cache_dtype = "int8"
    config.cache_config.mamba_block_size = 131072
    config.cache_config.mamba_page_size_padded = None
    config.attention_config = SimpleNamespace(hisparse_config=None)
    config.model_config = SimpleNamespace(
        architecture="DummyHybridMLA",
        use_mla=True,
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(model_type="kimi_k31"),
        hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        get_num_kv_heads=lambda _parallel: 1,
        get_head_size=lambda: 576,
    )
    local_heads = 12
    model_cls = SimpleNamespace(
        get_mamba_state_shape_from_config=lambda _cfg: ((3, local_heads * 384), (local_heads, 128, 128)),
        get_mamba_state_dtype_from_config=lambda _cfg: (torch.bfloat16, torch.float32),
    )

    with (
        patch("vllm.model_executor.models.ModelRegistry.resolve_model_cls", return_value=(model_cls, None)),
        patch("vllm.model_executor.layers.mamba.abstract.get_mamba_prefill_checkpoint_alignment", return_value=128),
        patch.object(NPUPlatform, "_find_non_ssm_backends", return_value=[AscendMLABackend]),
    ):
        config.cache_config.block_size = 128
        NPUPlatform._align_hybrid_block_size(config, AscendMLABackend)

    # Generic AscendMLABackend does not rewrite MLA probes into SFA packed slots.
    assert config.cache_config.block_size == 1536
    assert config.cache_config.mamba_page_size_padded == 1536 * 576

    # AscendSFABackend.customize_spec leaves owner specs, indexer specs, and hisparse specs untouched.
    owner_spec = AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=672, dtype=torch.int8)
    indexer_probe = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576, dtype=torch.int8, cache_role=SparseCacheRole.INDEXER
    )
    mla_probe = MLAAttentionSpec(block_size=1, num_kv_heads=1, head_size=576, dtype=torch.int8)
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(config):
        assert AscendSFABackend.customize_spec(owner_spec) is owner_spec
        assert AscendSFABackend.customize_spec(indexer_probe) is indexer_probe
        customized = AscendSFABackend.customize_spec(mla_probe)
        assert customized.head_size == 656
        config.attention_config.hisparse_config = object()
        assert AscendSFABackend.customize_spec(mla_probe) is mla_probe
