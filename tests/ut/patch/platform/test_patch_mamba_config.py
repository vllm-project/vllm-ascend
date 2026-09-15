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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.patch.platform.patch_mamba_config import (
    _get_sparse_index_kpool,
    _uses_mla_fa_quant_cache,
    verify_and_update_config,
)


def test_non_modelslim_quantization_does_not_enable_mla_fa_quant_cache():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            use_mla=True,
            quantization="compressed-tensors",
            model="/weights/other-model",
            revision=None,
        ),
        quant_config=None,
    )

    with patch(
        "vllm_ascend.patch.platform.patch_mamba_config.model_uses_fa_quantization",
        return_value=True,
    ) as detect_fa_quant:
        assert not _uses_mla_fa_quant_cache(vllm_config)

    detect_fa_quant.assert_not_called()


def test_mla_fa_quant_metadata_doubles_attention_block_size_before_quant_config_init():
    cache_config = SimpleNamespace(
        cache_dtype="auto",
        block_size=None,
        mamba_page_size_padded=None,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        mamba_block_size=None,
    )
    model_config = SimpleNamespace(
        use_mla=True,
        dtype=torch.bfloat16,
        architecture="AnyHybridMLAModel",
        model="/weights/any-mla-model",
        revision=None,
        quantization=None,
        hf_text_config=SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        ),
        max_model_len=32768,
        get_num_kv_heads=MagicMock(return_value=1),
    )
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=None,
        speculative_config=None,
        quant_config=None,
    )
    model_cls = SimpleNamespace(
        get_mamba_state_shape_from_config=MagicMock(return_value=((2304, 3), (6, 128, 128))),
        get_mamba_state_dtype_from_config=MagicMock(return_value=(torch.bfloat16, torch.float32)),
    )

    with (
        patch("vllm_ascend.patch.platform.patch_mamba_config.MambaModelConfig.verify_and_update_config"),
        patch(
            "vllm_ascend.patch.platform.patch_mamba_config.ModelRegistry.resolve_model_cls",
            return_value=(model_cls, None),
        ),
        patch(
            "vllm_ascend.patch.platform.patch_mamba_config.model_uses_fa_quantization",
            return_value=True,
        ) as detect_fa_quant,
    ):
        verify_and_update_config.__func__(None, vllm_config)

    assert cache_config.block_size == 768
    assert cache_config.mamba_page_size_padded == 505344
    assert cache_config.mamba_block_size == 768
    detect_fa_quant.assert_called_once_with(
        "/weights/any-mla-model",
        revision=None,
    )
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
