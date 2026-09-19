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
from unittest.mock import patch

import pytest
from vllm.model_executor.models.config import (
    HybridAttentionMambaModelConfig,
    MambaModelConfig,
)

from vllm_ascend.patch.platform.patch_mamba_config import (
    _get_sparse_index_kpool,
    _using_kv_store,
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


def _config(
    *,
    connector=None,
    disable_hybrid=False,
    prefix_caching=False,
    mamba_cache_mode="none",
    speculative_method=None,
):
    return SimpleNamespace(
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
        model_config=SimpleNamespace(max_model_len=4096),
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
