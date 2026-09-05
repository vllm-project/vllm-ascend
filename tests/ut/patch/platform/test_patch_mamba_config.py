from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.model_executor.models.config import (
    HybridAttentionMambaModelConfig,
    MambaModelConfig,
)

from vllm_ascend.patch.platform.patch_mamba_config import _using_kv_store


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
