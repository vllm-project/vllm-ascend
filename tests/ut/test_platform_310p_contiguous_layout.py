from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.platforms import Platform

from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import AscendDeviceType


def _config(block_size, *, hybrid=False, prefix=False, transfer=False):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            user_specified_block_size=False,
            mamba_page_size_padded=block_size * 2048 if hybrid else None,
            mamba_cache_mode="align" if hybrid else "none",
            mamba_block_size=8192,
            enable_prefix_caching=prefix,
        ),
        model_config=SimpleNamespace(is_hybrid=hybrid, max_model_len=8192),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=SimpleNamespace(kv_connector="AscendStoreConnector") if transfer else None,
    )


@pytest.mark.parametrize("block_size", [128, 1024])
@pytest.mark.parametrize("hybrid", [False, True])
def test_310p_preserves_pre_noncontiguous_cache_geometry(block_size, hybrid):
    config = _config(block_size, hybrid=hybrid)
    original_page_size = config.cache_config.mamba_page_size_padded

    def reselect_and_align(cfg):
        cfg.cache_config.block_size = 64
        cfg.cache_config.mamba_page_size_padded = 64 * 2048

    with (
        patch(
            "vllm_ascend.platform.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType._310P),
        ),
        patch.object(Platform, "update_block_size_for_backend", side_effect=reselect_and_align) as upstream,
    ):
        NPUPlatform.update_block_size_for_backend(config)

    upstream.assert_not_called()
    assert config.cache_config.block_size == block_size
    assert config.cache_config.mamba_page_size_padded == original_page_size
    assert config.cache_config.mamba_block_size == 8192


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
def test_other_devices_keep_upstream_cache_geometry_update(device):
    config = _config(128)

    def reselect(cfg):
        cfg.cache_config.block_size = 64

    with (
        patch("vllm_ascend.platform.get_current_hardware_profile", return_value=get_hardware_profile(device)),
        patch.object(Platform, "update_block_size_for_backend", side_effect=reselect) as upstream,
    ):
        NPUPlatform.update_block_size_for_backend(config)

    upstream.assert_called_once_with(config)
    assert config.cache_config.block_size == 64


@pytest.mark.parametrize("prefix", [False, True])
def test_310p_keeps_existing_hybrid_transfer_alignment(prefix):
    config = _config(1024, hybrid=True, prefix=prefix, transfer=True)

    with (
        patch(
            "vllm_ascend.platform.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType._310P),
        ),
        patch.object(Platform, "update_block_size_for_backend") as upstream,
    ):
        NPUPlatform.update_block_size_for_backend(config)

    upstream.assert_not_called()
    assert config.cache_config.block_size == 1024
    assert config.cache_config.mamba_page_size_padded == 1024 * 2048
    assert config.cache_config.mamba_block_size == (8192 if prefix else 1024)
