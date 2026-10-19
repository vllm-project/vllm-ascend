# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""The physical block size DeepSeek V4 builds its pages with must survive selection.

``Platform.update_block_size_for_backend`` derives ``cache_config.block_size`` from
``get_supported_kernel_block_sizes()``. For DeepSeek V4 the compressed-cache backends
answer that in logical (raw scheduler token) units -- ``AscendDSAC128Backend`` says
4096 for the same physical page -- so the derived value cannot be used as the
physical size the page-padding tables are keyed on.

Two outcomes are tested: a settled size the tables know is kept, and one they do not
know is reported as a message naming the sizes that work.
"""

from types import SimpleNamespace

import pytest
from vllm.platforms.interface import Platform

from vllm_ascend.platform import NPUPlatform


def _config(model_type: str, block_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=True,
            mamba_cache_mode=None,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        kv_transfer_config=None,
    )


@pytest.fixture(autouse=True)
def _not_c8_mxfp(monkeypatch):
    """C8 MXFP keys its layout on its own kernel block size; see its own test."""
    monkeypatch.setattr("vllm_ascend.utils.is_c8_mxfp_kv_quant", lambda _config: False)


@pytest.fixture
def derived_block_size(monkeypatch):
    """Stand in for the upstream hook, writing what the C128 backend reports."""

    def write_derived_size(cls, vllm_config):
        vllm_config.cache_config.block_size = 4096

    monkeypatch.setattr(
        Platform,
        "update_block_size_for_backend",
        classmethod(write_derived_size),
    )


@pytest.mark.usefixtures("derived_block_size")
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_deepseek_v4_keeps_the_physical_block_size(block_size):
    config = _config("deepseek_v4", block_size)

    NPUPlatform.update_block_size_for_backend(config)

    assert config.cache_config.block_size == block_size


@pytest.mark.usefixtures("derived_block_size")
def test_other_models_keep_what_the_backend_selected():
    config = _config("qwen3_moe", 32)

    NPUPlatform.update_block_size_for_backend(config)

    assert config.cache_config.block_size == 4096


@pytest.mark.parametrize("block_size", [16, 256, 512])
def test_deepseek_v4_rejects_a_size_it_has_no_pages_for(block_size):
    """Name the sizes that work, instead of letting compressor.py raise KeyError."""
    config = _config("deepseek_v4", block_size)

    with pytest.raises(ValueError, match="Pass --block-size one of: 32, 64, 128"):
        NPUPlatform.update_block_size_for_backend(config)


@pytest.mark.usefixtures("derived_block_size")
def test_c8_mxfp_is_left_alone(monkeypatch):
    """C8 MXFP keys its layout on its own kernel block size, not on the DSV4 table."""
    monkeypatch.setattr("vllm_ascend.utils.is_c8_mxfp_kv_quant", lambda _config: True)
    config = _config("deepseek_v4", 512)

    NPUPlatform.update_block_size_for_backend(config)

    # Neither rejected nor rewritten: the DSV4 table is not the authority here.
    assert config.cache_config.block_size == 4096
