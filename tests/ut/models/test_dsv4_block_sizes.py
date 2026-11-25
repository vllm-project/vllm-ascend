# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend import utils
from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.models.layer.attention import layer
from vllm_ascend.utils import AscendDeviceType


def test_a3_block16_compressor_pages_fit(monkeypatch):
    monkeypatch.setattr(layer, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    blocks, pages = layer.get_dsv4_block_sizes()[16]
    mla, swa, c4_state, c128_state = blocks
    assert mla == swa == 16
    assert c4_state == 1
    assert c128_state == 4
    # float32 compressor states: C4 indexer uses 512 values; C128 uses 1024.
    assert pages[0] >= c4_state * 512 * 4
    assert pages[1] >= c128_state * 1024 * 4
    # C4 int8 index KV + fp16 scale is the padded page shared with its state cache.
    assert pages[0] == mla * (128 + 2)
    assert pages[1] == c128_state * 1024 * 4


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_existing_a3_layouts_unchanged(monkeypatch, block_size):
    monkeypatch.setattr(layer, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    blocks, pages = layer.get_dsv4_block_sizes()[block_size]
    assert blocks == [block_size, block_size, block_size // 16, block_size // 4]
    assert pages == [block_size * 130, block_size * 1024]


def test_a5_layout_is_not_extended(monkeypatch):
    monkeypatch.setattr(layer, "get_ascend_device_type", lambda: AscendDeviceType.A5)
    assert set(layer.get_dsv4_block_sizes()) == {32, 64, 128}


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_block16_cache_groups_have_supported_kernel_sizes(block_size):
    # C4 state, C128 state, and SWA/MLA each pass through kernel selection.
    assert select_common_block_size(block_size, [AscendDSABackend]) == block_size


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("requested", [8, 16, 32, 64, 128])
def test_dsv4_platform_preserves_supported_block_size(monkeypatch, device, requested):
    monkeypatch.setattr(utils, "get_ascend_device_type", lambda: device)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=requested),
        scheduler_config=SimpleNamespace(),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v4")),
    )
    utils.refresh_block_size(config)
    supported = requested in (16, 32, 64, 128)
    assert config.cache_config.block_size == (requested if supported else 32)
