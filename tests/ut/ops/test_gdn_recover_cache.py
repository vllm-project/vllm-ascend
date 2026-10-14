# SPDX-License-Identifier: Apache-2.0
"""Allocator-visible checkpoint/record fields and default-off compatibility."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.model_executor.layers.mamba import mamba_utils
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_ascend.ops import gdn


def _layer():
    layer: Any = SimpleNamespace(
        tp_size=1,
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        num_spec=3,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto", mamba_ssm_cache_dtype="float32"),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        is_kv_cache_tp_replicated=False,
    )
    layer.get_state_shape = lambda: gdn.AscendGatedDeltaNetAttention.get_state_shape(layer)
    layer.get_state_dtype = lambda: gdn.AscendGatedDeltaNetAttention.get_state_dtype(layer)
    return layer


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("padding", [None, 3_670_016])
def test_cache_accounts_records_and_state_reservations(monkeypatch, enabled, padding):
    monkeypatch.setattr(gdn, "is_gdn_recover_enabled", lambda: enabled)
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: False)
    layer = _layer()
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_block_size=896,
            mamba_page_size_padded=padding,
            mamba_cache_mode="none",
            use_kda_recoverssm=False,
        ),
        num_speculative_tokens=3,
    )
    spec = gdn.AscendGatedDeltaNetAttention.get_kv_cache_spec(layer, config)
    assert spec is not None
    assert spec.shapes[:2] == ((6, 10240), (48, 128, 128))
    assert spec.dtypes[:2] == (torch.bfloat16, torch.float32)
    if enabled:
        assert spec.shapes[2:] == ((48, 4, 257), (3, 10240))
        assert spec.dtypes[2:] == (torch.float32, torch.bfloat16)
        assert spec.num_speculative_blocks == 0
        assert spec.page_size_bytes >= spec.state_content_size_bytes == 3_527_424
        assert spec.page_size_bytes % 32 == 0
        assert spec.max_num_blocks_per_req(config, 4096) == 1
    else:
        assert len(spec.shapes) == 2 and spec.num_speculative_blocks == 3
        assert spec.page_size_padded == padding
        assert spec.max_num_blocks_per_req(config, 4096) == 4
    if padding is not None:
        assert spec.page_size_bytes == padding


def test_recover_rejects_incompatible_ssm_dtype(monkeypatch):
    monkeypatch.setattr(gdn, "is_gdn_recover_enabled", lambda: True)
    layer = _layer()
    layer.cache_config.mamba_ssm_cache_dtype = "bfloat16"
    with pytest.raises(ValueError, match="FP32"):
        layer.get_state_dtype()
