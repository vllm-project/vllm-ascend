# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.patch.platform import patch_mamba_config
from vllm_ascend.utils import AscendDeviceType


class _MambaModel:
    @staticmethod
    def get_mamba_state_shape_from_config(_vllm_config):
        return [(786432,), (1,)]

    @staticmethod
    def get_mamba_state_dtype_from_config(_vllm_config):
        return [torch.uint8, torch.uint8]


@pytest.mark.parametrize(
    ("fa_quant_enabled", "expected_block_size", "expected_page_size"),
    [
        (False, 768, 884737),
        (True, 1536, 983041),
    ],
)
def test_mamba_alignment_uses_a5_c8_latent_k_storage_bytes(
    monkeypatch,
    fa_quant_enabled,
    expected_block_size,
    expected_page_size,
):
    monkeypatch.setattr(
        patch_mamba_config.MambaModelConfig,
        "verify_and_update_config",
        lambda _vllm_config: None,
    )
    monkeypatch.setattr(
        patch_mamba_config.ModelRegistry,
        "resolve_model_cls",
        lambda *_args, **_kwargs: (_MambaModel, None),
    )
    monkeypatch.setattr(
        patch_mamba_config,
        "get_ascend_device_type",
        lambda: AscendDeviceType.A5,
    )
    monkeypatch.setattr(
        patch_mamba_config,
        "enable_fa_quant",
        lambda _vllm_config: fa_quant_enabled,
    )

    cache_config = SimpleNamespace(
        cache_dtype="auto",
        block_size=None,
        mamba_page_size_padded=None,
        mamba_cache_mode="align",
        enable_prefix_caching=True,
        mamba_block_size=None,
    )
    model_config = SimpleNamespace(
        use_mla=True,
        dtype=torch.bfloat16,
        architecture="KimiK3ForConditionalGeneration",
        hf_text_config=SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        ),
        get_num_kv_heads=lambda _parallel_config: 1,
        max_model_len=200000,
    )
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=None,
        speculative_config=None,
    )

    patch_mamba_config.verify_and_update_config(None, vllm_config)

    assert cache_config.block_size == expected_block_size
    assert cache_config.mamba_block_size == expected_block_size
    assert cache_config.mamba_page_size_padded == expected_page_size
