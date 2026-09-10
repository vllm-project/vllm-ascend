# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.patch.platform.patch_mamba_config import verify_and_update_config


class _FakeHybridModel:
    @staticmethod
    def get_mamba_state_shape_from_config(_vllm_config):
        # Qwen3.5-style conv and SSM states.
        return ((3, 6144), (16, 128, 128))

    @staticmethod
    def get_mamba_state_dtype_from_config(_vllm_config):
        return (torch.bfloat16, torch.float32)


@pytest.mark.parametrize("requested_block_size", [512, 1024, 2048])
def test_block_size_is_aligned_for_hybrid_cache(monkeypatch, requested_block_size):
    model_config = SimpleNamespace(
        architecture="FakeHybridModel",
        dtype=torch.bfloat16,
        use_mla=False,
        max_model_len=16384,
        get_num_kv_heads=lambda _parallel_config: 2,
        get_head_size=lambda: 256,
    )

    cache_config = SimpleNamespace(
        cache_dtype="auto",
        block_size=requested_block_size,
        mamba_page_size_padded=None,
        mamba_cache_mode="none",
        enable_prefix_caching=False,
    )

    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )

    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_mamba_config.MambaModelConfig.verify_and_update_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_mamba_config.ModelRegistry.resolve_model_cls",
        lambda *_args, **_kwargs: (_FakeHybridModel, None),
    )

    verify_and_update_config.__func__(None, vllm_config)

    # The Qwen3.5-style layout above requires a 1024-token attention block:
    # 1024 tokens * 2 KV heads * 256 head size * 2 bytes = 1 MiB,
    # exactly matching one SSM state block.
    assert cache_config.block_size == 1024
