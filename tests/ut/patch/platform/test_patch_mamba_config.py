from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.patch.platform.patch_mamba_config import verify_and_update_config


def test_kimi_k3_hybrid_c8_doubles_attention_block_size():
    cache_config = SimpleNamespace(
        cache_dtype="auto",
        block_size=None,
        mamba_page_size_padded=None,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        mamba_block_size=None,
    )
    text_config = SimpleNamespace(
        model_type="kimi_linear",
        mla_use_output_gate=True,
        routed_expert_hidden_size=1024,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    model_config = SimpleNamespace(
        use_mla=True,
        dtype=torch.bfloat16,
        architecture="KimiK3ForCausalLM",
        hf_config=SimpleNamespace(model_type="kimi_k3"),
        hf_text_config=text_config,
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
        quant_config=SimpleNamespace(
            enable_fa_quant=True,
            # This is the value used by the Kimi-K3 W4A8C8 checkpoint.
            quant_description={"fa_quant_type": "FAKQuant"},
        ),
    )
    model_cls = SimpleNamespace(
        get_mamba_state_shape_from_config=MagicMock(
            return_value=((2304, 3), (6, 128, 128))
        ),
        get_mamba_state_dtype_from_config=MagicMock(
            return_value=(torch.bfloat16, torch.float32)
        ),
    )

    with (
        patch(
            "vllm_ascend.patch.platform.patch_mamba_config."
            "MambaModelConfig.verify_and_update_config"
        ),
        patch(
            "vllm_ascend.patch.platform.patch_mamba_config."
            "ModelRegistry.resolve_model_cls",
            return_value=(model_cls, None),
        ),
    ):
        verify_and_update_config.__func__(None, vllm_config)

    assert cache_config.block_size == 768
    assert cache_config.mamba_page_size_padded == 505344
    assert cache_config.mamba_block_size == 768
