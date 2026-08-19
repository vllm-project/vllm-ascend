from transformers import Qwen3Config
from vllm.config import SpeculativeConfig

from vllm_ascend.patch.platform.patch_speculative_config import hf_config_override
from vllm_ascend.transformers_utils.configs.kimi_k3 import K3DSparkConfig


def make_k3_dspark_config() -> K3DSparkConfig:
    return K3DSparkConfig(
        architectures=["K3DSparkModel"],
        hidden_size=7168,
        intermediate_size=14336,
        kv_lora_rank=512,
        markov_rank=256,
        num_attention_heads=64,
        num_hidden_layers=5,
        num_target_layers=5,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "yarn", "factor": 32},
        target_hidden_size=7168,
        v_head_dim=128,
        vocab_size=163840,
    )


def test_k3_dspark_config_with_required_fields_supports_repr():
    config = make_k3_dspark_config()

    assert K3DSparkConfig.has_no_defaults_at_init is True
    assert "K3DSparkConfig" in repr(config)
    assert config.to_diff_dict()["hidden_size"] == 7168


def test_legacy_qwen3_dspark_config_is_normalized_before_model_inspection():
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        block_size=7,
        dflash_config={
            "mask_token_id": 163824,
            "target_layer_ids": [7, 23, 51, 67, 83],
        },
        pad_token_id=163839,
    )

    normalized = hf_config_override(config)

    assert SpeculativeConfig.hf_config_override is hf_config_override
    assert normalized is config
    assert normalized.architectures == ["Qwen3DSparkModel"]
    assert normalized.mask_token_id == 163824
    assert normalized.target_layer_ids == [7, 23, 51, 67, 83]
    assert normalized.block_size == 7
    assert normalized.pad_token_id == 163839
