from types import SimpleNamespace

import pytest
from transformers import Qwen3Config
from vllm.config import SpeculativeConfig

from vllm_ascend.patch.platform import patch_speculative_config
from vllm_ascend.patch.platform.patch_speculative_config import (
    _dspark_post_init,
    hf_config_override,
)
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


def _make_qwen3_vl_dspark_config(num_speculative_tokens: int):
    draft_hf_config = SimpleNamespace(
        model_type="qwen3_vl_dflash",
        architectures=["Qwen3VLDSparkModel"],
        block_size=16,
        mask_token_id=151669,
        ptd_token_id=None,
    )
    config = SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        num_speculative_tokens=num_speculative_tokens,
    )
    return config, draft_hf_config


@pytest.mark.parametrize("num_speculative_tokens", [1, 15])
def test_qwen3_vl_dspark_accepts_tokens_up_to_checkpoint_block_size(
    monkeypatch: pytest.MonkeyPatch,
    num_speculative_tokens: int,
):
    monkeypatch.setattr(patch_speculative_config, "_orig_post_init", lambda self: None)
    config, draft_hf_config = _make_qwen3_vl_dspark_config(num_speculative_tokens)

    _dspark_post_init(config)

    assert config.num_speculative_tokens == num_speculative_tokens
    assert draft_hf_config.ptd_token_id == 151669


def test_qwen3_vl_dspark_rejects_tokens_above_checkpoint_block_size(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(patch_speculative_config, "_orig_post_init", lambda self: None)
    config, _ = _make_qwen3_vl_dspark_config(17)

    with pytest.raises(ValueError, match=r"no greater than the trained block_size"):
        _dspark_post_init(config)


def test_qwen3_vl_dspark_caps_tokens_to_fia_tnd_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(patch_speculative_config, "_orig_post_init", lambda self: None)
    config, draft_hf_config = _make_qwen3_vl_dspark_config(16)

    _dspark_post_init(config)

    assert config.num_speculative_tokens == 15
    assert draft_hf_config.ptd_token_id == 151669
