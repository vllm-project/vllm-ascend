from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from transformers import Qwen3Config
from vllm.config.model_arch import ModelArchitectureConfig
from vllm.config.speculative import SpeculativeConfig

from vllm_ascend.patch.platform import patch_speculative_config
from vllm_ascend.patch.platform.patch_speculative_config import (
    _normalize_deepseek_v4_dspark_draft,
)


def test_legacy_qwen3_dspark_config_uses_qwen3_loader():
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        block_size=7,
        dflash_config={
            "mask_token_id": 163824,
            "target_layer_ids": [7, 23, 51, 67, 83],
        },
    )

    normalized = SpeculativeConfig.hf_config_override(config)

    assert normalized is config
    assert normalized.architectures == ["Qwen3DSparkModel"]
    assert normalized.mask_token_id == 163824
    assert normalized.target_layer_ids == [7, 23, 51, 67, 83]
    assert normalized.block_size == 7


def test_deepseek_v4_vision_dspark_restores_draft_architecture():
    hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForConditionalGeneration"],
        dspark_target_layer_ids=[40, 41, 42],
    )
    hf_config.update = lambda values: hf_config.__dict__.update(values)
    model_arch_config = ModelArchitectureConfig(
        architectures=["DeepseekV4ForConditionalGeneration"],
        model_type="deepseek_v4",
        text_model_type=None,
        hidden_size=128,
        total_num_hidden_layers=43,
        total_num_attention_heads=8,
        head_size=16,
        vocab_size=1024,
        total_num_kv_heads=8,
        num_experts=256,
        num_experts_per_token=8,
        quantization_config=None,
        is_deepseek_mla=True,
        is_mm_prefix_lm=True,
        rswa_window=128,
        derived_max_model_len_and_key=(8192, "max_position_embeddings"),
    )
    registry = MagicMock()
    registry.inspect_model_cls.return_value = ("model-info", "DSparkDraftModel")

    class DraftModelConfig(SimpleNamespace):
        @property
        def architectures(self):
            return self.model_arch_config.architectures

    draft_model_config = DraftModelConfig(
        hf_config=hf_config,
        model_arch_config=model_arch_config,
        registry=registry,
    )

    _normalize_deepseek_v4_dspark_draft(draft_model_config)

    assert hf_config.architectures == ["DSparkDraftModel"]
    assert draft_model_config.architectures == ["DSparkDraftModel"]
    assert draft_model_config.model_arch_config.is_mm_prefix_lm is False
    assert draft_model_config._architecture == "DSparkDraftModel"
    registry.inspect_model_cls.assert_called_once_with(["DSparkDraftModel"], draft_model_config)


def _make_k3_dspark_config(
    dcp_size: int = 8,
    method: str = "dspark",
):
    draft_hf_config = SimpleNamespace(
        ptd_token_id=163839,
        dspark_noise_token_id=163839,
        mask_token_id=None,
    )
    return SimpleNamespace(
        method=method,
        target_parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
        ),
        draft_model_config=SimpleNamespace(
            architectures=["K3DSparkModel"],
            hf_config=draft_hf_config,
        ),
        use_dspark=lambda: method == "dspark",
    )


def test_k3_dspark_dcp_is_hidden_only_during_upstream_validation(monkeypatch):
    config = _make_k3_dspark_config()
    original_parallel_config = config.target_parallel_config
    validated_parallel_configs = []

    def validate_config(validated_config):
        validated_parallel_configs.append(validated_config.target_parallel_config)
        assert validated_config.target_parallel_config.decode_context_parallel_size == 1

    monkeypatch.setattr(
        patch_speculative_config,
        "_orig_post_init",
        validate_config,
    )

    patch_speculative_config._dspark_post_init(config)

    assert len(validated_parallel_configs) == 1
    assert validated_parallel_configs[0] is not original_parallel_config
    assert config.target_parallel_config is original_parallel_config
    assert config.target_parallel_config.decode_context_parallel_size == 8


def test_k3_dspark_dcp_restores_parallel_config_when_validation_fails(
    monkeypatch,
):
    config = _make_k3_dspark_config()
    original_parallel_config = config.target_parallel_config

    def raise_validation_error(validated_config):
        assert validated_config.target_parallel_config is not original_parallel_config
        assert validated_config.target_parallel_config.decode_context_parallel_size == 1
        raise ValueError("some other speculative config error")

    monkeypatch.setattr(
        patch_speculative_config,
        "_orig_post_init",
        raise_validation_error,
    )

    with pytest.raises(ValueError, match="some other speculative config error"):
        patch_speculative_config._dspark_post_init(config)

    assert config.target_parallel_config is original_parallel_config
    assert config.target_parallel_config.decode_context_parallel_size == 8


@pytest.mark.parametrize(
    ("config", "expected_dcp_size"),
    [
        (_make_k3_dspark_config(dcp_size=1), 1),
        (_make_k3_dspark_config(method="eagle"), 8),
    ],
)
def test_non_dcp_dspark_config_is_not_replaced_during_validation(
    monkeypatch,
    config,
    expected_dcp_size,
):
    original_parallel_config = config.target_parallel_config

    def validate_config(validated_config):
        assert validated_config.target_parallel_config is original_parallel_config
        assert validated_config.target_parallel_config.decode_context_parallel_size == expected_dcp_size

    monkeypatch.setattr(
        patch_speculative_config,
        "_orig_post_init",
        validate_config,
    )

    patch_speculative_config._dspark_post_init(config)
    assert config.target_parallel_config is original_parallel_config
