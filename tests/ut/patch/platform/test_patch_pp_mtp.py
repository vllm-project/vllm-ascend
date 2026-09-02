# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from transformers import PretrainedConfig
from vllm.config.model import ModelConfig

from vllm_ascend.patch.platform.patch_speculative_config import hf_config_override


@pytest.mark.parametrize(
    ("model_type", "expected_architecture"),
    [("qwen3_5_text", "Qwen3_5MTP"), ("qwen3_5_moe_text", "Qwen3_5MoeMTP")],
)
def test_qwen3_5_text_config_mtp_override(model_type, expected_architecture):
    hf_config = PretrainedConfig()
    hf_config.model_type = model_type
    hf_config.architectures = ["Qwen3_5MoeForCausalLM" if "moe" in model_type else "Qwen3_5ForCausalLM"]
    hf_config.mtp_num_hidden_layers = 1
    result = hf_config_override(hf_config)
    assert result.model_type == "qwen3_5_mtp"
    assert result.n_predict == 1
    assert result.architectures == [expected_architecture]


def test_model_config_validates_local_mtp_drafter_as_single_pp_rank(monkeypatch):
    fake_registry = SimpleNamespace(
        is_pp_supported_model=lambda _architectures, _model_config: False,
    )
    monkeypatch.setattr(ModelConfig, "registry", property(lambda _self: fake_registry))

    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type="qwen3_5_mtp")
    model_config.runner = "draft"
    model_config.model_arch_config = SimpleNamespace(
        total_num_attention_heads=1,
        architectures=["Qwen3_5MTP"],
    )
    model_config.multimodal_config = None

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=2,
        decode_context_parallel_size=1,
    )

    ModelConfig.verify_with_parallel_config(model_config, parallel_config)
    assert parallel_config.pipeline_parallel_size == 2


def test_model_config_keeps_target_model_pp_validation(monkeypatch):
    fake_registry = SimpleNamespace(
        is_pp_supported_model=lambda _architectures, _model_config: False,
    )
    monkeypatch.setattr(ModelConfig, "registry", property(lambda _self: fake_registry))

    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type="qwen3_5_mtp")
    model_config.runner = "generate"
    model_config.model_arch_config = SimpleNamespace(
        total_num_attention_heads=1,
        architectures=["UnsupportedForPP"],
    )

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=2,
        decode_context_parallel_size=1,
    )

    with pytest.raises(NotImplementedError):
        ModelConfig.verify_with_parallel_config(model_config, parallel_config)
