# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.config.model import ModelConfig

from vllm_ascend.patch.platform.patch_pp_mtp import (
    _apply_patch,
    _patch_model_config_validation,
)


@pytest.mark.parametrize(
    "model_type,architecture",
    [
        ("qwen3_5_mtp", "Qwen3_5MTP"),
        ("qwen3", "DSparkDraftModel"),
        ("qwen3", "Qwen3DSparkModel"),
        ("kimi_k3", "K3DSparkModel"),
    ],
)
def test_model_config_validates_local_drafter_as_single_pp_rank(monkeypatch, model_type, architecture):
    fake_registry = SimpleNamespace(
        is_pp_supported_model=lambda _architectures, _model_config: False,
    )
    monkeypatch.setattr(ModelConfig, "registry", property(lambda _self: fake_registry))

    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = SimpleNamespace(model_type=model_type)
    model_config.runner = "draft"
    model_config.model_arch_config = SimpleNamespace(
        total_num_attention_heads=1,
        architectures=[architecture],
    )
    model_config.multimodal_config = None

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=2,
        decode_context_parallel_size=1,
        # Both supported vLLM lanes read this during model/parallel validation.
        enable_hc_sp=False,
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
        # Both supported vLLM lanes read this during model/parallel validation.
        enable_hc_sp=False,
    )

    with pytest.raises(NotImplementedError):
        ModelConfig.verify_with_parallel_config(model_config, parallel_config)


def test_pp_mtp_patch_reentry():
    _apply_patch()
    _patch_model_config_validation()
