# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

from vllm_ascend import models


def _registered_architectures(vllm_029: bool) -> list[str]:
    architectures: list[str] = []
    with (
        patch.object(models, "vllm_version_is", return_value=vllm_029),
        patch.object(models.ModelRegistry, "register_model", side_effect=lambda name, _: architectures.append(name)),
    ):
        models.register_model()
    return architectures


def test_deepseek_v41_models_are_not_registered_on_vllm_029():
    architectures = _registered_architectures(vllm_029=True)

    assert "DeepseekV41ForCausalLM" not in architectures
    assert "DeepseekV41DSparkModel" not in architectures
    assert "DeepseekV4ForCausalLM" in architectures
    assert "DSparkDraftModel" in architectures


def test_deepseek_v41_models_are_registered_after_vllm_029():
    architectures = _registered_architectures(vllm_029=False)

    assert "DeepseekV41ForCausalLM" in architectures
    assert "DeepseekV41DSparkModel" in architectures
