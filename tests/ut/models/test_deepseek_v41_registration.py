# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

from vllm_ascend import models


def _registered_architectures() -> list[str]:
    architectures: list[str] = []
    with patch.object(
        models.ModelRegistry,
        "register_model",
        side_effect=lambda name, _: architectures.append(name),
    ):
        models.register_model()
    return architectures


def test_deepseek_v41_models_are_registered_on_vllm_029():
    architectures = _registered_architectures()

    assert "DeepseekV41ForCausalLM" in architectures
    assert "DeepseekV41DSparkModel" in architectures
    assert "DeepseekV4ForCausalLM" in architectures
    assert "DSparkDraftModel" in architectures
