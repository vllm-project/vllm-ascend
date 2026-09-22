# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

import pytest

from vllm_ascend import models


def _registered_architectures(version: str) -> list[str]:
    architectures: list[str] = []
    with (
        patch.object(models, "vllm_version_is", side_effect=lambda target: target == version),
        patch.object(models.ModelRegistry, "register_model", side_effect=lambda name, _: architectures.append(name)),
    ):
        models.register_model()
    return architectures


@pytest.mark.parametrize("version", ["0.29.0", "0.30.0"])
def test_deepseek_v41_models_are_not_registered_on_release(version):
    architectures = _registered_architectures(version)

    assert "DeepseekV41ForCausalLM" not in architectures
    assert "DeepseekV41DSparkModel" not in architectures
    assert "DeepseekV4ForCausalLM" in architectures
    assert "DSparkDraftModel" in architectures


def test_deepseek_v41_models_are_registered_on_main():
    architectures = _registered_architectures("main")

    assert "DeepseekV41ForCausalLM" in architectures
    assert "DeepseekV41DSparkModel" in architectures
