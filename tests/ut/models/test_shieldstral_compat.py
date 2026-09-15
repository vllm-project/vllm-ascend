# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.models.shieldstral import (
    _get_mistral3_text_architectures,
    _prepare_llama4_scaling,
)


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("mistral", "MistralForCausalLM"),
        ("ministral3", "Ministral3ForCausalLM"),
    ],
)
def test_shieldstral_nested_text_architecture(
    model_type: str,
    architecture: str,
) -> None:
    config = SimpleNamespace(model_type=model_type)
    assert _get_mistral3_text_architectures(config) == [architecture]


def test_mistral4_is_not_registered_by_shieldstral_branch() -> None:
    with pytest.raises(ValueError, match="Unsupported Shieldstral"):
        _get_mistral3_text_architectures(
            SimpleNamespace(model_type="mistral4")
        )


def test_shieldstral_llama4_scaling_is_normalized() -> None:
    config = SimpleNamespace(
        rope_parameters={
            "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": 16384,
        }
    )
    _prepare_llama4_scaling(config)
    assert config.llama_4_scaling == {
        "beta": 0.1,
        "original_max_position_embeddings": 16384,
    }


def test_shieldstral_scaling_requires_original_context() -> None:
    config = SimpleNamespace(
        rope_parameters={"llama_4_scaling_beta": 0.1}
    )
    with pytest.raises(ValueError, match="original_max_position_embeddings"):
        _prepare_llama4_scaling(config)
