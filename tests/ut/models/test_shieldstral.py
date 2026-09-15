# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from pathlib import Path

import pytest
import torch
from transformers import AutoConfig


SHIELDSTRAL_PATH = Path("/mnt/weight/Shieldstral-1.0-3B")


@pytest.mark.skipif(
    not (SHIELDSTRAL_PATH / "config.json").is_file(),
    reason="Shieldstral checkpoint is not available",
)
def test_shieldstral_uses_supported_mistral3_bf16_path() -> None:
    config = AutoConfig.from_pretrained(
        SHIELDSTRAL_PATH,
        local_files_only=True,
    )

    assert config.architectures == ["Mistral3ForConditionalGeneration"]
    assert config.model_type == "mistral3"
    assert config.dtype == torch.bfloat16
    assert not hasattr(config, "quantization_config")
    assert config.text_config.model_type == "ministral3"
    assert config.text_config.rope_parameters["llama_4_scaling_beta"] == 0.1
    assert (
        config.text_config.rope_parameters[
            "original_max_position_embeddings"
        ]
        == 16384
    )
