# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.fixture(autouse=True)
def lora_model_runner_env(model_runner_env: None) -> None:
    """Parametrize every LoRA test with the shared model runner fixture."""
    return model_runner_env
