# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform import patch_kv_cache_dtype


@pytest.mark.parametrize("model_type,expected", [("deepseek_v4", []), ("llama", ["0", "1", "6", "7"])])
def test_turboquant_boundary_skip_layers(model_type, expected):
    model_config = SimpleNamespace(
        is_hybrid=False,
        hf_text_config=SimpleNamespace(model_type=model_type, num_hidden_layers=8),
    )
    assert patch_kv_cache_dtype.TurboQuantConfig.get_boundary_skip_layers(model_config) == expected
