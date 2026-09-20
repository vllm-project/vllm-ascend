# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM

from vllm_ascend.models.deepseek_mtp import AscendGlmMoeDsaForCausalLM


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("pp_size,local_layers", [(1, 75), (2, 39), (2, 36), (2, 0), (3, 24)])
def test_eplb_layer_count_matches_local_weights_only_for_v2_pp(use_v2, pp_size, local_layers):
    config = SimpleNamespace(
        use_v2_model_runner=use_v2,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
    )
    original_count = 75 if local_layers else 0

    def init(self, *, vllm_config, prefix):
        nn.Module.__init__(self)
        assert vllm_config is config
        assert prefix == "target"
        self.num_moe_layers = original_count
        self.moe_layers = [object() for _ in range(local_layers)]

    with patch.object(GlmMoeDsaForCausalLM, "__init__", init):
        model = AscendGlmMoeDsaForCausalLM(vllm_config=config, prefix="target")

    expected = local_layers if use_v2 and pp_size > 1 else original_count
    assert model.num_moe_layers == expected
    assert len(model.moe_layers) == local_layers
