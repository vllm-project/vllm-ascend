# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.deepseek_v4.dspark import DSparkDeepseekV4ForCausalLM
from vllm_ascend.models.deepseek_v41.dspark import DSparkDeepseekV41ForCausalLM


@pytest.mark.parametrize(
    "model_cls",
    [DSparkDeepseekV4ForCausalLM, DSparkDeepseekV41ForCausalLM],
)
def test_draft_has_token_id_mapping_attribute(model_cls):
    # vLLM's DSparkSpeculator.load_draft_model directly reads
    # model.draft_id_to_target_id, so the attribute must exist.
    # Check instance attribute lookup without allocating model weights.
    draft = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(draft)
    assert hasattr(draft, "draft_id_to_target_id")
