# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import ModelConfig

from tests.e2e.common.rlhf.weight_transfer.deepseek_vision import deepseek_vision_case
from tests.e2e.common.rlhf.weight_transfer.retention import run_vision_retention

CASE = deepseek_vision_case()
DEVICE_INDEX = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))
MEMORY_UTILIZATION = float(os.environ.get("VLLM_RL_DSV4_MEM_UTIL", "0.45"))


@pytest.mark.e2e_model(CASE.model)
@pytest.mark.skipif(DEVICE_INDEX < 0 or torch.npu.device_count() <= DEVICE_INDEX, reason="Requires one visible NPU")
@pytest.mark.skipif(
    "sleep_preserve_parameter_names" not in ModelConfig.__dataclass_fields__,
    reason="vLLM lacks sleep parameter retention",
)
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_deepseek_vision_retention(packed, tmp_path):
    run_vision_retention(CASE, packed, tmp_path, device_index=DEVICE_INDEX, gpu_memory_utilization=MEMORY_UTILIZATION)
