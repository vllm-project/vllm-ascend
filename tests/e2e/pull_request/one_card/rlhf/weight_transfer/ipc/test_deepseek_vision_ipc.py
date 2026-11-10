# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.common.rlhf.weight_transfer.case import DEEPSEEK_V4_VISION_CASE
from tests.e2e.common.rlhf.weight_transfer.single_card_ipc import run_single_card_ipc

CASE = DEEPSEEK_V4_VISION_CASE
DEVICE_INDEX = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))
MEMORY_UTILIZATION = float(os.environ.get("VLLM_RL_DSV4_MEM_UTIL", "0.45"))


@pytest.mark.e2e_model(CASE.model)
@pytest.mark.skipif(DEVICE_INDEX < 0 or torch.npu.device_count() <= DEVICE_INDEX, reason="Requires one visible NPU")
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_deepseek_vision_ipc(packed):
    run_single_card_ipc(CASE, packed, device_index=DEVICE_INDEX, gpu_memory_utilization=MEMORY_UTILIZATION)
