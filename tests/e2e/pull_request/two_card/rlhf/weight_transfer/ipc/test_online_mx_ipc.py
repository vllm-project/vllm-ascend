# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 IPC updates with receiver-side compressed-tensors MX quantization."""

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import QWEN35_CASE, QWEN35_ONLINE_MX_CASES
from tests.e2e.common.rlhf.weight_transfer.npu_ipc import run_npu_ipc_weight_transfer_transaction
from vllm_ascend.device.hardware_profile import WeightFormat, get_current_hardware_profile


@pytest.mark.parametrize("case", QWEN35_ONLINE_MX_CASES, ids=lambda case: case.id)
@pytest.mark.e2e_model(QWEN35_CASE.model)
def test_online_mx_ipc(case):
    if not get_current_hardware_profile().supports_weight_format(WeightFormat.MXFP8):
        pytest.skip("Native MXFP weight quantization and matmul require Ascend 950")
    # Packed lanes remain separately disabled in main pending #18763.
    run_npu_ipc_weight_transfer_transaction(case, packed=False)
