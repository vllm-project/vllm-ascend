# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DeepSeek-V4 vision IPC coverage using the shared TP2 transaction."""

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import (
    DEEPSEEK_V4_VISION_CASE,
    WeightUpdateModelCase,
    pytest_model_cases,
)
from tests.e2e.common.rlhf.weight_transfer.npu_ipc import (
    run_npu_ipc_weight_transfer_transaction,
)


@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("npu_ipc", model_cases=(DEEPSEEK_V4_VISION_CASE,)))
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    run_npu_ipc_weight_transfer_transaction(case, packed)
