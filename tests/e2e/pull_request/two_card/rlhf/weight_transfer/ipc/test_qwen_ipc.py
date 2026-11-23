# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Qwen IPC weight-transfer coverage, scheduled and timed independently."""

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import (
    QWEN_CASES,
    WeightUpdateModelCase,
    pytest_model_cases,
)
from tests.e2e.common.rlhf.weight_transfer.npu_ipc import (
    run_npu_ipc_weight_transfer_transaction,
)


@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("npu_ipc", model_cases=QWEN_CASES))
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    run_npu_ipc_weight_transfer_transaction(case, packed)
