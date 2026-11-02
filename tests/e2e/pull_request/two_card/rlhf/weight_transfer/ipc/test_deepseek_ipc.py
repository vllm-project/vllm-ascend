# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DeepSeek IPC weight-transfer coverage, scheduled and timed independently."""

import pytest

from tests.e2e.pull_request.two_card.rlhf.weight_transfer.common.cases import (
    DEEPSEEK_CASES,
    WeightUpdateModelCase,
    pytest_model_cases,
)
from tests.e2e.pull_request.two_card.rlhf.weight_transfer.ipc.runner import (
    run_npu_ipc_weight_transfer_transaction,
)


@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("npu_ipc", model_cases=DEEPSEEK_CASES))
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    run_npu_ipc_weight_transfer_transaction(case, packed)
