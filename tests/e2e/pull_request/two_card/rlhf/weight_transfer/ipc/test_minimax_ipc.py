# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MiniMax IPC weight-transfer coverage, scheduled and timed independently."""

import pytest

from tests.e2e.pull_request.rlhf.weight_transfer.model import WeightUpdateModelCase
from tests.e2e.pull_request.two_card.rlhf.weight_transfer.cases import (
    MINIMAX_CASES,
    pytest_model_cases,
)
from tests.e2e.pull_request.two_card.rlhf.weight_transfer.ipc.runner import (
    run_npu_ipc_weight_transfer_transaction,
)


@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("npu_ipc", model_cases=MINIMAX_CASES))
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    run_npu_ipc_weight_transfer_transaction(case, packed)
