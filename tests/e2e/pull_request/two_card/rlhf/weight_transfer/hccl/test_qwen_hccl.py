# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Qwen HCCL weight-transfer coverage, scheduled and timed independently."""

import pytest

from tests.e2e.pull_request.two_card.rlhf.common.weight_transfer.models.base import WeightUpdateModelCase
from tests.e2e.pull_request.two_card.rlhf.common.weight_transfer.models.qwen import QWEN_CASES
from tests.e2e.pull_request.two_card.rlhf.common.weight_transfer.models.registry import pytest_model_cases
from tests.e2e.pull_request.two_card.rlhf.common.weight_transfer.runners.hccl import (
    run_hccl_weight_transfer_transaction,
)


@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("hccl", model_cases=QWEN_CASES))
def test_hccl_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    run_hccl_weight_transfer_transaction(case, packed)
