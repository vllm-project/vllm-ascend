# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Qwen HCCL weight-transfer coverage, scheduled and timed independently."""

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import (
    WeightUpdateModelCase,
    pytest_qwen_reload_cases,
)
from tests.e2e.common.rlhf.weight_transfer.hccl import (
    run_hccl_weight_transfer_transaction,
)


@pytest.mark.parametrize(("case", "packed", "reload_mode"), pytest_qwen_reload_cases("hccl"))
def test_hccl_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool, reload_mode: str):
    run_hccl_weight_transfer_transaction(case, packed, reload_mode=reload_mode)
