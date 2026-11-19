# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Qwen HCCL weight-transfer coverage, scheduled and timed independently."""

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import (
    QWEN_CASES,
    WeightUpdateModelCase,
    pytest_model_cases,
)
from tests.e2e.common.rlhf.weight_transfer.hccl import (
    run_hccl_weight_transfer_transaction,
)


@pytest.mark.parametrize(
    ("case", "packed", "reload_mode"),
    [
        pytest.param(*parameter.values, mode, marks=parameter.marks, id=f"{parameter.id}-{mode}")
        for parameter in pytest_model_cases("hccl", model_cases=QWEN_CASES)
        # Keep the existing packed skip once; only expand the working unpacked lane.
        for mode in (("layerwise",) if parameter.values[1] else ("layerwise", "direct"))
    ],
)
def test_hccl_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool, reload_mode: str):
    run_hccl_weight_transfer_transaction(case, packed, reload_mode=reload_mode)
