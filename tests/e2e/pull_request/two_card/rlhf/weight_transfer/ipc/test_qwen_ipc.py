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


@pytest.mark.parametrize(
    ("case", "packed", "reload_mode"),
    [
        pytest.param(*parameter.values, mode, marks=parameter.marks, id=f"{parameter.id}-{mode}")
        for mode in ("layerwise", "direct")
        for parameter in pytest_model_cases("npu_ipc", model_cases=QWEN_CASES, reload_mode=mode)
    ],
)
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool, reload_mode: str):
    run_npu_ipc_weight_transfer_transaction(case, packed, reload_mode=reload_mode, cold_start=reload_mode == "direct")
