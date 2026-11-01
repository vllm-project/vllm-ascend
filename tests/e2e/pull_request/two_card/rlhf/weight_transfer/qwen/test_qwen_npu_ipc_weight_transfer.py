# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Qwen NPU_IPC weight-transfer coverage, scheduled and timed independently."""

import pytest

from tests.e2e.pull_request.rlhf.weight_transfer_models.base import WeightUpdateModelCase
from tests.e2e.pull_request.rlhf.weight_transfer_models.qwen import QWEN_CASES
from tests.e2e.pull_request.rlhf.weight_transfer_models.registry import pytest_model_cases
from tests.e2e.pull_request.rlhf.weight_transfer_runners.npu_ipc import run_npu_ipc_weight_transfer_transaction


@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("npu_ipc", model_cases=QWEN_CASES))
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    run_npu_ipc_weight_transfer_transaction(case, packed)
