# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Qwen HCCL weight-transfer coverage, scheduled and timed independently."""

import pytest
import torch

from tests.e2e.common.rlhf.weight_transfer.case import (
    PACKED_LANE_SKIP_REASON,
    QWEN3_DFLASH_CASE,
    QWEN3_EAGLE3_CASE,
    QWEN_CASES,
    WeightUpdateModelCase,
    pytest_model_cases,
)
from tests.e2e.common.rlhf.weight_transfer.hccl import (
    run_draft_hccl_weight_transfer,
    run_hccl_weight_transfer_transaction,
)


@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("hccl", model_cases=QWEN_CASES))
def test_hccl_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    run_hccl_weight_transfer_transaction(case, packed)


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="HCCL draft updates require two NPUs.")
@pytest.mark.parametrize(
    "packed",
    [False, pytest.param(True, marks=pytest.mark.skip(reason=PACKED_LANE_SKIP_REASON))],
    ids=["unpacked", "packed"],
)
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "aclgraph"])
@pytest.mark.parametrize("speculative_case", (QWEN3_EAGLE3_CASE, QWEN3_DFLASH_CASE), ids=lambda case: case.id)
def test_hccl_draft_then_target_update(tmp_path, monkeypatch, packed, enforce_eager, speculative_case):
    run_draft_hccl_weight_transfer(tmp_path, monkeypatch, packed, enforce_eager, speculative_case)
