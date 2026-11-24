# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DeepSeek draft HCCL weight-transfer coverage."""

import pytest
import torch

from tests.e2e.common.rlhf.weight_transfer.case import (
    DEEPSEEK_V4_DSPARK_CASE,
    DEEPSEEK_V4_MTP_CASE,
    PACKED_LANE_SKIP_REASON,
)
from tests.e2e.common.rlhf.weight_transfer.hccl import run_draft_hccl_weight_transfer


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="HCCL draft updates require two NPUs.")
@pytest.mark.parametrize(
    "packed",
    [False, pytest.param(True, marks=pytest.mark.skip(reason=PACKED_LANE_SKIP_REASON))],
    ids=["unpacked", "packed"],
)
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "aclgraph"])
@pytest.mark.parametrize("speculative_case", (DEEPSEEK_V4_MTP_CASE, DEEPSEEK_V4_DSPARK_CASE), ids=lambda case: case.id)
def test_hccl_draft_then_target_update(tmp_path, monkeypatch, packed, enforce_eager, speculative_case):
    run_draft_hccl_weight_transfer(tmp_path, monkeypatch, packed, enforce_eager, speculative_case)
