# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Independent draft HCCL weight-update entry point."""

import pytest
import torch

from tests.e2e.rlhf.weight_transfer.case import PACKED_LANE_SKIP_REASON
from tests.e2e.rlhf.weight_transfer.draft_hccl import run_draft_hccl_weight_transfer


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="HCCL draft updates require two NPUs.")
@pytest.mark.parametrize(
    "packed",
    [False, pytest.param(True, marks=pytest.mark.skip(reason=PACKED_LANE_SKIP_REASON))],
    ids=["unpacked", "packed"],
)
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "aclgraph"])
def test_hccl_draft_then_target_update(tmp_path, monkeypatch, packed, enforce_eager):
    run_draft_hccl_weight_transfer(tmp_path, monkeypatch, packed, enforce_eager)
