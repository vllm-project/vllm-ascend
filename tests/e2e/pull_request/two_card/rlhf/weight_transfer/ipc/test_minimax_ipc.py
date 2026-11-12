# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MiniMax-M3 model coverage through unpacked TP2 IPC weight transfer."""

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import MINIMAX_M3_CASE
from tests.e2e.common.rlhf.weight_transfer.npu_ipc import (
    run_npu_ipc_weight_transfer_transaction,
)


@pytest.mark.e2e_model(MINIMAX_M3_CASE.model)
def test_minimax_m3_ipc_unpacked():
    run_npu_ipc_weight_transfer_transaction(MINIMAX_M3_CASE, packed=False)
