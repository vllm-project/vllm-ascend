# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare normal Kimi K3.1 Pro startup with two unpacked TP2 IPC updates."""

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import KIMI_K31_CASE, WeightUpdateModelCase, pytest_model_cases
from tests.e2e.common.rlhf.weight_transfer.npu_ipc import run_npu_ipc_weight_transfer_transaction


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="weight_transfer,sleep_wake",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("npu_ipc", model_cases=(KIMI_K31_CASE,)))
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    run_npu_ipc_weight_transfer_transaction(case, packed)
