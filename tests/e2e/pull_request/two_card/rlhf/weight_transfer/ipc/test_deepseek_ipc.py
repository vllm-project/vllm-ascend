# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DeepSeek-V4.1 Engram host-offload over unpacked NPU IPC, timed independently.

Scope: the host/UVA Engram table across ``sleep(level=2)`` -> ``wake_up`` ->
weight transfer -> ``wake_up(kv_cache)``, and the value update of the non-Engram
parameters.  An Engram *value* update is deliberately out of scope: the loader
streams the rows from the bound indexed shard instead of consuming the transferred
tensor, so this lane cannot distinguish a landed update from a disk re-read.  See
the scope note above ``DEEPSEEK_V41_ENGRAM_CASE``.
"""

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import (
    DEEPSEEK_V41_CASES,
    WeightUpdateModelCase,
    pytest_model_cases,
)
from tests.e2e.common.rlhf.weight_transfer.npu_ipc import (
    run_npu_ipc_weight_transfer_transaction,
)


@pytest.mark.parametrize(("case", "packed"), pytest_model_cases("npu_ipc", model_cases=DEEPSEEK_V41_CASES))
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    run_npu_ipc_weight_transfer_transaction(case, packed)
