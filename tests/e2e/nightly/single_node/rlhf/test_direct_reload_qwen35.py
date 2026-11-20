# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 full text-backbone expansion after the Qwen3 transport matrix."""

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import DIRECT_QWEN35_CASE
from tests.e2e.common.rlhf.weight_transfer.direct import run_direct_reload_transaction


@pytest.mark.e2e_model(DIRECT_QWEN35_CASE.model)
@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
@pytest.mark.parametrize("backend", ["hccl", "npu_ipc"])
def test_direct_reload_qwen35(backend: str, packed: bool, execution_mode: str):
    run_direct_reload_transaction(DIRECT_QWEN35_CASE, backend, packed, execution_mode)
