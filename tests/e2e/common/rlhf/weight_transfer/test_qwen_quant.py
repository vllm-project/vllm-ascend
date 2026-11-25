# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen35 sender-side MXFP8 export and HCCL refit equivalence."""

import pytest

from tests.e2e.common.rlhf.weight_transfer.case import QWEN35_W8A8_MXFP_CASE
from tests.e2e.common.rlhf.weight_transfer.hccl import run_hccl_weight_transfer_transaction


@pytest.mark.e2e_model(QWEN35_W8A8_MXFP_CASE.model)
def test_qwen_quant():
    # Packed HCCL transports opaque bytes and preserves FP8 weight/scale dtypes.
    run_hccl_weight_transfer_transaction(QWEN35_W8A8_MXFP_CASE, packed=True)
