# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.routed_experts import _record_v2_eplb_load


def _router(record_enabled: bool = True):
    return SimpleNamespace(
        eplb_state=SimpleNamespace(
            expert_load_view=torch.zeros(8, dtype=torch.int32),
            should_record_tensor=torch.tensor(record_enabled),
            local_expert_start=4,
            local_expert_count=2,
        )
    )


def test_record_v2_eplb_load_uses_operator_counts():
    router = _router()
    counts = torch.tensor([2, 3], dtype=torch.int64)
    result = FusedExpertsResult(routed_out=torch.empty(0), expert_tokens=counts)

    with patch.object(torch.ops.vllm, "ascend_eplb_record_expert_tokens") as record_op:
        _record_v2_eplb_load(router, result)

    record_op.assert_called_once_with(
        counts,
        router.eplb_state.expert_load_view,
        router.eplb_state.should_record_tensor,
        1,
        4,
    )


def test_record_v2_eplb_load_requires_counts_even_when_collection_is_disabled():
    router = _router(record_enabled=False)
    result = FusedExpertsResult(routed_out=torch.empty(0), expert_tokens=None)
    with pytest.raises(RuntimeError, match="operator-provided expert counts"):
        _record_v2_eplb_load(router, result)


def test_record_v2_eplb_load_rejects_wrong_local_count():
    result = FusedExpertsResult(routed_out=torch.empty(0), expert_tokens=torch.zeros(3))
    with pytest.raises(RuntimeError, match="does not match local EPLB experts"):
        _record_v2_eplb_load(_router(), result)
