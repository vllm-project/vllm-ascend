# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.routed_experts import _record_v2_eplb_load


def test_record_v2_eplb_load_uses_operator_expert_counts():
    expert_load = torch.zeros(8, dtype=torch.int32)
    record_enabled = torch.tensor(True)
    expert_tokens = torch.tensor([2, 3], dtype=torch.int64)
    router = SimpleNamespace(
        eplb_state=SimpleNamespace(
            expert_load_view=expert_load,
            should_record_tensor=record_enabled,
            local_expert_start=4,
            local_expert_count=2,
        )
    )
    result = FusedExpertsResult(
        routed_out=torch.empty(0),
        group_list_type=1,
        expert_tokens=expert_tokens,
    )

    with patch.object(
        torch.ops.vllm,
        "ascend_eplb_record_expert_tokens",
    ) as record_op:
        _record_v2_eplb_load(
            router,
            torch.tensor([[4, 5]], dtype=torch.int32),
            result,
        )

    record_op.assert_called_once_with(
        expert_tokens,
        expert_load,
        record_enabled,
        1,
        4,
    )


def test_record_v2_eplb_load_falls_back_to_physical_ids():
    expert_load = torch.zeros(8, dtype=torch.int32)
    record_enabled = torch.tensor(True)
    num_unpadded_tokens = torch.tensor(1, dtype=torch.int32)
    physical_ids = torch.tensor([[4, 5], [6, 7]], dtype=torch.int32)
    router = SimpleNamespace(
        eplb_state=SimpleNamespace(
            expert_load_view=expert_load,
            should_record_tensor=record_enabled,
            num_unpadded_tokens_tensors=[num_unpadded_tokens],
        )
    )
    result = FusedExpertsResult(
        routed_out=torch.empty(0),
        expert_tokens=None,
    )

    with (
        patch(
            "vllm_ascend.ops.fused_moe.routed_experts."
            "dbo_current_ubatch_id",
            return_value=0,
        ),
        patch.object(
            torch.ops.vllm,
            "ascend_eplb_record_physical_expert_load",
        ) as record_op,
    ):
        _record_v2_eplb_load(router, physical_ids, result)

    record_op.assert_called_once_with(
        physical_ids,
        expert_load,
        record_enabled,
        num_unpadded_tokens,
    )
