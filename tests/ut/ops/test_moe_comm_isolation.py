# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target and DSpark experts must not overwrite each other's dispatch state."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_comm_method as comm


@pytest.mark.parametrize("ep_size", [1, 8])
def test_expert_shapes_select_distinct_dispatchers(monkeypatch, ep_size):
    monkeypatch.setattr(comm, "_MoECommMethods", {})
    monkeypatch.setattr(comm, "_MoECommMethodsByConfig", {})
    monkeypatch.setattr(comm, "_EXTRA_CTX", SimpleNamespace())
    constructors = []
    for name in ("AllGatherCommImpl", "AlltoAllCommImpl", "MC2CommImpl", "FusedMC2CommImpl"):
        constructor = MagicMock(side_effect=lambda config: SimpleNamespace(owner=config))
        monkeypatch.setattr(comm, name, constructor)
        constructors.append(constructor)
    common = dict(
        hidden_dim=5120, intermediate_size_per_partition=2048, ep_size=ep_size, tp_size=1, dp_size=1, pcp_size=1
    )
    target = SimpleNamespace(num_experts=384, num_local_experts=384 // ep_size, experts_per_token=6, **common)
    draft = SimpleNamespace(num_experts=128, num_local_experts=128 // ep_size, experts_per_token=3, **common)
    comm.setup_moe_comm_method(target)
    comm.setup_moe_comm_method(draft)
    kinds = [MoECommType.ALLGATHER]
    if ep_size > 1:
        kinds += [MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2]
    for kind in kinds:
        target_method = comm.get_moe_comm_method(kind, target)
        draft_method = comm.get_moe_comm_method(kind, draft)
        assert target_method is not draft_method
        assert target_method.owner is target and draft_method.owner is draft
        for config, expected in ((target, target_method), (draft, draft_method), (target, target_method)):
            assert comm.activate_moe_comm_method(kind, config) is expected
            assert comm._EXTRA_CTX.moe_comm_method is expected
    assert sum(c.call_count for c in constructors) == 2 * len(kinds)
