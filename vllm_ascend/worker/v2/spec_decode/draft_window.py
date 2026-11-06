# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Qualify the actual Draft consumer and create its capture-bank owner.

Method policy and interval arithmetic are shared with the ordinary eager
runtime. The provider owns source geometry and persistent Graph backing;
Full write tables and Target state remain owned by the existing runner.
"""

from typing import TYPE_CHECKING, Any

from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowProviderStatus,
    inventory_draft_window_groups,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider


def create_draft_window_graph_provider(owner: Any) -> "DraftWindowGraphBindingProvider":
    """Build source geometry after the caller validates its real Graph executor.

    Executor features do not turn a dense Draft cache into a sharded provider.
    Check the actual source layout directly, without constructing an
    artificial eager capability decision.
    """
    policy = owner._draft_window_policy
    block_tables = owner.block_tables
    if policy is None:
        raise ValueError("Graph source requires an eligible Window policy")
    dflash2 = getattr(owner, "_speculator_name", None) == "DFlash2"
    groups = inventory_draft_window_groups(owner)
    if len(groups) != 1:
        raise NotImplementedError(f"Draft Graph requires one executable consumer group, found {len(groups)}")
    group = groups[0]
    if not dflash2 and (group.status is not DraftWindowProviderStatus.DIRECT_DENSE or group.native_window is not None):
        raise NotImplementedError("Draft Graph requires its supported dense or DFlash2 native reader")
    if dflash2:
        from vllm_ascend.worker.v2.spec_decode.draft_window_dflash2_graph import validate_dflash2_graph_source

        validate_dflash2_graph_source(owner, group)
    from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider

    return DraftWindowGraphBindingProvider(policy=policy, group=group, block_tables=block_tables)
