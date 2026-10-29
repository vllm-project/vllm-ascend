# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl


def _new_comm_impl(symm_buffer=None):
    comm_impl = FusedMC2CommImpl.__new__(FusedMC2CommImpl)
    comm_impl.mega_moe_symm_buffer = symm_buffer
    comm_impl._mega_moe_hccl_state_stale = False
    return comm_impl


def test_fused_mc2_hccl_lifecycle_is_noop_before_buffer_initialization():
    comm_impl = _new_comm_impl()

    assert comm_impl.prepare_hccl_teardown()
    assert comm_impl.refresh_hccl_runtime_state()
    assert not comm_impl._mega_moe_hccl_state_stale


def test_fused_mc2_prepare_marks_existing_context_stale_without_destroying_buffer():
    context_manager = MagicMock()
    symm_buffer = SimpleNamespace(_ctx_manager=context_manager)
    comm_impl = _new_comm_impl(symm_buffer)

    assert comm_impl.prepare_hccl_teardown()

    assert comm_impl.mega_moe_symm_buffer is symm_buffer
    assert comm_impl._mega_moe_hccl_state_stale
    context_manager.update_group.assert_not_called()


def test_fused_mc2_prepare_refuses_teardown_without_update_group_api():
    comm_impl = _new_comm_impl(SimpleNamespace(_ctx_manager=object()))

    with pytest.raises(RuntimeError, match="update_group"):
        comm_impl.prepare_hccl_teardown()

    assert not comm_impl._mega_moe_hccl_state_stale


def test_fused_mc2_refreshes_existing_context_against_restored_group():
    context_manager = MagicMock(ccl_buffer_size=4096)
    inference_modes: list[bool] = []
    context_manager.update_group.side_effect = lambda *_: inference_modes.append(torch.is_inference_mode_enabled())
    context = MagicMock()
    symm_buffer = SimpleNamespace(
        _ctx_manager=context_manager,
        context=context,
        group="old-group",
        group_name="old-name",
        rank_id=7,
        ep_world_size=1,
        ccl_buffer_size=1,
    )
    comm_impl = _new_comm_impl(symm_buffer)
    comm_impl._mega_moe_hccl_state_stale = True
    new_group = MagicMock()
    backend = MagicMock()
    backend.get_hccl_comm_name.return_value = "new-name"
    new_group._get_backend.return_value = backend

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group",
            return_value=SimpleNamespace(device_group=new_group),
        ),
        patch.object(torch.distributed, "get_rank", return_value=3),
        patch.object(torch.distributed, "get_world_size", return_value=8),
        patch.object(torch.distributed, "barrier") as mock_barrier,
        patch.object(torch.npu, "current_device", return_value=4),
    ):
        assert comm_impl.refresh_hccl_runtime_state()

    new_group._get_backend.assert_called_once_with(torch.device("npu"))
    backend.get_hccl_comm_name.assert_called_once_with(3)
    context_manager.update_group.assert_called_once_with("new-name", context)
    assert inference_modes == [True]
    assert symm_buffer.group is new_group
    assert symm_buffer.group_name == "new-name"
    assert symm_buffer.rank_id == 3
    assert symm_buffer.ep_world_size == 8
    assert symm_buffer.ccl_buffer_size == 4096
    mock_barrier.assert_called_once_with(group=new_group, device_ids=[4])
    assert not comm_impl._mega_moe_hccl_state_stale


def test_fused_mc2_refresh_is_noop_when_context_was_not_invalidated():
    context_manager = MagicMock()
    comm_impl = _new_comm_impl(SimpleNamespace(_ctx_manager=context_manager))

    assert comm_impl.refresh_hccl_runtime_state()

    context_manager.update_group.assert_not_called()
