# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from vllm_ascend.device_allocator.sleep_mem_optimized import HcclSleepWakeupManager
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


@pytest.mark.parametrize("draft_first", [True, False])
def test_channel_wakeup_replaces_manager_but_preserves_captured_context(draft_first):
    old_manager = MagicMock()
    old_manager.update_group.side_effect = RuntimeError("create a replacement CommContextManager instead")
    new_manager = SimpleNamespace(ccl_buffer_size=4096)
    with torch.inference_mode():
        captured_context = torch.tensor([1, 2])
    replacement = SimpleNamespace(
        _ctx_manager=new_manager, context=torch.tensor([3, 4]), topo_type=1, rank_num_per_server=8
    )
    recreate = MagicMock(return_value=replacement)
    owner = SimpleNamespace(
        group=object(),
        _ctx_manager=old_manager,
        context=captured_context,
        _recreate_for_group=recreate,
        num_experts=384,
    )
    draft = SimpleNamespace(**vars(owner))
    draft.num_experts = 128
    new_group = MagicMock()
    coordinator = SimpleNamespace(device_group=new_group, _main_mega_moe_symm_buffer=owner)
    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group", return_value=coordinator),
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(torch.distributed, "get_world_size", return_value=8),
        patch.object(torch.distributed, "barrier") as barrier,
        patch.object(torch.npu, "current_device", return_value=0),
    ):
        for buf in (draft, owner) if draft_first else (owner, draft):
            comm = _new_comm_impl(buf)
            assert comm.prepare_hccl_teardown()
            assert comm.refresh_hccl_runtime_state()

    recreate.assert_called_once_with(new_group)
    old_manager.update_group.assert_not_called()
    barrier.assert_called_once()
    for buf in (owner, draft):
        assert buf.context is captured_context
        assert buf._ctx_manager is new_manager
        assert (buf.ccl_buffer_size, buf.topo_type, buf.rank_num_per_server) == (4096, 1, 8)
    assert (owner.num_experts, draft.num_experts) == (384, 128)
    torch.testing.assert_close(captured_context, replacement.context)


def test_fused_mc2_sleep_wakeup_rebinds_shared_context_once_per_restored_group():
    context = MagicMock()
    context_manager = MagicMock(ccl_buffer_size=4096)
    old_group = object()
    target = SimpleNamespace(
        _ctx_manager=context_manager,
        context=context,
        group=old_group,
        group_name="old-group",
        rank_id=0,
        ep_world_size=2,
        ccl_buffer_size=1024,
        num_experts=256,
    )
    draft = SimpleNamespace(**vars(target))
    draft.num_experts = 64
    target_comm, draft_comm = _new_comm_impl(target), _new_comm_impl(draft)
    coordinator = SimpleNamespace(device_group=old_group, _main_mega_moe_symm_buffer=target)
    new_groups = [MagicMock(), MagicMock()]
    for index, group in enumerate(new_groups):
        group._get_backend.return_value.get_hccl_comm_name.return_value = f"restored-group-{index}"

    with (
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group", return_value=coordinator),
        patch.object(torch.distributed, "get_rank", return_value=3),
        patch.object(torch.distributed, "get_world_size", return_value=8),
        patch.object(torch.distributed, "barrier") as mock_barrier,
        patch.object(torch.npu, "current_device", return_value=4),
    ):
        # Either view can be first in the wake-up iteration.
        for index, methods in enumerate(((draft_comm, target_comm), (target_comm, draft_comm))):
            with patch.dict(
                "vllm_ascend.ops.fused_moe.moe_comm_method._MoECommMethods",
                dict(enumerate(methods)),
                clear=True,
            ):
                assert HcclSleepWakeupManager.prepare_moe_hccl_teardown()
                assert all(method._mega_moe_hccl_state_stale for method in methods)
                coordinator.device_group = new_groups[index]
                HcclSleepWakeupManager.refresh_moe_hccl_runtime_state()
                assert not any(method._mega_moe_hccl_state_stale for method in methods)

            for buffer in (target, draft):
                assert buffer.group is new_groups[index]
                assert buffer.group_name == f"restored-group-{index}"
                assert buffer.rank_id == 3
                assert buffer.ep_world_size == 8
                assert buffer.ccl_buffer_size == 4096
                assert buffer.context is context
                assert buffer._ctx_manager is context_manager
            assert target.num_experts == 256
            assert draft.num_experts == 64

    assert context_manager.update_group.call_args_list == [
        call("restored-group-0", context),
        call("restored-group-1", context),
    ]
    assert mock_barrier.call_args_list == [call(group=group, device_ids=[4]) for group in new_groups]
