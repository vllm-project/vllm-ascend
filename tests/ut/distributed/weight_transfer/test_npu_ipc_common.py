# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

from vllm_ascend.distributed.weight_transfer import npu_ipc_common


def test_npu_uuid_uses_visible_physical_device(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2, 5")
    npu_ipc_common.npu_generate_uuid.cache_clear()
    with patch.object(npu_ipc_common, "_get_ip", return_value="host"):
        assert npu_ipc_common.npu_generate_uuid(1) == "host-5"
    npu_ipc_common.npu_generate_uuid.cache_clear()


def test_all_gather_is_a_noop_without_distributed_group():
    handles = [{"host-0": ("handle",)}]
    with patch.object(npu_ipc_common.torch.distributed, "is_initialized", return_value=False):
        assert npu_ipc_common.all_gather_and_merge_handles(handles) == handles
        assert npu_ipc_common.is_rank_zero() is True


def test_all_gather_merges_handles_on_sender_rank():
    handles = [{"host-0": ("rank-0",)}]
    gathered = [handles, [{"host-1": ("rank-1",)}]]
    all_gather = MagicMock(side_effect=lambda output, _: output.__setitem__(slice(None), gathered))
    with (
        patch.object(npu_ipc_common.torch.distributed, "is_initialized", return_value=True),
        patch.object(npu_ipc_common.torch.distributed, "get_world_size", return_value=2),
        patch.object(npu_ipc_common.torch.distributed, "all_gather_object", all_gather),
        patch.object(npu_ipc_common.torch.distributed, "barrier"),
        patch.object(npu_ipc_common.torch.npu, "synchronize"),
    ):
        assert npu_ipc_common.all_gather_and_merge_handles(handles, is_sender=True) == [
            {"host-0": ("rank-0",), "host-1": ("rank-1",)}
        ]
