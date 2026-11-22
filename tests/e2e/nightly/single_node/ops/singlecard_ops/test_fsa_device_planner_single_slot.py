# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT,
    FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT,
    SparseKVOffloadManager,
)


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="The FSA device planner requires A3")
def test_single_slot_plan_rewrites_current_and_historical_tokens():
    device = torch.device("npu:0")
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.topk = manager.topk_buffer_size = 1
    manager.max_num_topk_rows = 3
    manager.max_model_len = 8
    manager.tp_size = 1
    manager.tp_rank = 0
    manager.fsa_plan_ipc = None
    manager._fsa_plan_ipc_bytes = 0
    manager._fused_device_plan_library = manager._load_fused_device_planner()
    manager._init_fused_device_planner(device)
    manager.fused_plan_metadata_npu = torch.empty(4, dtype=torch.int32, device=device)

    ids = torch.tensor([10, 20], dtype=torch.int64, device=device)
    last_ids, _, _ = manager._get_fused_device_planner_layer_state(0, False)
    last_ids[:2].copy_(ids.flip(0))
    payload = manager.fused_device_planner_payload
    membership = torch.empty((3, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT), dtype=torch.int16, device=device)
    offset = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - 1

    for tokens, lengths, expected_codes in (([3, 2], [4, 6], [1, 0]), ([-1, 4], [0, 5], [0, 1])):
        topk = torch.tensor(tokens, dtype=torch.int32, device=device).view(2, 1)
        visible = torch.tensor(lengths, dtype=torch.int32, device=device)
        poison = -12345
        membership.fill_(poison)
        payload.fill_(poison)
        manager.fused_plan_metadata_npu.fill_(poison)
        manager._submit_fused_device_plan(0, 2, (ids, topk, visible, visible), membership, capturing=False)

        # The current token is injected into physical slot 0 of each row.
        # A historical token uses the host pool. The last row is inactive.
        expected_payload = torch.full_like(payload, poison, device="cpu")
        expected_membership = torch.full_like(membership, poison, device="cpu")
        for row, (physical, code) in enumerate(zip((1, 0), expected_codes)):
            expected_payload[row].zero_()
            packet = [code, -1, 0x5A45, 1, offset, physical, 0x5A44, 1, 0x5A56] + [-1] * 8
            wire = torch.tensor(packet, dtype=torch.int16)
            expected_payload[row].view(torch.int16)[: len(packet)] = wire
            expected_payload[row, -8] = physical
            expected_membership[row, offset:] = wire
        torch.testing.assert_close(payload.cpu(), expected_payload)
        torch.testing.assert_close(membership.cpu(), expected_membership)
        assert manager.fused_plan_metadata_npu.cpu().tolist() == [0, 1, 0, poison]
