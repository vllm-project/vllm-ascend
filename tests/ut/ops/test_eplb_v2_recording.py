# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState
from vllm_ascend.ops.fused_moe.eplb import record_expert_tokens


def _state(record_enabled: bool = True):
    state = AscendEplbLayerState()
    state.expert_load_view = torch.zeros(8, dtype=torch.int32)
    state.should_record_tensor = torch.tensor(record_enabled)
    state.local_expert_start = 4
    state.local_expert_count = 2
    return state


def test_record_load_uses_operator_counts():
    state = _state()
    counts = torch.tensor([2, 3], dtype=torch.int64)
    with patch.object(torch.ops.vllm, "ascend_eplb_record_expert_tokens") as record_op:
        state.record_load(counts, 1, topk_ids=torch.empty(0, 1, dtype=torch.int32))
    record_op.assert_called_once_with(counts, state.expert_load_view, state.should_record_tensor, 1, 4)


def test_record_load_requires_counts_even_when_collection_is_disabled():
    with pytest.raises(RuntimeError, match="operator-provided expert counts"):
        _state(record_enabled=False).record_load(None, 1, topk_ids=torch.empty(0, 1, dtype=torch.int32))


def test_record_load_rejects_wrong_local_count():
    with pytest.raises(RuntimeError, match="does not match local EPLB experts"):
        _state().record_load(torch.zeros(3), 1, topk_ids=torch.empty(0, 1, dtype=torch.int32))


@pytest.mark.parametrize("field", ["expert_load_view", "should_record_tensor"])
def test_record_load_requires_initialized_state(field):
    state = _state()
    setattr(state, field, None)
    with pytest.raises(RuntimeError, match="load-recording state is not initialized"):
        state.record_load(torch.tensor([2, 3]), 1, topk_ids=torch.empty(0, 1, dtype=torch.int32))


@pytest.mark.parametrize("group_list_type", [0, 1])
def test_allgather_records_masked_load_on_source_ineligible_owner(monkeypatch, group_list_type):
    state = _state(record_enabled=False)
    state.allgather_record_tensor = torch.tensor(-1, dtype=torch.int32)
    record_pointer = state.allgather_record_tensor.data_ptr()
    # The execution layer supplies a mask aligned to all gathered routes.
    counts = torch.tensor([9, 8])
    topk_ids = torch.tensor([[0, 0], [0, 0], [0, 0], [4, 5], [4, 0], [4, 5]])
    token_mask = torch.tensor([False, False, False, True, True, False])
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_record_expert_tokens", record_expert_tokens)
    state.record_load(counts, group_list_type, topk_ids=topk_ids, token_mask=token_mask)
    expected = torch.tensor([0, 0, 0, 0, 2, 1, 0, 0], dtype=torch.int32)
    torch.testing.assert_close(state.expert_load_view, expected)
    assert state.allgather_record_tensor.item() == 1
    # An ineligible later layer/draft must not clear the shared step decision.
    token_mask.zero_()
    state.record_load(counts, group_list_type, topk_ids=topk_ids, token_mask=token_mask)
    assert state.allgather_record_tensor.item() == 1
    assert state.allgather_record_tensor.data_ptr() == record_pointer
    torch.testing.assert_close(counts, torch.tensor([9, 8]))
    torch.testing.assert_close(state.expert_load_view, expected)


def test_record_load_updates_step_flag_before_rejecting_missing_counts():
    state = _state()
    state.allgather_record_tensor = torch.tensor(-1, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="operator-provided expert counts"):
        state.record_load(None, 1, topk_ids=torch.tensor([[4]]), token_mask=torch.tensor([True]))
    assert state.allgather_record_tensor.item() == 1


@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("record_enabled", [False, True])
def test_record_load_preserves_operator_count_formats(monkeypatch, group_list_type, record_enabled):
    state = _state(record_enabled)
    counts = torch.tensor([2, 5] if group_list_type == 0 else [2, 3])
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_record_expert_tokens", record_expert_tokens)
    state.record_load(counts, group_list_type, topk_ids=torch.empty(0, 1, dtype=torch.int32))
    expected = [0, 0, 0, 0, 2, 3, 0, 0] if record_enabled else [0] * 8
    torch.testing.assert_close(state.expert_load_view, torch.tensor(expected, dtype=torch.int32))
