# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.eplb import eplb_state as state_module
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState
from vllm_ascend.ops.fused_moe import eplb as eplb_ops
from vllm_ascend.ops.fused_moe.eplb import record_expert_tokens
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather


@pytest.fixture
def prepare_finalize(monkeypatch):
    monkeypatch.setattr(eplb_ops, "use_allgather_eplb", lambda enabled: False)
    return MagicMock(spec=PrepareAndFinalizeWithAllGather)


def _state(record_enabled: bool = True):
    state = AscendEplbLayerState()
    state.expert_load_view = torch.zeros(8, dtype=torch.int32)
    state.should_record_tensor = torch.tensor(record_enabled)
    state.local_expert_start = 4
    state.local_expert_count = 2
    state.num_unpadded_tokens_tensors = [torch.tensor(3)]
    return state


def test_record_load_uses_operator_counts(prepare_finalize):
    state = _state()
    counts = torch.tensor([2, 3], dtype=torch.int64)
    with patch.object(torch.ops.vllm, "ascend_eplb_record_expert_tokens") as record_op:
        state.record_load(
            counts,
            1,
            topk_ids=torch.empty(0, 1, dtype=torch.int32),
            num_source_tokens=0,
            is_sequence_parallel=False,
            prepare_finalize=prepare_finalize,
        )
    record_op.assert_called_once_with(counts, state.expert_load_view, state.should_record_tensor, 1, 4)


def test_record_load_requires_counts_even_when_collection_is_disabled(prepare_finalize):
    with pytest.raises(RuntimeError, match="operator-provided expert counts"):
        _state(record_enabled=False).record_load(
            None,
            1,
            topk_ids=torch.empty(0, 1, dtype=torch.int32),
            num_source_tokens=0,
            is_sequence_parallel=False,
            prepare_finalize=prepare_finalize,
        )


def test_record_load_rejects_wrong_local_count(prepare_finalize):
    with pytest.raises(RuntimeError, match="does not match local EPLB experts"):
        _state().record_load(
            torch.zeros(3),
            1,
            topk_ids=torch.empty(0, 1, dtype=torch.int32),
            num_source_tokens=0,
            is_sequence_parallel=False,
            prepare_finalize=prepare_finalize,
        )


@pytest.mark.parametrize("field", ["expert_load_view", "should_record_tensor"])
def test_record_load_requires_initialized_state(field, prepare_finalize):
    state = _state()
    setattr(state, field, None)
    with pytest.raises(RuntimeError, match="load-recording state is not initialized"):
        state.record_load(
            torch.tensor([2, 3]),
            1,
            topk_ids=torch.empty(0, 1, dtype=torch.int32),
            num_source_tokens=0,
            is_sequence_parallel=False,
            prepare_finalize=prepare_finalize,
        )


def test_allgather_records_masked_load_on_source_ineligible_owner(monkeypatch, prepare_finalize):
    state = _state(record_enabled=False)
    state.allgather_record_tensor = torch.tensor(-1, dtype=torch.int32)
    record_pointer = state.allgather_record_tensor.data_ptr()
    # Operator counts include communication/graph padding and ineligible sources.
    counts = torch.tensor([9, 8])
    topk_ids = torch.tensor([[0, 0], [0, 0], [0, 0], [4, 5], [4, 0], [4, 5]])
    peer_mask = torch.tensor([True, True, False])

    def gather_mask(local_mask):
        assert not local_mask.any()
        return torch.cat([local_mask, peer_mask])

    prepare_finalize.all_gather_eplb_mask.side_effect = gather_mask
    monkeypatch.setattr(eplb_ops, "use_allgather_eplb", lambda enabled: enabled)
    monkeypatch.setattr(state_module, "get_forward_context", lambda: SimpleNamespace(is_padding=None))
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_record_expert_tokens", record_expert_tokens)
    state.record_load(
        counts, 1, topk_ids=topk_ids, num_source_tokens=3, is_sequence_parallel=False, prepare_finalize=prepare_finalize
    )
    expected = torch.tensor([0, 0, 0, 0, 2, 1, 0, 0], dtype=torch.int32)
    torch.testing.assert_close(state.expert_load_view, expected)
    assert state.allgather_record_tensor.item() == 1
    # An ineligible later layer/draft must not clear the shared step decision.
    peer_mask.zero_()
    state.record_load(
        counts, 1, topk_ids=topk_ids, num_source_tokens=3, is_sequence_parallel=False, prepare_finalize=prepare_finalize
    )
    assert prepare_finalize.all_gather_eplb_mask.call_count == 2
    assert state.allgather_record_tensor.item() == 1
    assert state.allgather_record_tensor.data_ptr() == record_pointer
    torch.testing.assert_close(counts, torch.tensor([9, 8]))
    torch.testing.assert_close(state.expert_load_view, expected)


@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("record_enabled", [False, True])
def test_record_load_preserves_operator_count_formats(monkeypatch, group_list_type, record_enabled, prepare_finalize):
    state = _state(record_enabled)
    counts = torch.tensor([2, 5] if group_list_type == 0 else [2, 3])
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_record_expert_tokens", record_expert_tokens)
    state.record_load(
        counts,
        group_list_type,
        topk_ids=torch.empty(0, 1, dtype=torch.int32),
        num_source_tokens=0,
        is_sequence_parallel=False,
        prepare_finalize=prepare_finalize,
    )
    expected = [0, 0, 0, 0, 2, 3, 0, 0] if record_enabled else [0] * 8
    torch.testing.assert_close(state.expert_load_view, torch.tensor(expected, dtype=torch.int32))
