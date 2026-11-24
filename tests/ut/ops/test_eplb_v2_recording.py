# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.distributed.eplb import eplb_state as state_module
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState
from vllm_ascend.ops.fused_moe.eplb import record_expert_tokens, record_source_routes


def _forward_context(is_padding=None):
    """A context like the one every forward enters, with its own extras dict."""
    return SimpleNamespace(is_padding=is_padding, additional_kwargs={})


def _state(record_enabled: bool = True):
    state = AscendEplbLayerState()
    state.expert_load_view = torch.zeros(8, dtype=torch.int32)
    state.should_record_tensor = torch.tensor(record_enabled)
    state.num_unpadded_tokens_tensors = [torch.tensor(3)]
    state.local_expert_start = 4
    state.local_expert_count = 2
    return state


def _record_source_routes(monkeypatch, context=None, tp_size=1, tp_rank=0):
    """Run the source-side operator on its CPU reference for one TP rank."""
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_record_source_routes", record_source_routes)
    monkeypatch.setattr(state_module, "get_forward_context", lambda: context or _forward_context())
    monkeypatch.setattr(state_module, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(state_module, "get_tensor_model_parallel_rank", lambda: tp_rank)


def test_record_load_counts_source_routes_for_every_physical_expert(monkeypatch):
    # This rank owns experts [4, 6). Counting what the owner executed dropped
    # the routes to the experts of the other ranks.
    state = _state()
    topk_ids = torch.tensor([[0, 7], [5, 0], [4, 5]])
    _record_source_routes(monkeypatch)

    state.record_load(None, 1, topk_ids=topk_ids, num_source_tokens=3)

    expected = torch.tensor([2, 0, 0, 0, 1, 2, 0, 1], dtype=torch.int32)
    torch.testing.assert_close(state.expert_load_view, expected)


def test_record_load_skips_padding_rows(monkeypatch):
    state = _state()
    context = _forward_context(is_padding=torch.tensor([False, True, False]))
    topk_ids = torch.tensor([[0, 1], [7, 7], [1, 0]])
    _record_source_routes(monkeypatch, context)

    state.record_load(None, 1, topk_ids=topk_ids, num_source_tokens=3)

    # Rows 0 and 2 each route to experts 0 and 1.
    expected = torch.tensor([2, 2, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    torch.testing.assert_close(state.expert_load_view, expected)


def test_record_load_records_nothing_from_an_ineligible_source(monkeypatch):
    # An idle DP rank or one running the other collection phase contributes
    # zero, whatever its rows route to.
    state = _state(record_enabled=False)
    topk_ids = torch.tensor([[0, 7], [5, 0], [4, 5]])
    _record_source_routes(monkeypatch)

    state.record_load(None, 1, topk_ids=topk_ids, num_source_tokens=3)

    torch.testing.assert_close(state.expert_load_view, torch.zeros(8, dtype=torch.int32))


def test_idle_source_keeps_peer_routes_after_the_ep_sum(monkeypatch):
    # The idle rank owns experts [4, 6) but executes the routes the active peer
    # dispatched, so only the peer's count may enter the sum.
    idle, active = _state(record_enabled=False), _state()
    _record_source_routes(monkeypatch)
    active.record_load(None, 1, topk_ids=torch.tensor([[4, 5], [0, 4]]), num_source_tokens=2)
    idle.record_load(None, 1, topk_ids=torch.tensor([[4, 4], [4, 4]]), num_source_tokens=2)

    total = idle.expert_load_view + active.expert_load_view
    torch.testing.assert_close(total, torch.tensor([1, 0, 0, 0, 2, 1, 0, 0], dtype=torch.int32))


def test_record_load_prefers_the_dispatch_mask(monkeypatch):
    # The comm method's mask already follows the routed rows, including the
    # rows padded to the DP-wide length.
    state = _state()
    context = _forward_context(is_padding=torch.zeros(2, dtype=torch.bool))
    topk_ids = torch.tensor([[0, 1], [7, 7], [2, 3]])
    dispatch_mask = torch.tensor([True, False, True])
    _record_source_routes(monkeypatch, context)

    state.record_load(None, 1, topk_ids=topk_ids, num_source_tokens=2, dispatch_mask=dispatch_mask)

    expected = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.int32)
    torch.testing.assert_close(state.expert_load_view, expected)
    assert context.additional_kwargs == {}


def test_record_load_ignores_expert_ids_outside_the_physical_range(monkeypatch):
    state = _state()
    topk_ids = torch.tensor([[-1, 8], [2, 2], [0, 7]])
    _record_source_routes(monkeypatch)

    state.record_load(None, 1, topk_ids=topk_ids, num_source_tokens=3)

    expected = torch.tensor([1, 0, 2, 0, 0, 0, 0, 1], dtype=torch.int32)
    torch.testing.assert_close(state.expert_load_view, expected)


def test_derived_source_mask_is_built_once_per_forward(monkeypatch):
    # Every MoE layer derives the same mask from the same inputs.
    state = _state()
    context = _forward_context()
    topk_ids = torch.tensor([[0, 1], [1, 0], [2, 3]])
    _record_source_routes(monkeypatch, context)
    with patch.object(state_module, "make_allgather_mask", wraps=state_module.make_allgather_mask) as build_mask:
        for _ in range(2):
            state.record_load(None, 1, topk_ids=topk_ids, num_source_tokens=3)

    assert build_mask.call_count == 1
    # Each layer routes twice to experts 0 and 1 and once to experts 2 and 3.
    expected = torch.tensor([4, 4, 2, 2, 0, 0, 0, 0], dtype=torch.int32)
    torch.testing.assert_close(state.expert_load_view, expected)


@pytest.mark.parametrize(
    "is_padding,expected_by_rank",
    [
        # Graph padding at the end of eight rows. The second chunk keeps two rows.
        ([0, 0, 0, 0, 0, 0, 1, 1], ([1, 1, 2, 2, 0, 0, 0, 2], [1, 1, 0, 0, 0, 0, 0, 2])),
        # Four rows padded to the DP-wide length of eight. The second chunk is
        # all padding although it is as long as the runner's rows.
        ([0, 0, 0, 0], ([1, 1, 2, 2, 0, 0, 0, 2], [0] * 8)),
    ],
)
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_derived_source_mask_takes_the_tp_chunk(monkeypatch, is_padding, expected_by_rank, tp_rank):
    """``prepare`` pads the rows to eight and splits them into two TP chunks."""
    state = _state()
    state.num_unpadded_tokens_tensors = [torch.tensor(4)]
    context = _forward_context(is_padding=torch.tensor(is_padding, dtype=torch.bool))
    topk_ids = torch.tensor([[0, 1], [7, 7], [2, 3], [3, 2]])
    _record_source_routes(monkeypatch, context, tp_size=2, tp_rank=tp_rank)

    state.record_load(None, 1, topk_ids=topk_ids, num_source_tokens=4)

    expected = torch.tensor(expected_by_rank[tp_rank], dtype=torch.int32)
    torch.testing.assert_close(state.expert_load_view, expected)


def test_derived_source_mask_of_a_sequence_parallel_shard_uses_local_rows(monkeypatch):
    # Sequence parallelism hands ``prepare`` this rank's shard, so the runner
    # padding already describes the routed rows.
    state = _state()
    context = _forward_context(is_padding=torch.tensor([True, False, False]))
    topk_ids = torch.tensor([[7, 7], [0, 1], [1, 0]])
    _record_source_routes(monkeypatch, context, tp_size=2, tp_rank=1)

    state.record_load(None, 1, topk_ids=topk_ids, num_source_tokens=3, is_sequence_parallel=True)

    expected = torch.tensor([2, 2, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    torch.testing.assert_close(state.expert_load_view, expected)


@pytest.mark.parametrize("field", ["expert_load_view", "should_record_tensor", "num_unpadded_tokens_tensors"])
def test_record_load_requires_initialized_state(field, monkeypatch):
    state = _state()
    setattr(state, field, None)
    _record_source_routes(monkeypatch)
    topk_ids = torch.tensor([[0, 1], [1, 0], [2, 3]])

    with pytest.raises(RuntimeError, match="load-recording state is not initialized"):
        state.record_load(None, 1, topk_ids=topk_ids, num_source_tokens=3)


@pytest.mark.parametrize("group_list_type", [0, 1])
def test_allgather_records_masked_load_on_source_ineligible_owner(monkeypatch, group_list_type):
    state = _state(record_enabled=False)
    state.allgather_record_tensor = torch.tensor(-1, dtype=torch.int32)
    record_pointer = state.allgather_record_tensor.data_ptr()
    state.allgather_source_is_padding = torch.zeros(3, dtype=torch.bool)
    counts = torch.tensor([9, 8])
    topk_ids = torch.tensor([[0, 0], [0, 0], [0, 0], [4, 5], [4, 0], [4, 5]])
    token_mask = torch.tensor([False, False, False, True, True, False])

    def gather_mask(local_mask):
        assert local_mask.tolist() == [False, False, False]
        return token_mask

    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_record_expert_tokens", record_expert_tokens)
    context = _forward_context()
    monkeypatch.setattr(state_module, "get_forward_context", lambda: context)
    state.record_load(counts, group_list_type, topk_ids=topk_ids, num_source_tokens=3, all_gather_mask=gather_mask)
    expected = torch.tensor([0, 0, 0, 0, 2, 1, 0, 0], dtype=torch.int32)
    torch.testing.assert_close(state.expert_load_view, expected)
    assert state.allgather_record_tensor.item() == 1
    # An ineligible later layer/draft must not clear the shared step decision.
    token_mask.zero_()
    context = _forward_context()
    state.record_load(counts, group_list_type, topk_ids=topk_ids, num_source_tokens=3, all_gather_mask=gather_mask)
    assert state.allgather_record_tensor.item() == 1
    assert state.allgather_record_tensor.data_ptr() == record_pointer
    torch.testing.assert_close(counts, torch.tensor([9, 8]))
    torch.testing.assert_close(state.expert_load_view, expected)


def test_record_load_updates_step_flag_before_rejecting_missing_counts(monkeypatch):
    state = _state()
    state.allgather_record_tensor = torch.tensor(-1, dtype=torch.int32)
    state.allgather_source_is_padding = torch.tensor([False])
    monkeypatch.setattr(state_module, "get_forward_context", _forward_context)
    with pytest.raises(RuntimeError, match="operator-provided expert counts"):
        state.record_load(None, 1, topk_ids=torch.tensor([[4]]), num_source_tokens=1, all_gather_mask=lambda mask: mask)
    assert state.allgather_record_tensor.item() == 1
