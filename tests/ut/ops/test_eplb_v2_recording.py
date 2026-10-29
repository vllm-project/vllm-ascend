# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.fused_moe import routed_experts as routed_module
from vllm_ascend.ops.fused_moe.eplb import record_expert_tokens
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts


def _layer(record_enabled: bool = True):
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    layer._use_v2_model_runner = True
    layer.dynamic_eplb = False
    layer.moe_config = SimpleNamespace(ep_rank=2, num_local_experts=2)
    layer.router = SimpleNamespace(
        eplb_state=SimpleNamespace(
            expert_load_view=torch.zeros(8, dtype=torch.int32),
            should_record_tensor=torch.tensor(record_enabled),
            local_expert_start=4,
            local_expert_count=2,
        )
    )
    return layer


def test_record_eplb_load_uses_operator_counts():
    layer = _layer()
    counts = torch.tensor([2, 3], dtype=torch.int64)
    result = FusedExpertsResult(routed_out=torch.empty(0), expert_tokens=counts)

    with patch.object(torch.ops.vllm, "ascend_eplb_record_expert_tokens") as record_op:
        layer._record_eplb_load(result, topk_ids=torch.empty(0, 1, dtype=torch.int32), token_mask=None)

    record_op.assert_called_once_with(
        counts,
        layer.router.eplb_state.expert_load_view,
        layer.router.eplb_state.should_record_tensor,
        1,
        4,
    )


def test_record_eplb_load_requires_counts_even_when_collection_is_disabled():
    layer = _layer(record_enabled=False)
    result = FusedExpertsResult(routed_out=torch.empty(0), expert_tokens=None)
    with pytest.raises(RuntimeError, match="operator-provided expert counts"):
        layer._record_eplb_load(result, topk_ids=torch.empty(0, 1, dtype=torch.int32), token_mask=None)


def test_record_eplb_load_rejects_wrong_local_count():
    result = FusedExpertsResult(routed_out=torch.empty(0), expert_tokens=torch.zeros(3))
    with pytest.raises(RuntimeError, match="does not match local EPLB experts"):
        _layer()._record_eplb_load(result, topk_ids=torch.empty(0, 1, dtype=torch.int32), token_mask=None)


@pytest.mark.parametrize("field", ["expert_load_view", "should_record_tensor"])
def test_record_eplb_load_requires_initialized_v2_state(field):
    layer = _layer()
    setattr(layer.router.eplb_state, field, None)
    result = FusedExpertsResult(routed_out=torch.empty(0), expert_tokens=torch.tensor([2, 3]))
    with pytest.raises(RuntimeError, match="load-recording state is not initialized"):
        layer._record_eplb_load(result, topk_ids=torch.empty(0, 1, dtype=torch.int32), token_mask=None)


def test_record_eplb_load_skips_v2_without_eplb_state():
    layer = _layer()
    layer.router.eplb_state = None
    result = FusedExpertsResult(routed_out=torch.empty(0), expert_tokens=None)
    with patch.object(torch.ops.vllm, "ascend_eplb_record_expert_tokens") as record_op:
        layer._record_eplb_load(result, topk_ids=torch.empty(0, 1, dtype=torch.int32), token_mask=None)
    record_op.assert_not_called()


def test_allgather_records_masked_load_on_source_ineligible_owner(monkeypatch):
    layer = _layer(record_enabled=False)
    router = layer.router
    router.eplb_state.allgather_record_tensor = torch.tensor(-1, dtype=torch.int32)
    record_pointer = router.eplb_state.allgather_record_tensor.data_ptr()
    # Operator counts include communication/graph padding and ineligible sources.
    result = FusedExpertsResult(routed_out=torch.empty(0), expert_tokens=torch.tensor([9, 8]))
    topk_ids = torch.tensor([[4, 5], [4, 0], [4, 5]])
    mask = torch.tensor([True, True, False])
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_record_expert_tokens", record_expert_tokens)
    layer._record_eplb_load(result, topk_ids=topk_ids, token_mask=mask)
    torch.testing.assert_close(
        router.eplb_state.expert_load_view, torch.tensor([0, 0, 0, 0, 2, 1, 0, 0], dtype=torch.int32)
    )
    assert router.eplb_state.allgather_record_tensor.item() == 1
    # An ineligible later layer/draft must not clear the shared step decision.
    layer._record_eplb_load(result, topk_ids=topk_ids, token_mask=torch.zeros_like(mask))
    assert router.eplb_state.allgather_record_tensor.item() == 1
    assert router.eplb_state.allgather_record_tensor.data_ptr() == record_pointer
    torch.testing.assert_close(result.expert_tokens, torch.tensor([9, 8]))
    torch.testing.assert_close(
        router.eplb_state.expert_load_view, torch.tensor([0, 0, 0, 0, 2, 1, 0, 0], dtype=torch.int32)
    )


@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("record_enabled", [False, True])
def test_record_eplb_load_preserves_v2_operator_counts(monkeypatch, group_list_type, record_enabled):
    layer = _layer(record_enabled)
    result = FusedExpertsResult(
        routed_out=torch.empty(0),
        expert_tokens=torch.tensor([2, 5] if group_list_type == 0 else [2, 3]),
        group_list_type=group_list_type,
    )
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_record_expert_tokens", record_expert_tokens)
    layer._record_eplb_load(result, topk_ids=torch.empty(0, 1, dtype=torch.int32), token_mask=None)
    expected = [0, 0, 0, 0, 2, 3, 0, 0] if record_enabled else [0] * 8
    torch.testing.assert_close(layer.router.eplb_state.expert_load_view, torch.tensor(expected, dtype=torch.int32))


@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("multi_stage", [False, True])
@pytest.mark.parametrize("dynamic_eplb,collect", [(True, True), (True, False), (False, True)])
def test_record_eplb_load_preserves_legacy_collection(monkeypatch, group_list_type, multi_stage, dynamic_eplb, collect):
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    layer.router = None
    layer._use_v2_model_runner = False
    layer.dynamic_eplb = dynamic_eplb
    layer.multi_stage = multi_stage
    layer.num_iter = 3
    layer.load_counter = torch.tensor(0, dtype=torch.int32)
    layer.moe_load = torch.zeros((3, 2), dtype=torch.int32) if multi_stage else torch.zeros(2, dtype=torch.int64)
    monkeypatch.setattr(routed_module, "_EXTRA_CTX", SimpleNamespace(eplb_heat_collection_status=collect))
    for step in range(1, 5):
        counts = torch.tensor([step, 3 * step] if group_list_type == 0 else [step, 2 * step])
        layer._record_eplb_load(
            FusedExpertsResult(routed_out=torch.empty(0), expert_tokens=counts, group_list_type=group_list_type),
            topk_ids=torch.empty(0, 1, dtype=torch.int32),
            token_mask=None,
        )
    if not dynamic_eplb or not collect:
        assert not layer.moe_load.any()
        assert layer.load_counter.item() == 0
    elif multi_stage:
        torch.testing.assert_close(layer.moe_load, torch.tensor([[5, 10], [2, 4], [3, 6]], dtype=torch.int32))
        assert layer.load_counter.item() == 4
        assert layer.load_counter.dtype == torch.int32
    else:
        torch.testing.assert_close(layer.moe_load, torch.tensor([10, 20], dtype=torch.int64))
