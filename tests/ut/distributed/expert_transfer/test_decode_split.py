# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.expert_transfer import expert_offload_method
from vllm_ascend.distributed.expert_transfer.expert_offload_manager import DecodeSplitPlan, _find_decode_split
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

MIB = 1024**2


@pytest.mark.parametrize(
    ("routes", "expected"),
    [
        ([], None),
        ([0, 1], None),
        ([0, 1, 0, 1, 2, 3, 2, 3], 2),
        ([0, 1, 2, 3, 2, 3, 2, 3], 1),
        ([0, 1, 0, 1, 0, 1, 2, 3], 3),
        ([0, 1, 1, 2, 2, 3], None),
    ],
)
def test_find_nearest_feasible_contiguous_cut(routes, expected):
    assert _find_decode_split(routes, top_k=2, capacity=2) == expected


@pytest.mark.parametrize("reverse", [False, True])
def test_decode_split_near_edge_of_large_batch(reverse):
    rows = 2048
    routes = list(range(8)) + list(range(8, 16)) * (rows - 1)
    if reverse:
        routes.reverse()
    assert _find_decode_split(routes, top_k=8, capacity=8) == (rows - 1 if reverse else 1)


@pytest.mark.parametrize(
    ("has_prefill", "expert_mib"),
    [(True, 8), (False, 4)],
)
def test_eager_split_fallback_does_not_advance_residency(expert_factory, manager_factory, has_prefill, expert_mib):
    layer, _ = expert_factory(experts=12)
    manager = manager_factory(layer)
    manager.has_prefill = has_prefill
    binding = layer._expert_offload_binding
    binding.expert_weight_bytes = expert_mib * MIB
    ids = torch.tensor([[0, 1], [2, 3], [2, 3]])

    view, physical_ids, _ = binding.prepare(layer, ids)

    assert view.w13_weight is manager.full_pool.tensors["w13_weight"]
    assert physical_ids is ids
    assert binding.slot_to_expert == (0, 1)


def test_eager_split_executes_original_routes_and_row_metadata(expert_factory, manager_factory, monkeypatch):
    layer, _ = expert_factory(experts=12)
    manager = manager_factory(layer)
    manager.has_prefill = False
    binding = layer._expert_offload_binding
    binding.expert_weight_bytes = 8 * MIB
    layer.ascend_pertoken_scale = torch.arange(4)
    layer.ascend_mc2_mask = torch.tensor([True, False, False, True])
    ids = torch.tensor([[0, 1], [2, 3], [2, 3], [2, 3]])
    hidden = torch.arange(8).view(4, 2)
    weights = torch.arange(8, dtype=torch.float32).view(4, 2)
    plan = binding.prepare(layer, ids)
    assert isinstance(plan, DecodeSplitPlan) and plan.cut == 1
    assert binding.slot_to_expert == (0, 1)

    calls = []

    def apply(layer, x, topk_weights, topk_ids, **_kwargs):
        logical_ids = torch.tensor(binding.slot_to_expert)[topk_ids]
        calls.append((x, topk_weights, logical_ids, layer.ascend_pertoken_scale, layer.ascend_mc2_mask))
        return FusedExpertsResult(routed_out=logical_ids + x)

    original_comm = object()
    context = SimpleNamespace(moe_comm_method=original_comm)
    monkeypatch.setattr(expert_offload_method, "_EXTRA_CTX", context)
    method = object.__new__(expert_offload_method.ExpertOffloadMoEMethod)
    method.quant_method = SimpleNamespace(apply=apply)

    result = method.apply(layer, hidden, weights, ids)

    assert context.moe_comm_method is original_comm
    assert [call[0].size(0) for call in calls] == [1, 3]
    for actual, expected in zip(
        zip(*calls), (hidden, weights, ids, layer.ascend_pertoken_scale, layer.ascend_mc2_mask)
    ):
        torch.testing.assert_close(torch.cat(actual), expected)
    torch.testing.assert_close(result.routed_out, ids + hidden)
