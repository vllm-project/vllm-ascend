# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_local_compute_and_global_finalize_match_original_router(monkeypatch, scoring):
    router = AscendGroupedTopKRouter(
        top_k=3,
        global_num_experts=12,
        num_expert_group=None,
        topk_group=None,
        scoring_func=scoring,
    )
    torch.manual_seed(7)
    hidden, logits = torch.randn(9, 16), torch.randn(9, 12)
    captures, mappings = [], []
    router.capture_fn = lambda ids: captures.append(ids.clone())

    def mapping(ids):
        mappings.append(ids.clone())
        return (ids + torch.arange(len(ids)).unsqueeze(-1)) % 12

    monkeypatch.setattr(router, "_apply_eplb_mapping", mapping)
    expected_w, expected_i = router._select_experts(hidden, logits)
    captures.clear()
    mappings.clear()
    parts = [router.select_experts_locally(hidden[a:b], logits[a:b]) for a, b in [(0, 3), (3, 3), (3, 9)]]
    assert not captures and not mappings
    actual_w, actual_i = router.finalize_local_routing(
        torch.cat([x[0] for x in parts]), torch.cat([x[1] for x in parts])
    )
    torch.testing.assert_close(actual_w, expected_w, rtol=1e-6, atol=1e-7)
    assert torch.equal(actual_i, expected_i)
    assert captures[0].shape == mappings[0].shape == (9, 3)
    assert len(captures) == len(mappings) == 1


@pytest.mark.parametrize("kind", ["plain", "hash", "vision"])
def test_fused_router_only_requires_local_input_alignment(kind):
    router = AscendFusedTopKRouter(
        top_k=3,
        global_num_experts=12,
        scoring_func="sqrtsoftplus",
        tid2eid=torch.zeros(16, 3) if kind == "hash" else None,
        bias_vl=torch.zeros(12) if kind == "vision" else None,
    )
    assert router.supports_local_routing == (kind == "plain")
