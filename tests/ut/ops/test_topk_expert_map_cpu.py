# SPDX-License-Identifier: Apache-2.0
"""Exercise physical-ID fusion dispatch without loading the NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_method(path, cls_name, method, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if getattr(n, "name", None) == cls_name)
    fn = next(n for n in cls.body if getattr(n, "name", None) == method)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), scope)
    return scope[method]


@pytest.mark.parametrize(
    "excluded", [None, "device", "capture", "eplb", "custom", "hash", "vision", "tid2eid", "shape", "dtype"]
)
def test_supported_routing_protocols(excluded):
    scope = {"torch": torch, "DeviceOperator": SimpleNamespace(supports_moe_gating_top_k_log2phy=excluded != "device")}
    supports = load_method(
        "vllm_ascend/ops/fused_moe/router/fused_topk_router.py",
        "AscendFusedTopKRouter",
        "supports_log2phy_fusion",
        scope,
    )
    router = SimpleNamespace(
        capture_fn=object() if excluded == "capture" else None,
        eplb_state=object() if excluded == "eplb" else None,
        custom_routing_function=object() if excluded == "custom" else None,
        scoring_func="sqrtsoftplus" if excluded == "hash" else "sigmoid",
        bias_vl=object() if excluded == "vision" else None,
        tid2eid=object() if excluded == "tid2eid" else None,
        is_fused_supported=lambda _: excluded != "dtype",
    )
    logits = torch.zeros(2, 2049 if excluded == "shape" else 4)
    assert supports(router, logits, logits) is (excluded is None)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("shared", [0, 2])
def test_map_applied_once_and_shared_ids_appended(fused, shared):
    logical = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)
    mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    weights = torch.full((2, 2), 0.5)
    calls = []

    class Router:
        def supports_log2phy_fusion(self, *args):
            return fused

        def _select_experts_with_log2phy(self, *, log2phy, **kwargs):
            calls.append("fused")
            return weights, log2phy[logical]

        def _select_experts(self, **kwargs):
            calls.append("original")
            return weights, logical

    scope = dict(
        torch=torch,
        AscendFusedTopKRouter=Router,
        get_ascend_config=lambda: SimpleNamespace(enable_force_eplb=False),
        get_moe_num_logical_experts=lambda *args, **kwargs: 4,
    )
    select = load_method("vllm_ascend/ops/fused_moe/routed_experts.py", "AscendRoutedExperts", "_select_experts", scope)
    experts = SimpleNamespace(
        router=Router(),
        log2phy=mapping,
        n_shared_experts=shared,
        moe_config=SimpleNamespace(num_experts=4),
        global_redundant_expert_num=0,
        mix_placement=shared > 0,
    )
    actual_weights, ids = select(experts, torch.zeros(2, 4), torch.zeros(2, 4), False)
    torch.testing.assert_close(ids[:, :2], mapping[logical])
    torch.testing.assert_close(actual_weights[:, :2], weights)
    if shared:
        torch.testing.assert_close(ids[:, 2:], torch.tensor([[4, 5], [4, 5]], dtype=torch.int32))
        torch.testing.assert_close(actual_weights[:, 2:], torch.ones(2, shared))
    assert calls == ["fused" if fused else "original"]
