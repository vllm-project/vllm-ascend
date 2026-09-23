# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, DeviceOperator
from vllm_ascend.utils import enable_custom_op

pytestmark = pytest.mark.skipif(
    not DeviceOperator.supports_moe_gating_top_k_log2phy,
    reason="TopK with logical-to-physical mapping requires A5",
)


def _topk(x, bias, mapping=None, *, norm_type=1, groups=1, selected_groups=1, scale=1.0):
    return A5DeviceAdaptor.moe_gating_top_k(
        x,
        k=16,
        k_group=selected_groups,
        group_count=groups,
        group_select_mode=1,
        renorm=1,
        norm_type=norm_type,
        out_flag=False,
        routed_scaling_factor=scale,
        bias_opt=bias,
        log2phy=mapping,
    )


@pytest.mark.parametrize("rows", [8, 32, 64])
@pytest.mark.parametrize("map_kind", ["identity", "permutation", "redundant"])
def test_kimi_topk_with_map_matches_native(rows, map_kind):
    enable_custom_op()
    torch.manual_seed(17)
    x = torch.randn(rows, 896, dtype=torch.float32, device="npu")
    bias = torch.randn(896, dtype=torch.float32, device="npu")
    if map_kind == "identity":
        mapping = torch.arange(896, dtype=torch.int32, device="npu")
    elif map_kind == "permutation":
        mapping = torch.randperm(896, dtype=torch.int32, device="npu")
    else:
        # Values beyond the logical expert count expose an accidental second map.
        mapping = torch.randperm(960, dtype=torch.int32, device="npu")[:896].contiguous()
    old_weights, logical_ids, _ = _topk(x, bias)
    weights, physical_ids, _ = _topk(x, bias, mapping)
    torch.testing.assert_close(weights, old_weights, rtol=0, atol=0)
    torch.testing.assert_close(physical_ids, mapping[logical_ids], rtol=0, atol=0)
    # The mapped operator must not shadow the native ordinary TopK symbol.
    after_weights, after_ids, _ = _topk(x, bias)
    torch.testing.assert_close(after_weights, old_weights, rtol=0, atol=0)
    torch.testing.assert_close(after_ids, logical_ids, rtol=0, atol=0)


def test_topk_with_map_reads_updated_table_on_graph_replay():
    enable_custom_op()
    torch.manual_seed(23)
    x = torch.randn(8, 896, dtype=torch.float32, device="npu")
    bias = torch.randn(896, dtype=torch.float32, device="npu")
    mapping = torch.arange(896, dtype=torch.int32, device="npu")
    reference_weights, logical_ids, _ = _topk(x, bias)
    _topk(x, bias, mapping)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        weights, physical_ids, _ = _topk(x, bias, mapping)
    for _ in range(3):
        mapping.copy_(torch.randperm(960, dtype=torch.int32, device="npu")[:896])
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(weights, reference_weights, rtol=0, atol=0)
        torch.testing.assert_close(physical_ids, mapping[logical_ids], rtol=0, atol=0)


@pytest.mark.parametrize("norm_type", [0, 1])
@pytest.mark.parametrize(("groups", "selected_groups"), [(1, 1), (8, 4)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_topk_with_map_preserves_ordinary_routing(norm_type, groups, selected_groups, has_bias):
    enable_custom_op()
    torch.manual_seed(31)
    x = torch.randn(8, 896, dtype=torch.float32, device="npu")
    bias = torch.randn(896, dtype=torch.float32, device="npu") if has_bias else None
    mapping = torch.randperm(960, dtype=torch.int32, device="npu")[:896].contiguous()
    kwargs = dict(norm_type=norm_type, groups=groups, selected_groups=selected_groups, scale=2.5)
    old_weights, logical_ids, _ = _topk(x, bias, **kwargs)
    weights, physical_ids, _ = _topk(x, bias, mapping, **kwargs)
    torch.testing.assert_close(weights, old_weights, rtol=0, atol=0)
    torch.testing.assert_close(physical_ids, mapping[logical_ids], rtol=0, atol=0)


@pytest.mark.parametrize("all_zero", [False, True])
def test_topk_with_map_preserves_native_tie_order(all_zero):
    enable_custom_op()
    x = torch.arange(896, device="npu").remainder(7).float().expand(8, -1).contiguous()
    if all_zero:
        x.zero_()
    bias = torch.zeros(896, device="npu")
    mapping = torch.randperm(960, dtype=torch.int32, device="npu")[:896].contiguous()
    old_weights, logical_ids, _ = _topk(x, bias)
    weights, physical_ids, _ = _topk(x, bias, mapping)
    torch.testing.assert_close(weights, old_weights, rtol=0, atol=0)
    torch.testing.assert_close(physical_ids, mapping[logical_ids], rtol=0, atol=0)
