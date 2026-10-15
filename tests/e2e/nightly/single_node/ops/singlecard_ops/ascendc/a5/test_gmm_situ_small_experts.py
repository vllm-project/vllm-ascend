# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check small-expert tiling and changing device routing under ACLGraph."""

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@pytest.mark.parametrize("experts", [14, 15])
@pytest.mark.parametrize("group_list_type", [0, 1])
@torch.inference_mode()
def test_small_expert_gmm_situ_graph(experts, group_list_type):
    torch.npu.set_device(0)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("requires A5 MXFP8/MXFP4 grouped matmul")
    torch.manual_seed(921 + experts)
    rows, capacity, k, n = 128, 17280, 3584, 6144
    weights, scales = [], []
    for _ in range(experts):
        weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="npu").view(torch.float4_e2m1fn_x2)
        torch_npu.npu_format_cast_(weight, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch.float4_e2m1fn_x2)
        weights.append(weight)
        scales.append(
            torch.randint(122, 126, (n, k // 64, 2), dtype=torch.uint8, device="npu").view(torch.float8_e8m0fnu)
        )
    x, x_scale = torch_npu.npu_dynamic_mx_quant(
        torch.randn(capacity, k, dtype=torch.bfloat16, device="npu") * 0.1,
        dst_type=torch.float8_e4m3fn,
    )
    x_scale = x_scale.view(torch.float8_e8m0fnu)
    balanced = [rows // experts + int(i < rows % experts) for i in range(experts)]
    routes = [
        balanced,
        [16] * 8 + [0] * (experts - 8),
        [17] + [16] * 6 + [15] + [0] * (experts - 8),
        [64] + [64 // (experts - 1) + int(i < 64 % (experts - 1)) for i in range(experts - 1)],
        [rows] + [0] * (experts - 1),
        balanced,
    ]
    groups = torch.empty(experts, dtype=torch.int64, device="npu")

    def set_groups(counts):
        value = torch.tensor(counts, dtype=torch.int64)
        if group_list_type == 0:
            value = value.cumsum(0)
        groups.copy_(value.npu())

    def fused():
        return torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz.list(
            x,
            weights,
            scales,
            None,
            None,
            x_scale,
            None,
            groups,
            dequant_mode=1,
            dequant_dtype=0,
            quant_mode=1,
            group_list_type=group_list_type,
            tuning_config=None,
            beta=4.0,
            linear_beta=25.0,
        )

    def reference():
        intermediate = torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[w.transpose(0, 1) for w in weights],
            scale=None,
            antiquant_scale=[s.transpose(0, 1) for s in scales],
            per_token_scale=[x_scale],
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=2,
            group_type=0,
            group_list=groups,
            group_list_type=group_list_type,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            output_dtype=torch.bfloat16,
        )[0]
        return torch.ops._C_ascend.situ_mx_quant(
            intermediate[:rows], beta=4.0, linear_beta=25.0, activate_left=True, dst_type=36
        )

    def check(actual, expected):
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a[:rows].view(torch.uint8).cpu(), b.view(torch.uint8).cpu(), rtol=0, atol=0)

    set_groups(balanced)
    check(fused(), reference())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = fused()
    # Cross the small-M boundary and the dual-role boundary without recapturing.
    for counts in routes:
        set_groups(counts)
        graph.replay()
        check(captured, reference())
