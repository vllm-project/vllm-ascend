# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared SiTU output contracts; these tests allocate only Meta tensors."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend.vllm_ascend_C")


@pytest.mark.parametrize("msd", [False, True], ids=["a5_mx", "a3_msd"])
@pytest.mark.parametrize("capacity", [0, 8])
@pytest.mark.parametrize("experts", [1, 3])
@pytest.mark.parametrize("expert_list", [False, True])
@pytest.mark.parametrize("nz", [False, True])
def test_shared_situ_meta_contract(msd, capacity, experts, expert_list, nz):
    k, n = 320, 768
    with torch.device("meta"):
        x = torch.empty(capacity, k, dtype=torch.int8 if msd else torch.float8_e4m3fn)
        weight = torch.empty(
            (experts, k, n // 8) if msd else (experts, n, k // 2),
            dtype=torch.int32 if msd else torch.float4_e2m1fn_x2,
        )
        scale = torch.empty(
            (experts, n) if msd else (experts, n, k // 64, 2),
            dtype=torch.int64 if msd else torch.float8_e8m0fnu,
        )
        x_scale = torch.empty(
            (capacity,) if msd else (capacity, k // 64, 2),
            dtype=torch.float32 if msd else torch.float8_e8m0fnu,
        )
        groups = torch.empty(experts, dtype=torch.int64)
    op = (
        torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz if nz else torch.ops._C_ascend.grouped_matmul_situ_quant
    )
    if expert_list:
        op, weight, scale = op.list, list(weight.unbind()), list(scale.unbind())
    y, y_scale = op(
        x, weight, scale, None, None, x_scale, None, groups, int(not msd), 0, int(not msd), 1, None, 4.0, 25.0
    )
    assert y.device.type == y_scale.device.type == "meta"
    assert y.shape == (capacity, n // 2)
    assert y.dtype == (torch.int8 if msd else torch.float8_e4m3fn)
    assert y_scale.shape == ((capacity,) if msd else (capacity, n // 128, 2))
    assert y_scale.dtype == (torch.float32 if msd else torch.float8_e8m0fnu)
