# SPDX-License-Identifier: Apache-2.0
"""Public pack validation must fail before a malformed launch is submitted."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.dcp.dcp_a2a import pack_dcp_output_lse


@pytest.mark.parametrize("world", [0, -1, True, 1.5, 3])
def test_invalid_world(world):
    output = torch.empty((4, 8, 16), device="npu", dtype=torch.bfloat16)
    lse = torch.empty((4, 8, 1), device="npu")
    with pytest.raises((ValueError, RuntimeError)):
        pack_dcp_output_lse(output, lse, world, 1)


@pytest.mark.parametrize("scatter", [-1, 2])
def test_invalid_scatter(scatter):
    output = torch.empty((4, 8, 16), device="npu", dtype=torch.bfloat16)
    lse = torch.empty((4, 8, 1), device="npu")
    with pytest.raises(ValueError, match="scatter_dim"):
        pack_dcp_output_lse(output, lse, 2, scatter)


@pytest.mark.parametrize("shape", [(0, 8, 16), (4, 0, 16), (4, 8, 0)])
def test_empty_dimensions(shape):
    output = torch.empty(shape, device="npu", dtype=torch.bfloat16)
    lse = torch.empty((*shape[:2], 1), device="npu")
    with pytest.raises(RuntimeError, match="non-empty"):
        pack_dcp_output_lse(output, lse, 2, 1)


@pytest.mark.parametrize("case", ["rank", "lse_shape", "lse_dtype", "output_dtype", "device", "cpu"])
def test_invalid_tensor_contract(case):
    output = torch.empty((4, 8, 16), device="npu", dtype=torch.bfloat16)
    lse = torch.empty((4, 8, 1), device="npu")
    if case == "rank":
        output = output[0]
    elif case == "lse_shape":
        lse = lse[..., 0]
    elif case == "lse_dtype":
        lse = lse.half()
    elif case == "output_dtype":
        output = output.int()
    elif case == "device":
        lse = lse.cpu()
    else:
        output, lse = output.cpu(), lse.cpu()
    with pytest.raises((RuntimeError, TypeError)):
        pack_dcp_output_lse(output, lse, 2, 1)


@pytest.mark.parametrize("case", ["shape", "dtype", "device"])
def test_invalid_stats_contract(case):
    output = torch.empty((4, 8, 16), device="npu", dtype=torch.bfloat16)
    lse = torch.empty((4, 8, 1), device="npu")
    total = torch.ones_like(lse)
    if case == "shape":
        total = total[..., 0]
    elif case == "dtype":
        total = total.half()
    else:
        total = total.cpu()
    with pytest.raises(ValueError, match="softmax_sum"):
        pack_dcp_output_lse(output, lse, 2, 1, softmax_sum=total)
