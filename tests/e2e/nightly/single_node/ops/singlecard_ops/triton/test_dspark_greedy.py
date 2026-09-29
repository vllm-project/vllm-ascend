# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm_ascend.ops.triton.v2.greedy import (
    sample_greedy_markov,
    scratch_shape,
)


@pytest.mark.parametrize(
    ("vocab", "input_dtype", "output_dtype"),
    [
        (1, torch.float32, torch.int32),
        (1, torch.float32, torch.int64),
        (2048, torch.float16, torch.int64),
        (2049, torch.float16, torch.int32),
        (2049, torch.float16, torch.int64),
        (16385, torch.bfloat16, torch.int32),
        (16385, torch.bfloat16, torch.int64),
        (32769, torch.float32, torch.int64),
    ],
)
def test_fused_greedy_matches_materialized_argmax(
    vocab: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    rows = 3

    base_storage = torch.full((rows, vocab + 3), -10, dtype=input_dtype, device="npu")
    bias_storage = torch.zeros((rows, vocab + 5), dtype=input_dtype, device="npu")
    base = base_storage[:, :vocab]
    bias = bias_storage[:, :vocab]
    assert base.stride(1) == bias.stride(1) == 1

    base[0, -1] = 1
    bias[0, -1] = 3

    base[1, 0] = 3
    bias[1, 0] = 4
    base[1, -1] = 5
    bias[1, -1] = 2

    if vocab > 2048:
        base[2, 2047] = 5
        bias[2, 2047] = 4
        base[2, 2048] = 6
        bias[2, 2048] = 3
    else:
        base[2, 0] = 2
        bias[2, 0] = 3

    output_storage = torch.full((rows, 2), -1, dtype=output_dtype, device="npu")
    output = output_storage[:, 1]
    assert output.stride(0) == 2

    shape = scratch_shape(rows, vocab)
    partial_values = torch.empty(shape, dtype=torch.float32, device="npu")
    partial_indices = torch.empty(shape, dtype=torch.int32, device="npu")

    sample_greedy_markov(
        base,
        bias,
        output,
        partial_values,
        partial_indices,
    )

    expected = (base + bias).cpu().argmax(dim=1).to(output_dtype)

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output_storage[:, 0].cpu(),
        torch.full((rows,), -1, dtype=output_dtype),
        rtol=0,
        atol=0,
    )
