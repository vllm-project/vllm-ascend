# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.triton.sfa_cp import (
    pack_sfa_dcp_output_max_sum,
    sfa_dcp_a2a_fused_max_sum_fake,
)


@pytest.mark.parametrize(
    ("statistics_shape", "message"),
    [
        ((2, 4, 4), "PA_BSND max/sum"),
        ((1, 5, 8), "PA_BSND max/sum"),
    ],
)
def test_max_sum_pack_rejects_non_pa_bsnd_statistics(statistics_shape, message) -> None:
    output = torch.empty(4, 8, 16)
    softmax_max = torch.empty(statistics_shape, dtype=torch.float32)
    softmax_sum = torch.empty_like(softmax_max)

    with pytest.raises(RuntimeError, match=message):
        pack_sfa_dcp_output_max_sum(output, softmax_max, softmax_sum, 2, 1)


def test_max_sum_pack_requires_fp32_statistics() -> None:
    output = torch.empty(4, 8, 16)
    softmax_max = torch.empty(1, 4, 8, dtype=torch.float16)
    softmax_sum = torch.empty_like(softmax_max)

    with pytest.raises(TypeError, match="requires float32 PA_BSND max/sum"):
        pack_sfa_dcp_output_max_sum(output, softmax_max, softmax_sum, 2, 1)


@pytest.mark.parametrize(
    ("scatter_dim", "expected_shape"),
    [(0, (2, 8, 16)), (1, (8, 2, 16))],
)
def test_max_sum_fake_preserves_local_shape_dtype_and_device(scatter_dim, expected_shape) -> None:
    output = torch.empty(8, 8, 16, dtype=torch.bfloat16, device="meta")
    softmax_max = torch.empty(1, 8, 8, dtype=torch.float32, device="meta")
    softmax_sum = torch.empty_like(softmax_max)

    actual = sfa_dcp_a2a_fused_max_sum_fake(
        output,
        softmax_max,
        softmax_sum,
        4,
        scatter_dim,
        "dcp:0",
    )

    assert actual.shape == expected_shape
    assert actual.dtype == output.dtype
    assert actual.device == output.device


@pytest.mark.parametrize("scatter_dim", [0, 1])
def test_registered_max_sum_custom_op_uses_fake_implementation(scatter_dim) -> None:
    output = torch.empty(8, 8, 16, dtype=torch.float16, device="meta")
    softmax_max = torch.empty(1, 8, 8, dtype=torch.float32, device="meta")
    softmax_sum = torch.empty_like(softmax_max)

    actual = torch.ops.vllm.sfa_dcp_a2a_fused_max_sum(
        output,
        softmax_max,
        softmax_sum,
        4,
        scatter_dim,
        "dcp:0",
    )

    expected_shape = (2, 8, 16) if scatter_dim == 0 else (8, 2, 16)
    assert actual.shape == expected_shape
    assert actual.dtype == output.dtype
    assert actual.device == output.device
