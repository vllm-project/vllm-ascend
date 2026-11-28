# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real BF16 output projections with ND and NZ weights, including graph replay."""

import pytest
import torch
import torch_npu


@pytest.mark.parametrize("projection", ["wo_a", "wo_b"])
@torch.inference_mode()
def test_bf16_output_projection_nz(projection):
    torch.manual_seed(123)
    previous = torch.npu.config.allow_internal_format
    torch.npu.config.allow_internal_format = True
    try:
        if projection == "wo_a":
            x = torch.randn((48, 8, 4096), dtype=torch.bfloat16, device="npu") * 0.1
            weight = torch.randn((8, 4096, 1024), dtype=torch.bfloat16, device="npu") * 0.1

            def project(w):
                return torch_npu.npu_transpose_batchmatmul(
                    x,
                    w,
                    perm_x1=(1, 0, 2),
                    perm_x2=(0, 1, 2),
                    perm_y=(1, 0, 2),
                    batch_split_factor=1,
                )
        else:
            x = torch.randn((48, 8192), dtype=torch.bfloat16, device="npu") * 0.1
            weight = torch.randn((5120, 8192), dtype=torch.bfloat16, device="npu") * 0.1

            def project(w):
                return torch.nn.functional.linear(x, w)

        nz_weight = torch_npu.npu_format_cast(weight, 29)
        assert torch_npu.get_npu_format(nz_weight) == 29
        torch.testing.assert_close(project(nz_weight), project(weight), rtol=0.01, atol=0.001)
        for _ in range(3):
            project(nz_weight)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = project(nz_weight)
        for seed in (501, 502):
            torch.manual_seed(seed)
            x.normal_(mean=0.0, std=0.1)
            expected = project(weight)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output, expected, rtol=0.01, atol=0.001)
    finally:
        torch.npu.config.allow_internal_format = previous
