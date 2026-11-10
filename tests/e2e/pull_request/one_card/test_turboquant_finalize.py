# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical and graph coverage for framework TurboQuant cache packing."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.turboquant_finalize import turboquant_finalize
from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import CENTROIDS


@pytest.mark.parametrize("rows", [0, 1, 8, 257])
def test_fused_scale_and_packing(rows):
    init_device_properties_triton()
    torch.manual_seed(7)
    # Enumerate every packed byte; use zero and varying nonzero input norms.
    packed = torch.arange(256, dtype=torch.int32, device="npu").to(torch.uint8).repeat(rows, 1)
    norm = torch.arange(rows, dtype=torch.float16, device="npu")
    centroids = torch.tensor(CENTROIDS, dtype=torch.float32, device="npu")
    byte = torch.arange(256, device="npu")
    norm_lut = centroids[byte & 15].square() + centroids[byte >> 4].square()
    scale = (norm.float() / norm_lut[packed.long()].sum(-1).sqrt()).half()
    expected = torch.cat((packed, scale.view(torch.uint8).reshape(-1, 2)), dim=-1).unsqueeze(1)
    actual = turboquant_finalize(packed, norm, norm_lut)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    if rows:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = turboquant_finalize(packed, norm, norm_lut)
        norm.mul_(0.5)
        graph.replay()
        eager = turboquant_finalize(packed, norm, norm_lut)
        torch.testing.assert_close(captured.cpu(), eager.cpu(), rtol=0, atol=0)


def test_finalize_reuses_kernel_for_new_row_counts(monkeypatch):
    from vllm_ascend.ops.triton.turboquant_finalize import _finalize

    init_device_properties_triton()
    packed = torch.zeros((513, 256), dtype=torch.uint8, device="npu")
    norm = torch.ones(513, dtype=torch.float16, device="npu")
    norm_lut = torch.ones(256, dtype=torch.float32, device="npu")
    turboquant_finalize(packed[:1], norm[:1], norm_lut)

    def unexpected_compile(*args, **kwargs):
        pytest.fail("A new row count must reuse the compiled finalize kernel")

    monkeypatch.setattr(_finalize, "_do_compile", unexpected_compile)
    for rows in (8, 17, 256, 257, 513):
        actual = turboquant_finalize(packed[:rows], norm[:rows], norm_lut)
        scales = actual[:, 0, 256:].contiguous().view(torch.float16)
        torch.testing.assert_close(scales.cpu(), torch.full((rows, 1), 1 / 16, dtype=torch.float16))


@pytest.mark.parametrize("rows", [1, 17, 513])
def test_finalize_random_codes_match_scale_reference(rows):
    init_device_properties_triton()
    torch.manual_seed(73)
    packed = torch.randint(0, 256, (rows, 256), device="npu", dtype=torch.uint8)
    norm = (torch.rand(rows, device="npu") * 32).half()
    byte = torch.arange(256, device="npu")
    centroids = torch.tensor(CENTROIDS, dtype=torch.float32, device="npu")
    lut = centroids[byte & 15].square() + centroids[byte >> 4].square()
    result = turboquant_finalize(packed, norm, lut)
    expected_scale = (norm.float() / lut[packed.long()].sum(-1).sqrt()).half()
    torch.testing.assert_close(result[:, 0, :256].cpu(), packed.cpu(), rtol=0, atol=0)
    actual_scale = result[:, 0, 256:].contiguous().view(torch.float16).flatten()
    torch.testing.assert_close(actual_scale.cpu(), expected_scale.cpu(), rtol=0.001, atol=0)
