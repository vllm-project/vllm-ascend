# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

mrope_vit_setup = pytest.importorskip("vllm.model_executor.layers.rotary_embedding.mrope_vit_setup")
if not hasattr(mrope_vit_setup, "VitMropeSetup"):
    pytest.skip("Requires the pluggable ViT setup layer from vLLM #63017", allow_module_level=True)


def _reference_positions(grids, merge):
    positions = []
    for t, h, w in grids:
        axes = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
        positions.append(
            torch.stack(axes, dim=-1)
            .reshape(t, h // merge, merge, w // merge, merge, 3)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, 3)
        )
    return torch.cat(positions).float()


@pytest.mark.parametrize(
    "grids,merge",
    [
        ([[1, 1, 1]], 1),
        ([[1, 1, 17]], 1),
        ([[1, 2, 2], [3, 6, 10]], 2),
        ([[1, 48, 48]], 2),
    ],
)
@pytest.mark.parametrize("sections", [(13, 13, 13), (21, 21, 21)])
@torch.inference_mode()
def test_vit_mrope_setup(grids, merge, sections):
    """Resolve the upstream public layer on NPU and compare with CPU reference."""
    init_device_properties_triton()
    inv_cpu = [1.0 / (10000.0 ** (torch.arange(size, dtype=torch.float32) / size)) for size in sections]
    inv = [freq.to("npu") for freq in inv_cpu]
    positions = _reference_positions(grids, merge)
    setup = mrope_vit_setup.VitMropeSetup()
    assert type(setup).__module__ == "vllm_ascend.ops.mrope_vit_setup"
    cos, sin = setup(inv[0], inv[1], inv[2], grids, merge)
    expected_shape = (3, positions.shape[0], sum(sections))
    assert cos.shape == sin.shape == expected_shape
    cos, sin = cos.cpu(), sin.cpu()
    start = 0
    for axis, size in enumerate(sections):
        end = start + size
        freq = positions[:, axis, None] * inv_cpu[axis]
        for actual, expected in (
            (cos, freq.cos()),
            (sin, freq.sin()),
        ):
            torch.testing.assert_close(actual[axis, :, start:end], expected, atol=1e-5, rtol=1e-5)
        start = end
