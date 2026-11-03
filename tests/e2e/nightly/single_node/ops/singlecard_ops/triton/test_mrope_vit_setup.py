# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.patch.worker.patch_mrope_vit_setup import _vit_mrope_setup_kernel, vit_mrope_setup


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
        ([[1, 1, 33]], 1),
        ([[1, 5, 7], [2, 3, 5]], 1),
        ([[1, 2, 2], [3, 6, 10]], 2),
        ([[0, 4, 4], [2, 4, 8], [0, 4, 4]], 2),
        ([[1, 48, 48]], 2),
        ([[4, 48, 64]], 2),
    ],
)
@pytest.mark.parametrize("sections", [(21, 21, 21), (16, 24, 24), (0, 16, 16), (32, 32, 32)])
@torch.inference_mode()
def test_vit_mrope_setup(grids, merge, sections):
    """Compile on NPU and compare both the kernel and patched public entry point."""
    init_device_properties_triton()
    inv_cpu = [torch.linspace(0.001, 1.0, size) for size in sections]
    inv = [freq.to("npu") for freq in inv_cpu]
    positions = _reference_positions(grids, merge)
    n = positions.shape[0]
    half_rot = sum(sections)
    cu = [0]
    for t, h, w in grids:
        cu.append(cu[-1] + t * h * w)
    metadata = torch.tensor(cu + [d for grid in grids for d in grid], dtype=torch.int32, device="npu")
    # NaN sentinels detect missing writes and writes outside axis sections.
    cos = torch.full((3, n, half_rot), float("nan"), dtype=torch.float32, device="npu")
    sin = torch.full_like(cos, float("nan"))
    _vit_mrope_setup_kernel[(triton.cdiv(n, 16),)](
        *inv,
        cos,
        sin,
        metadata[: len(cu)],
        metadata[len(cu) :],
        n,
        len(grids),
        half_t=sections[0],
        half_h=sections[1],
        half_w=sections[2],
        spatial_merge=merge,
        BLOCK_N=16,
        BLOCK_D=triton.next_power_of_2(half_rot),
        num_warps=8,
    )
    public_cos, public_sin = vit_mrope_setup(*inv, grids, merge)
    cos, sin, public_cos, public_sin = [value.cpu() for value in (cos, sin, public_cos, public_sin)]
    start = 0
    for axis, size in enumerate(sections):
        end = start + size
        freq = positions[:, axis, None] * inv_cpu[axis]
        for actual, expected in (
            (cos, freq.cos()),
            (sin, freq.sin()),
            (public_cos, freq.cos()),
            (public_sin, freq.sin()),
        ):
            torch.testing.assert_close(actual[axis, :, start:end], expected, atol=1e-5, rtol=1e-5)
        for output in (cos, sin):
            assert output[axis, :, :start].isnan().all()
            assert output[axis, :, end:].isnan().all()
        start = end
