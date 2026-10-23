# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.model_executor.layers.rotary_embedding.mrope_vit_setup import VitMropeSetup, _vit_mrope_setup_kernel
from vllm.triton_utils import triton
from vllm.utils.torch_utils import async_tensor_h2d


@VitMropeSetup.register_oot
class AscendVitMropeSetup(VitMropeSetup):
    """ViT mRoPE setup with an Ascend-sized token tile."""

    def forward(
        self,
        inv_freq_t: torch.Tensor,
        inv_freq_h: torch.Tensor,
        inv_freq_w: torch.Tensor,
        grid_thw: list[list[int]],
        spatial_merge_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(3, total_N, half_rot_dim) fp32 cos/sin t/h/w planes for triton_mrope.

        One fused kernel: positions are computed arithmetically from the grids
        and the rotation comes from the per-axis inv_freq, so there is no
        cos/sin cache to size or grow.

        Args:
            inv_freq_t: t-axis inverse frequencies, (t_dim // 2,).
            inv_freq_h: h-axis inverse frequencies, (h_dim // 2,).
            inv_freq_w: w-axis inverse frequencies, (w_dim // 2,).
            grid_thw: per-grid [t, h, w] patch counts (host metadata).
            spatial_merge_size: spatial merge factor of the ViT.

        """
        device = inv_freq_t.device
        cu = [0]
        for t, h, w in grid_thw:
            cu.append(cu[-1] + t * h * w)
        n = cu[-1]
        g = len(grid_thw)
        # grid_thw is per-request host metadata, so one H2D is unavoidable;
        # pack cu_seqlens and grids into a single pinned async copy.
        buf = async_tensor_h2d(cu + [d for grid in grid_thw for d in grid], device=device, dtype=torch.int32)
        cu_seqlens = buf[: g + 1]
        grids = buf[g + 1 :].view(g, 3)

        half_t = inv_freq_t.numel()
        half_h = inv_freq_h.numel()
        half_w = inv_freq_w.numel()
        half_rot = half_t + half_h + half_w
        cos = torch.empty(3, n, half_rot, device=device, dtype=torch.float32)
        sin = torch.empty(3, n, half_rot, device=device, dtype=torch.float32)
        # Bound the token tile to fit Ascend A5 Unified Buffer capacity.
        block_n = 16
        _vit_mrope_setup_kernel[(triton.cdiv(n, block_n),)](
            inv_freq_t,
            inv_freq_h,
            inv_freq_w,
            cos,
            sin,
            cu_seqlens,
            grids,
            n,
            g,
            half_t=half_t,
            half_h=half_h,
            half_w=half_w,
            spatial_merge=spatial_merge_size,
            BLOCK_N=block_n,
            BLOCK_D=triton.next_power_of_2(half_rot),
            num_warps=8,
        )
        return cos, sin
