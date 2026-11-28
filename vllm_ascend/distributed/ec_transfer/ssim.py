# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""SSIM structural similarity utilities for similar-image decisions in L2
fuzzy matching (test helpers).

Computes SSIM over the (gh, gw) patch gray plane restored from resized
pixel_values (the merge inverse permutation matches compute_phash), as an
additional similar-image criterion beyond pHash, consumed by the SSIM matcher
of ECMemcacheConnector. The gray plane is rebuilt from the persisted resized
tensor according to its original grid before window convolution.

data_range rationale: pixel_values are normalized by the image processor
((x/255 - mean) / std, CLIP std ≈ 0.26~0.28), so a single channel has a
theoretical range of 1/std ≈ 3.6~3.8; the gray plane averages channels and
pixels within each patch, staying within that interval, hence the global fixed
L = 3.7. The SSIM stability constants C1=(K1·L)² / C2=(K2·L)² scale with L:
a larger L inflates scores (false positives, the most dangerous direction),
while a smaller L deflates them (only losing recall). The value must be
globally fixed; per-pair max-min adaptivity would base different image pairs
on different constants, making scores incomparable and the threshold
meaningless.

The threshold is an empirical starting point (default 0.99) to be calibrated
against the score distributions of positives (near duplicates from
recompression/resize round trips) and negatives (same layout, different
content); a false positive silently injects the candidate's embedding into
this image, so the threshold should lean toward "prefer a miss over a false
hit".
"""

from functools import lru_cache

import torch
import torch.nn.functional as F

from vllm_ascend.distributed.ec_transfer.similarity_cache_config import DEFAULT_DATA_RANGE as DEFAULT_DATA_RANGE

_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
_SSIM_WIN = 7  # Gaussian window size; actually min(win, gh, gw), abandoned below 3
_SSIM_SIGMA = 1.5
_KERNEL_CACHE_SIZE = 64


@lru_cache(maxsize=_KERNEL_CACHE_SIZE)
def _gaussian_kernel(win: int, device: torch.device | None = None) -> torch.Tensor:
    """win x win 2D Gaussian kernel (float32, cached per size and device)."""
    if device is None:
        device = torch.device("cpu")
    coords = torch.arange(win, dtype=torch.float32, device="cpu") - (win - 1) / 2
    g = torch.exp(-(coords**2) / (2 * _SSIM_SIGMA**2))
    g = g / g.sum()
    kernel = (g.unsqueeze(1) @ g.unsqueeze(0)).reshape(1, 1, win, win).to(device=device)
    if device.type == "npu":
        # Ensure other streams can safely reuse the cached constant.
        torch.npu.current_stream(device).synchronize()
    return kernel


def build_patch_gray_plane(
    resized: torch.Tensor,
    grid: tuple[int, ...],
    merge_size: int = 2,
) -> torch.Tensor | None:
    """Restore the (1, 1, gh, gw) patch gray plane from (resized tensor,
    grid_thw).

    Un-permutes the Qwen2/3-VL merge layout (same as compute_phash) and
    averages patch_dim for gray; video (t > 1) averages frames into a single
    frame. Returns None on shape mismatch, and the caller gives up L2 for that
    entry (same behavior as no L2).
    """
    if len(grid) < 3:
        return None
    t, gh, gw = int(grid[0]), int(grid[1]), int(grid[2])
    if t <= 0 or gh <= 0 or gw <= 0:
        return None
    if gh % merge_size or gw % merge_size:
        return None
    if resized.numel() == 0 or resized.shape[0] != t * gh * gw:
        return None
    x = (
        resized.detach()
        .to(dtype=torch.float32)
        .reshape(t, gh // merge_size, gw // merge_size, merge_size, merge_size, -1)
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(t, gh, gw, -1)
    gray = x.mean(dim=-1)  # (t, gh, gw) patch mean -> gray plane
    img = gray.mean(dim=0)  # average video frames -> (gh, gw)
    return img.unsqueeze(0).unsqueeze(0)


def ssim_score(
    a: torch.Tensor,
    b: torch.Tensor,
    data_range: float = DEFAULT_DATA_RANGE,
    win: int = _SSIM_WIN,
) -> float | None:
    """Mean SSIM between two (1, 1, gh, gw) gray planes.

    Returns None on shape mismatch (different grids make per-window comparison
    meaningless) or when a plane side is smaller than 3. Window convolution
    uses the valid region (edges cropped, matching the skimage default).
    """
    if a.shape != b.shape or a.device != b.device:
        return None
    h, w = a.shape[-2], a.shape[-1]
    win = min(win, h, w)
    if win < 3:
        return None

    c1 = (_SSIM_K1 * data_range) ** 2
    c2 = (_SSIM_K2 * data_range) ** 2
    a = a.detach().to(dtype=torch.float32)
    b = b.detach().to(dtype=torch.float32)
    kernel = _gaussian_kernel(win, a.device)

    mu_a = F.conv2d(a, kernel)
    mu_b = F.conv2d(b, kernel)
    mu_a_sq, mu_b_sq, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a = F.conv2d(a * a, kernel) - mu_a_sq
    sigma_b = F.conv2d(b * b, kernel) - mu_b_sq
    sigma_ab = F.conv2d(a * b, kernel) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / ((mu_a_sq + mu_b_sq + c1) * (sigma_a + sigma_b + c2))
    return float(ssim_map.mean().item())
