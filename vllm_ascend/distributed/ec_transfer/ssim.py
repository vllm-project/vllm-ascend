# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""SSIM 结构相似度工具, 用于 L2 模糊匹配的相似图判定 (测试用)。

在 resize 后 pixel_values 还原出的 (gh, gw) patch 灰度平面上计算
SSIM (merge 逆置换与 compute_phash 一致), 作为 pHash 之外的另一种
相似图判据, 供 ECMemcacheConnector 的 SSIM matcher 使用。灰度平面
由持久化的 resized tensor 按其原始 grid 重建, 再进行窗口卷积。

data_range 的设定: pixel_values 经 image processor 归一化
((x/255 - mean) / std, CLIP std ≈ 0.26~0.28), 单通道理论值域为
1/std ≈ 3.6~3.8; 灰度平面是通道与 patch 内像素的均值, 值域不超出
该区间, 故取全局固定 L = 3.7。SSIM 稳定常数 C1=(K1·L)² / C2=(K2·L)²
随 L 缩放: L 偏大 → 分数虚高 (假阳性, 最危险方向), L 偏小 → 分数
虚低 (只损失命中率)。必须取全局固定值, 不能用逐对图片的 max-min
自适应 —— 那会让不同图片对的分数基于不同常数, 彼此不可比, 阈值失效。

阈值是经验起点 (默认 0.99), 需按正例 (重压缩/resize 往返的近重复)
与负例 (同版式不同内容) 的分数分布标定后调整; 假阳性会把候选图的
embedding 静默注入给本图, 阈值应偏向"宁可 miss 不可错命中"。
"""

from functools import lru_cache

import torch
import torch.nn.functional as F

_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
_SSIM_WIN = 7       # 高斯窗边长; 实际取 min(win, gh, gw), 不足 3 放弃
_SSIM_SIGMA = 1.5

# 归一化 pixel_values 的单通道理论值域 1/std (CLIP std ≈ 0.26~0.28)
DEFAULT_DATA_RANGE = 3.7


@lru_cache(maxsize=8)
def _gaussian_kernel(win: int) -> torch.Tensor:
    """win x win 二维高斯核 (float32, 按边长缓存)。"""
    coords = torch.arange(win, dtype=torch.float32) - (win - 1) / 2
    g = torch.exp(-(coords**2) / (2 * _SSIM_SIGMA**2))
    g = g / g.sum()
    return (g.unsqueeze(1) @ g.unsqueeze(0)).reshape(1, 1, win, win)


def build_patch_gray_plane(
    resized: torch.Tensor,
    grid: tuple[int, ...],
    merge_size: int = 2,
) -> torch.Tensor | None:
    """由 (resize 后张量, grid_thw) 还原 (1, 1, gh, gw) patch 灰度平面。

    反解 Qwen2/3-VL 的 merge 置换 (与 compute_phash 相同), 对
    patch_dim 取均值得灰度; 视频 (t > 1) 按帧均值压成单帧。形状对不上
    时返回 None, 调用方对该条目放弃 L2 (行为与无 L2 一致)。
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

    x = resized.detach().to(device="cpu", dtype=torch.float32)
    x = x.reshape(t, gh // merge_size, gw // merge_size,
                  merge_size, merge_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(t, gh, gw, -1)
    gray = x.mean(dim=-1)     # (t, gh, gw) patch 均值 → 灰度
    img = gray.mean(dim=0)    # 视频多帧按帧均值 → (gh, gw)
    return img.unsqueeze(0).unsqueeze(0)


def ssim_score(
    a: torch.Tensor,
    b: torch.Tensor,
    data_range: float = DEFAULT_DATA_RANGE,
    win: int = _SSIM_WIN,
) -> float | None:
    """两张 (1, 1, gh, gw) 灰度平面的均值 SSIM。

    形状不一致 (grid 不同, 逐窗口比较无意义) 或平面边长不足 3 时
    返回 None。窗口卷积取 valid 区域 (边缘裁剪, 与 skimage 默认一致)。
    """
    if a.shape != b.shape:
        return None
    h, w = a.shape[-2], a.shape[-1]
    win = min(win, h, w)
    if win < 3:
        return None

    c1 = (_SSIM_K1 * data_range) ** 2
    c2 = (_SSIM_K2 * data_range) ** 2
    kernel = _gaussian_kernel(win)

    mu_a = F.conv2d(a, kernel)
    mu_b = F.conv2d(b, kernel)
    mu_a_sq, mu_b_sq, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a = F.conv2d(a * a, kernel) - mu_a_sq
    sigma_b = F.conv2d(b * b, kernel) - mu_b_sq
    sigma_ab = F.conv2d(a * b, kernel) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a_sq + mu_b_sq + c1) * (sigma_a + sigma_b + c2))
    return float(ssim_map.mean().item())
