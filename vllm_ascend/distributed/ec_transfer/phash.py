# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""感知 hash (pHash) 计算与汉明距离工具, 用于 L2 resize cache 模糊匹配。

对 resize 后的 pixel_values 计算 64-bit pHash: 内容相似的图片得到
汉明距离相近的 hash, 从而可共享 encoder 缓存 (近似复用); 逐位相同
的图片必得同一 hash, 精确匹配被自然包含 (hamming=0 的特例)。

pHash 流程 (与 imagehash.phash 一致):
  1. pixel_values (num_patches, patch_dim) 按 Qwen2/3-VL 处理器的
     patch 排列反解 merge 置换, 还原 (t, gh, gw) 灰度平面;
  2. t > 1 (视频) 按 t 取均值, 双线性插值到 32x32;
  3. DCT-II (预生成余弦矩阵做两次 matmul), 取左上 8x8 低频;
  4. 与 median 比较得 64 bit; 落在 median 死区内的位强制置 0,
     抑制阈值附近的位翻转; hash 碰撞与低距离假阳性仍需要
     grid 门控及可选 SSIM 精排过滤。

确定性: 全部 CPU float32, 同一输入张量必得同一 hash。
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

_PHASH_SIZE = 32   # DCT 输入边长
_HASH_SIDE = 8     # 取左上 8x8 低频 → 64 bit
# median 死区系数: |coef - median| <= _DEAD_ZONE * std 的位强制置 0
_DEAD_ZONE = 0.05


@lru_cache(maxsize=1)
def _dct_matrix() -> torch.Tensor:
    """32 点 DCT-II 矩阵 (float32, 进程内只生成一次)。"""
    n = _PHASH_SIZE
    k = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float32).unsqueeze(0)
    m = torch.cos(math.pi / n * (i + 0.5) * k)
    m[0] *= 1.0 / math.sqrt(2.0)
    return m


def compute_phash(
    resized: torch.Tensor,
    grid: tuple[int, ...],
    merge_size: int = 2,
) -> int | None:
    """由 (resize 后张量, grid_thw) 计算 64-bit pHash。

    grid 为 (t, h, w) patch 计数; 形状对不上 (num_patches != t*h*w
    或 h/w 不能被 merge_size 整除) 时返回 None, 调用方对该条目放弃
    L2 (行为与旧版无 L2 一致)。
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
    # 反解 Qwen2/3-VL 的 merge 置换: flat patch 顺序为
    # (t, gh/m, gw/m, m, m), permute 回 (t, gh/m, m, gw/m, m)
    # 即还原 2D patch 平面 (HF Qwen2VLImageProcessor 的逆操作)
    x = x.reshape(t, gh // merge_size, gw // merge_size,
                  merge_size, merge_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(t, gh, gw, -1)
    gray = x.mean(dim=-1)                 # (t, gh, gw) patch 均值 → 灰度
    img = gray.mean(dim=0, keepdim=True)  # 视频多帧按帧均值 → (1, gh, gw)
    img = F.interpolate(img.unsqueeze(0), size=(_PHASH_SIZE, _PHASH_SIZE),
                        mode="bilinear", align_corners=False)[0, 0]

    m = _dct_matrix()
    dct_low = (m @ img @ m.T)[:_HASH_SIDE, :_HASH_SIDE].flatten()

    median = dct_low.median()
    dead = _DEAD_ZONE * dct_low.std()
    bits = (dct_low > median) & ((dct_low - median).abs() > dead)
    phash = 0
    for b in bits.tolist():
        phash = (phash << 1) | int(b)
    return phash


def hamming(a: int, b: int) -> int:
    """两个 pHash 的汉明距离 (兼容 Python 3.9, 不用 int.bit_count)。"""
    return bin(a ^ b).count("1")


def bands(phash: int, num_bands: int = 8) -> list[tuple[int, int]]:
    """LSH banding: 64 bit 切成 num_bands 段, 返回 (段号, 段值) 列表。

    查询候选 = 任一段 (段号, 段值) 完全相同的已登记 hash。汉明距离
    ≤ τ < num_bands 时由鸽笼原理必共享至少一段, 即 banding 不漏召;
    共享段不代表真相似, 误候选由调用方的 hamming 过滤兜底。
    """
    width = 64 // num_bands
    mask = (1 << width) - 1
    return [(i, (phash >> (i * width)) & mask) for i in range(num_bands)]
