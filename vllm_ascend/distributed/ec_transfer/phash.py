# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""感知 hash (pHash) 计算与汉明距离工具, 用于 L2 resize cache 模糊匹配。

对 resize 后的 pixel_values 计算 64-bit pHash: 内容相似的图片得到
汉明距离相近的 hash, 从而可共享 encoder 缓存 (近似复用);
hamming=0 表示 pHash 相同, 仍需 SSIM 确认。

pHash 流程 (带 median 死区的 DCT 感知 hash):
  1. pixel_values (num_patches, patch_dim) 按 Qwen2/3-VL 处理器的
     patch 排列反解 merge 置换, 还原 (t, gh, gw) 灰度平面;
  2. t > 1 (视频) 按 t 取均值, 双线性插值到 32x32;
  3. DCT-II (预生成余弦矩阵做两次 matmul), 取左上 8x8 低频;
  4. 与 median 比较得 64 bit; 落在 median 死区内的位强制置 0,
     抑制阈值附近的位翻转; hash 碰撞与低距离假阳性仍需要
     grid 门控及 SSIM 精排过滤。

计算在输入设备上使用 float32; 仅将打包后的 8 字节 hash 返回 CPU。
CPU/NPU 浮点运算的舍入可能不同, 不保证跨设备逐位一致。
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

_PHASH_SIZE = 32    # DCT 输入边长
_HASH_SIDE = 8      # 取左上 8x8 低频 → 64 bit
# median 死区系数: |coef - median| <= _DEAD_ZONE * std 的位强制置 0
_DEAD_ZONE = 0.05
_DEVICE_CACHE_SIZE = 16
_BITS_PER_BYTE = 8
_HASH_BITS = _HASH_SIDE * _HASH_SIDE
# Number of equal-width pHash bands; use a positive divisor of 64.
# The similarity-cache Hamming limit is derived from this value.
PHASH_NUM_BANDS = 4


@lru_cache(maxsize=_DEVICE_CACHE_SIZE)
def _dct_matrix(device: torch.device | None = None) -> torch.Tensor:
    """32 点 DCT-II 矩阵 (float32, 按设备缓存)。"""
    if device is None:
        device = torch.device("cpu")
    n = _PHASH_SIZE
    k = torch.arange(n, dtype=torch.float32, device="cpu").unsqueeze(1)
    i = torch.arange(n, dtype=torch.float32, device="cpu").unsqueeze(0)
    m = torch.cos(math.pi / n * (i + 0.5) * k)
    m[0] *= 1.0 / math.sqrt(2.0)
    m = m.to(device=device)
    if device.type == "npu":
        # Publish a ready constant: later calls may use another NPU stream.
        torch.npu.current_stream(device).synchronize()
    return m


@lru_cache(maxsize=_DEVICE_CACHE_SIZE)
def _byte_weights(device: torch.device) -> torch.Tensor:
    weights = torch.tensor(
        [1 << bit for bit in reversed(range(_BITS_PER_BYTE))],
        dtype=torch.int32,
        device=device,
    )
    if device.type == "npu":
        torch.npu.current_stream(device).synchronize()
    return weights


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

    # 反解 Qwen2/3-VL 的 merge 置换: flat patch 顺序为
    # (t, gh/m, gw/m, m, m), permute 回 (t, gh/m, m, gw/m, m)
    # 即还原 2D patch 平面 (HF Qwen2VLImageProcessor 的逆操作)
    x = (
        resized.detach()
        .to(dtype=torch.float32)
        .reshape(t, gh // merge_size, gw // merge_size, merge_size, merge_size, -1)
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(t, gh, gw, -1)
    gray = x.mean(dim=-1)  # (t, gh, gw) patch 均值 → 灰度
    img = gray.mean(dim=0, keepdim=True)  # 视频多帧按帧均值 → (1, gh, gw)
    img = F.interpolate(img.unsqueeze(0), size=(_PHASH_SIZE, _PHASH_SIZE), mode="bilinear", align_corners=False)[0, 0]

    m = _dct_matrix(img.device)
    dct_low = (m @ img @ m.T)[:_HASH_SIDE, :_HASH_SIDE].flatten()

    median = dct_low.median()
    dead = _DEAD_ZONE * dct_low.std()
    bits = (dct_low > median) & ((dct_low - median).abs() > dead)
    packed = (
        (bits.reshape(-1, _BITS_PER_BYTE).to(torch.int32) * _byte_weights(bits.device))
        .sum(dim=1, dtype=torch.int32)
        .to(torch.uint8)
    )
    # The Python dictionary lookup needs a host integer. Transfer only one
    # packed hash, preserving the original MSB-first order and unsigned bit 63.
    return int.from_bytes(bytes(packed.cpu().tolist()), byteorder="big")


def hamming(a: int, b: int) -> int:
    """两个 pHash 的汉明距离。"""
    return (a ^ b).bit_count()


def bands(phash: int, num_bands: int = PHASH_NUM_BANDS) -> list[tuple[int, int]]:
    """LSH banding: 64 bit 切成 num_bands 段, 返回 (段号, 段值) 列表。

    查询候选 = 任一段 (段号, 段值) 完全相同的已登记 hash。汉明距离
    ≤ τ < num_bands 时由鸽笼原理必共享至少一段, 即 banding 不漏召;
    共享段不代表真相似, 误候选由调用方的 hamming 过滤兜底。
    """
    if (
        isinstance(num_bands, bool)
        or not isinstance(num_bands, int)
        or num_bands <= 0
        or _HASH_BITS % num_bands
    ):
        raise ValueError(f"num_bands must be a positive integer divisor of {_HASH_BITS}")
    width = _HASH_BITS // num_bands
    mask = (1 << width) - 1
    return [(i, (phash >> (i * width)) & mask) for i in range(num_bands)]
