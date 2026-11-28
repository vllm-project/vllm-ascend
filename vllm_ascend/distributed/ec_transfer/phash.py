# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Perceptual hash (pHash) computation and Hamming distance utilities for
fuzzy matching in the L2 resized cache.

Computes a 64-bit pHash from the resized pixel_values: images with similar
content get hashes with a small Hamming distance, so encoder cache entries can
be shared (approximate reuse); hamming=0 means an identical pHash, which still
requires SSIM confirmation.

pHash pipeline (DCT perceptual hash with a median dead zone):
  1. pixel_values (num_patches, patch_dim) are un-permuted from the
     Qwen2/3-VL processor's merge layout back into (t, gh, gw) gray planes;
  2. t > 1 (video) averages frames, then bilinearly resizes to 32x32;
  3. DCT-II (two matmuls with a precomputed cosine matrix), keeping the
     top-left 8x8 low-frequency block;
  4. Compare against the median for 64 bits; bits within the median dead zone
     are forced to 0 to suppress bit flips near the threshold. Hash collisions
     and low-distance false positives still require grid gating and SSIM
     re-ranking.

Computation uses float32 on the input device; only the packed 8-byte hash is
returned to the CPU. CPU/NPU floating-point rounding may differ, so bitwise
identity across devices is not guaranteed.
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from vllm_ascend.distributed.ec_transfer.phash_index import (
    PHASH_NUM_BANDS as PHASH_NUM_BANDS,
)
from vllm_ascend.distributed.ec_transfer.phash_index import (
    bands as bands,
)
from vllm_ascend.distributed.ec_transfer.phash_index import (
    hamming as hamming,
)

_PHASH_SIZE = 32  # DCT input size
_HASH_SIDE = 8  # Keep the top-left 8x8 low frequencies -> 64 bits
# Median dead-zone ratio: bits with |coef - median| <= _DEAD_ZONE * std are
# forced to 0
_DEAD_ZONE = 0.05
_DEVICE_CACHE_SIZE = 16
_BITS_PER_BYTE = 8
_HASH_BITS = _HASH_SIDE * _HASH_SIDE


@lru_cache(maxsize=_DEVICE_CACHE_SIZE)
def _dct_matrix(device: torch.device | None = None) -> torch.Tensor:
    """32-point DCT-II matrix (float32, cached per device)."""
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
    """Compute a 64-bit pHash from (resized tensor, grid_thw).

    grid holds (t, h, w) patch counts; returns None on shape mismatch
    (num_patches != t*h*w, or h/w not divisible by merge_size), and the caller
    gives up L2 for that entry (same behavior as the legacy path without L2).
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

    # Undo the Qwen2/3-VL merge permutation: the flat patch order is
    # (t, gh/m, gw/m, m, m); permuting back to (t, gh/m, m, gw/m, m)
    # restores the 2D patch planes (the inverse of HF Qwen2VLImageProcessor).
    x = (
        resized.detach()
        .to(dtype=torch.float32)
        .reshape(t, gh // merge_size, gw // merge_size, merge_size, merge_size, -1)
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(t, gh, gw, -1)
    gray = x.mean(dim=-1)  # (t, gh, gw) patch mean -> gray plane
    img = gray.mean(dim=0, keepdim=True)  # average video frames -> (1, gh, gw)
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
