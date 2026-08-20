# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TurboQuant KV-cache codec (arXiv 2504.19874) for Ascend 310P.

Quantizes each head_dim vector to b bits:
    n = ||x||               -> stored fp16, one scalar per (token, kv_head)
    y = Pi @ (x / n)        -> Pi = D*H*D, symmetric AND self-inverse, O(d log d)
    idx = quantize(y)       -> b-bit index against a fixed codebook

Two properties make this cheap on device, both exploited here:

1. `Pi` is its own inverse, so a single matrix serves both directions.
2. `Pi` never needs to touch the cache. Because it is orthogonal,
       <q, n*Pi@yhat>            == n * <Pi@q, yhat>
       sum_i p_i*n_i*Pi@yhat_i   == Pi @ (sum_i p_i*n_i*yhat_i)
   so attention can run entirely in the ROTATED basis: rotate the query once,
   run ordinary attention against dequantized-but-still-rotated K/V, then rotate
   the output once. Per cached token the cost is a codebook lookup, not a matmul.

Codebook is UNIFORM (affine `scale * (idx - offset)`) rather than Lloyd-Max:
measured on Qwen3.5-4B this costs +0.01 equivalent bits at b=3 while turning
dequantization from a gather into a multiply.
"""

import math

import torch

# MSE-optimal uniform step for a unit-variance Gaussian, by bit-width. After the
# rotation each coordinate is ~N(0, 1/d), so the codebook is (step * (i - (2^b-1)/2)) / sqrt(d).
UNIFORM_STEP = {2: 0.995686, 3: 0.586020, 4: 0.335201, 5: 0.188138}

# Lloyd-Max centroids for N(0,1); kept for parity checks against the paper's
# Theorem 1 distortion values (0.36 / 0.117 / 0.03 / 0.009 at b=1..4).
LLOYDMAX = {
    2: [-1.510417, -0.452780, 0.452780, 1.510417],
    3: [-2.151944, -1.343909, -0.756005, -0.245094, 0.245094, 0.756005, 1.343909, 2.151944],
    4: [-2.732588, -2.069016, -1.618046, -1.256231, -0.942340, -0.656759, -0.388048, -0.128395,
        0.128395, 0.388048, 0.656759, 0.942340, 1.256231, 1.618046, 2.069016, 2.732588],
}

NZ_LAST_DIM = 32  # FRACTAL_NZ C0 for int8 on 310P; packed head_size must be a multiple of this


def hadamard(n: int, device, dtype) -> torch.Tensor:
    """Normalized Sylvester-Hadamard. Symmetric, and H @ H == I."""
    if n & (n - 1):
        raise ValueError(f"Hadamard requires a power-of-2 dimension, got {n}")
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def packed_bytes(head_dim: int, bits: int) -> int:
    """Bytes per vector once packed.

    Must be EVEN: the packed plane is stored in an fp16-typed NZ cache and
    reinterpreted byte-wise (`_npu_reshape_and_cache` is a pure scatter and moves
    the bytes verbatim -- verified byte-exact on 310P). The per-head NZ alignment
    constraint depends on num_kv_heads and is checked by the backend.
    """
    total = head_dim * bits
    if total % 8:
        raise ValueError(f"head_dim*bits must be byte-aligned, got {total} bits")
    nbytes = total // 8
    if nbytes % 2:
        raise ValueError(f"packed size {nbytes}B must be even to view as fp16")
    return nbytes


class TurboQuantCodec:
    """Quantize/pack and unpack/dequantize head_dim vectors, staying on-device.

    All tensors are shaped (..., head_dim) on input and (..., packed_bytes) when packed.
    Dequantization deliberately returns vectors in the ROTATED basis; callers rotate
    the query and the attention output instead (see module docstring).
    """

    def __init__(self, head_dim: int, bits: int, device, seed: int = 0,
                 codebook: str = "uniform", compute_dtype: torch.dtype = torch.float32):
        if bits not in UNIFORM_STEP:
            raise ValueError(f"unsupported bit-width {bits}")
        self.head_dim = head_dim
        self.bits = bits
        self.nbytes = packed_bytes(head_dim, bits)
        self.cdt = compute_dtype
        self.levels = 1 << bits

        if codebook == "uniform":
            step = UNIFORM_STEP[bits]
            cb = torch.tensor([(i - (self.levels - 1) / 2.0) * step for i in range(self.levels)],
                              device=device, dtype=compute_dtype)
        elif codebook == "lloydmax":
            cb = torch.tensor(LLOYDMAX[bits], device=device, dtype=compute_dtype)
        else:
            raise ValueError(codebook)
        self.codebook = cb / math.sqrt(head_dim)
        self.boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2

        # Pi = D @ H @ D : symmetric and self-inverse. Seeded on (seed, head_dim) ONLY --
        # never on bit-width, or configs draw different rotations and cross-config
        # comparisons pick up ~1% PPL of rotation-draw variance.
        g = torch.Generator(device="cpu").manual_seed(seed + head_dim * 1000)
        signs = (torch.randint(0, 2, (head_dim,), generator=g, dtype=torch.int8) * 2 - 1)
        d = signs.to(device=device, dtype=compute_dtype)
        self.pi = d.unsqueeze(1) * hadamard(head_dim, device, compute_dtype) * d.unsqueeze(0)
        # SINGLE SOURCE OF TRUTH for the rotation. Tier 0 uses `pi` directly;
        # the Tier-2 AscendC kernels take this raw sign vector and rebuild
        # Pi = D@H@D on device. If the two ever drifted, keys written by one
        # path and read by the other would decode to garbage, so both must
        # come from this one draw.
        self.signs = d

        # (idx >> shift) & mask extracts each sub-byte code; see pack()/unpack().
        self._shifts = torch.arange(0, 8 * (self.nbytes // (head_dim * bits // 8)), device=device)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Pi. Self-inverse, so this is both the forward and inverse rotation."""
        return (x.to(self.cdt) @ self.pi).to(x.dtype)

    def quantize(self, x: torch.Tensor):
        """(..., head_dim) float -> (packed int8 (..., nbytes), norms fp16 (...,))."""
        v = x.to(self.cdt)
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y = (v / n) @ self.pi
        idx = torch.bucketize(y, self.boundaries)  # 0 .. levels-1
        return self.pack(idx), n.squeeze(-1).to(torch.float16)

    def dequantize(self, packed: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """(packed, norms) -> (..., head_dim) fp16 in the ROTATED basis."""
        idx = self.unpack(packed)
        return (self.codebook[idx.long()] * norms.to(self.cdt).unsqueeze(-1)).to(torch.float16)

    def pack(self, idx: torch.Tensor) -> torch.Tensor:
        """(..., head_dim) codes -> (..., nbytes) int8, little-endian bit order."""
        b, per_word = self.bits, 8  # 8 codes -> b bytes (b*8 bits), exact for b in {2,3,4}
        x = idx.reshape(*idx.shape[:-1], -1, per_word).to(torch.int32)
        acc = torch.zeros(x.shape[:-1], dtype=torch.int32, device=idx.device)
        for i in range(per_word):
            acc = acc | ((x[..., i] & (self.levels - 1)) << (b * i))
        out = torch.stack([(acc >> (8 * j)) & 0xFF for j in range(b)], dim=-1)
        return out.reshape(*idx.shape[:-1], self.nbytes).to(torch.uint8).view(torch.int8)

    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        """(..., nbytes) int8 -> (..., head_dim) codes."""
        b, per_word = self.bits, 8
        p = packed.view(torch.uint8).reshape(*packed.shape[:-1], -1, b).to(torch.int32)
        acc = torch.zeros(p.shape[:-1], dtype=torch.int32, device=packed.device)
        for j in range(b):
            acc = acc | (p[..., j] << (8 * j))
        codes = torch.stack([(acc >> (b * i)) & (self.levels - 1) for i in range(per_word)], dim=-1)
        return codes.reshape(*packed.shape[:-1], self.head_dim)

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize->dequantize->inverse-rotate, back in the ORIGINAL basis. For validation."""
        packed, norms = self.quantize(x)
        return self.rotate(self.dequantize(packed, norms))
