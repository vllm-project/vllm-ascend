"""Python helpers for the bundled SplitQuant MoE operator."""

import os
from functools import lru_cache

import torch
import torch.nn.functional as F

from .w4a4_splitquant import load_splitquant_operator


@lru_cache(None)
def silu_table(device_index, dtype):
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    return F.silu(bits.to(f"npu:{device_index}")).contiguous()


def prepare(up_packed, up_scale, down_codes, down_scale, cp, cn, hcp, hcn,
            rotation, dtype):
    load_splitquant_operator()
    e, h, i = down_codes.shape
    down = (down_codes.to(dtype).view(e, h, i // 128, 128)
            * down_scale.to(dtype).unsqueeze(-1)).reshape(e, h, i).contiguous()
    if os.getenv("SPLITQUANT_DOWN_NZ", "0") == "1":
        down = down.reshape(e, h // 16, 16, i // 16, 16).permute(0, 1, 3, 4, 2).contiguous()
    group_packed = pack_up_nz(up_packed) if os.getenv("SPLITQUANT_UP_NZ", "0") == "1" else up_packed.view(e, 2 * i, h // 128, 64).contiguous()
    return (group_packed, up_scale.float().transpose(1, 2).contiguous(), down,
            cp.float().contiguous(), cn.float().contiguous(),
            hcp.float().contiguous(), hcn.float().contiguous(),
            rotation.to(dtype).contiguous(), silu_table(down.device.index, dtype))


def apply(prepared, x, ids, weights, diagnostic=False):
    load_splitquant_operator()
    up, scale, down, cp, cn, hp, hn, rotation, lut = prepared
    fn = (torch.ops.npu.splitquant_moe_fused_debug if diagnostic
          else torch.ops.npu.splitquant_moe_fused)
    return fn(x.contiguous(), up, scale, down, cp, cn, hp, hn, rotation,
              ids.contiguous(), weights.contiguous(), lut)


def pack_up_nz(packed):
    """Pack nibble weights into the MoE NZ layout.

    ``packed`` is stored as ``[expert, N, K/2]`` bytes, with two INT4
    values per byte.  The NZ kernel consumes ``[expert, N/64, K/16, 16, 32]``
    bytes.  Since the input's last dimension is ``K/2`` bytes, its NZ block
    count is ``(K/2)/8``.  The previous code used ``K/2`` directly, leaving
    the reshape eight times too large for Qwen3-30B.
    """
    e, n, khalf = packed.shape
    if n % 64 or khalf % 8:
        raise ValueError(
            f"NZ up weight shape must be divisible by (N=64, K/2=8), got {tuple(packed.shape)}"
        )
    raw = packed.to(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), -1).reshape(e, n, khalf * 2).transpose(1, 2).contiguous()
    paired = (codes[:, :, ::2] | (codes[:, :, 1::2] << 4)).contiguous()
    return paired.reshape(e, khalf // 8, 16, n // 64, 32).permute(0, 3, 1, 2, 4).contiguous()


def unpack_up_nz(packed):
    nblocks, kblocks = packed.shape[:2]
    n, k = nblocks * 64, kblocks * 16
    pairs = packed.permute(1, 2, 0, 3).contiguous().reshape(k, n // 2)
    codes = torch.stack((pairs & 15, pairs >> 4), -1).reshape(k, n).t().contiguous()
    return (codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous()
