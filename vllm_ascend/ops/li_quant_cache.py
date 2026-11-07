# SPDX-License-Identifier: Apache-2.0
"""N6 LI quantization/cache fusion metadata contract; no device-to-host reads."""

import torch


def supports_li_quant_cache(x, slots, key_cache, scale_cache) -> bool:
    tensors = (x, slots, key_cache, scale_cache)
    return (
        all(t.device.type == "npu" and t.device == x.device and t.is_contiguous() for t in tensors)
        and x.shape == (6, 128)
        and x.dtype in (torch.bfloat16, torch.float16)
        and slots.shape == (6,)
        and slots.dtype in (torch.int32, torch.int64)
        and key_cache.ndim >= 2
        and key_cache.numel() > 0
        and key_cache.shape[-1] == 128
        and key_cache.dtype == torch.int8
        and scale_cache.numel() == key_cache.numel() // 128
        and scale_cache.dtype == torch.float16
    )


def try_li_quant_cache(x, slots, key_cache, scale_cache):
    """Return None, without launch/writes, when metadata or build is unsupported.

    Valid slots are unique, as required by the decode slot mapping.
    The caller must only defer quantization where these outputs do not need
    communication before their cache write.
    """
    if not supports_li_quant_cache(x, slots, key_cache, scale_cache):
        return None
    if not hasattr(torch.ops._C_ascend, "li_quant_cache"):
        return None
    return torch.ops._C_ascend.li_quant_cache(x, slots, key_cache.view(-1, 128), scale_cache.view(-1, 1))
