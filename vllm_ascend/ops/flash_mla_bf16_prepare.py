# SPDX-License-Identifier: Apache-2.0
"""Pack expanded MLA K/V projection views without quantization or Q copies."""

import torch

from vllm_ascend.utils import is_950


def prepare_flash_mla_bf16(key_nope, value, key_rope):
    """Return contiguous BF16 K192/V128; preserve every input BF16 bit."""
    if (
        key_nope.device.type != "cpu"
        and is_950()
        and key_nope.dtype == torch.bfloat16
        and 0 < key_nope.shape[1] <= 128
        and key_rope.stride(1) >= 64
        and hasattr(torch.ops._C_ascend, "flash_mla_bf16_prepare")
    ):
        return torch.ops._C_ascend.flash_mla_bf16_prepare(key_nope, value, key_rope)
    return torch.cat((key_nope, key_rope.expand(-1, key_nope.shape[1], -1)), dim=-1), value.contiguous()
