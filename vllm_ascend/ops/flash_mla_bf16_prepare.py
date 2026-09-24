# SPDX-License-Identifier: Apache-2.0
"""Pack expanded MLA K/V projection views without quantization or Q copies."""

import torch


def prepare_flash_mla_bf16(key_nope, value, key_rope):
    """Return contiguous BF16 K192/V128; preserve every input BF16 bit."""
    return torch.cat((key_nope, key_rope.expand(-1, key_nope.shape[1], -1)), dim=-1), value.contiguous()
