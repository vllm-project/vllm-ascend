# SPDX-License-Identifier: Apache-2.0
"""Precompiled AscendC operand preparation for expanded C8 MLA prefill."""

import torch


def _prepare(query, key_nope, value, key_rope, *, fake_quant):
    partial = torch.ops._C_ascend.flash_attn_c8_quant_stats(key_nope, value)
    return torch.ops._C_ascend.flash_attn_c8_prepare(query, key_nope, value, key_rope, partial, fake_quant)


def quantize_flash_attn_c8(query, key_nope, value, key_rope):
    """Return Q8,K8,V8, compensated QR, expanded KR, SQ[Tq,H], SK[H], SV[H].

    BF16 inputs may be projection views with noncontiguous token/head axes.
    K/V scales reduce over all tokens and 128 channels independently for each
    head. Q scales reduce over each 128-channel row. RoPE is already rotated;
    this operator divides query RoPE by SQ then SK before BF16 rounding.
    """
    return _prepare(query, key_nope, value, key_rope, fake_quant=False)


def fake_quant_flash_attn_c8(query, key_nope, value, key_rope):
    """Return BF16 Q192/K192/V128 reconstructed from the same FP8 codes.

    The native fake branch materializes FP8 bytes in UB before reconstruction;
    the round trip cannot become an identity cast. Both statistics and native
    preparation belong in fake-quant timing. BF16 reconstruction and native
    attention's FP8 probability rounding remain additional error sources.
    """
    return _prepare(query, key_nope, value, key_rope, fake_quant=True)[:3]
