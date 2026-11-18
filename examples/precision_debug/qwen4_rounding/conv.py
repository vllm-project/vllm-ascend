# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Eager diagnostic for GDN convolution product rounding; not a cache updater."""

import torch


def conv_product_bf16(x, weight, history=None, *, round_products=True):
    """Replay one sequence's four-tap, bias-free SiLU convolution.

    x is [tokens, channels], weight is [4, channels], and history contains
    the three preceding tokens in chronological order. Inputs are not mutated.
    An integration must retain the native kernel's cache update and output
    layout. This function deliberately does not implement speculative decode.
    """
    if x.ndim != 2 or weight.shape != (4, x.shape[1]):
        raise ValueError("Expected x [T, C] and weight [4, C]")
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("x and weight must be BF16 on the same device")
    if history is None:
        history = x.new_zeros((3, x.shape[1]))
    if history.shape != (3, x.shape[1]) or history.dtype != x.dtype or history.device != x.device:
        raise ValueError("history must be BF16 [3, C] on the input device")
    padded = torch.cat((history, x), dim=0)
    acc = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for tap in range(4):
        product = padded[tap : tap + x.shape[0]].float() * weight[tap].float()
        if round_products:
            product = product.to(torch.bfloat16)
        acc = acc + product.float()
    return (acc * torch.sigmoid(acc)).to(x.dtype)
