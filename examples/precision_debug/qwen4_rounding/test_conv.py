# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import pytest
import torch
from conv import conv_product_bf16


def test_prefill_and_decode_have_identical_history_semantics():
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(9, 13, generator=generator).bfloat16()
    weight = torch.randn(4, 13, generator=generator).bfloat16()
    before = x.clone()
    prefill = conv_product_bf16(x, weight)
    history = torch.zeros(3, 13, dtype=x.dtype)
    decoded = []
    for token in x.split(1):
        decoded.append(conv_product_bf16(token, weight, history))
        history = torch.cat((history[1:], token))
    assert torch.equal(prefill, torch.cat(decoded))
    assert torch.equal(x, before)
    assert not torch.equal(prefill, conv_product_bf16(x, weight, round_products=False))


def test_empty_and_invalid_inputs():
    assert conv_product_bf16(torch.empty(0, 13, dtype=torch.bfloat16), torch.ones(4, 13).bfloat16()).shape == (0, 13)
    with pytest.raises(ValueError):
        conv_product_bf16(torch.ones(1, 13), torch.ones(4, 13))
