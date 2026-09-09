# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.model_executor.models import deepencoder

from vllm_ascend.ops.rel_pos_attention import AscendRelPosAttention
from vllm_ascend.utils import vllm_version_is


@pytest.mark.parametrize("window_size", [0, 2])
def test_actual_deepencoder_block_constructs_ascend_attention(monkeypatch, window_size):
    # Run the exact upstream caller that started passing the new keyword.
    monkeypatch.setattr(deepencoder, "RelPosAttention", AscendRelPosAttention)
    block = deepencoder.Block(
        dim=16,
        num_heads=2,
        qkv_bias=False,
        use_rel_pos=True,
        rel_pos_zero_init=False,
        input_size=(4, 4),
        window_size=window_size,
    )
    assert isinstance(block.attn, AscendRelPosAttention)
    assert block.attn.qkv.bias is None
    assert block.attn.num_heads == 2
    assert block.attn.scale == 8**-0.5
    assert block.attn.use_rel_pos
    expected_side = window_size or 4
    assert block.attn.rel_pos_h.shape == (2 * expected_side - 1, 8)
    assert block.attn.rel_pos_w.shape == (2 * expected_side - 1, 8)
    torch.testing.assert_close(block.attn.rel_pos_h, torch.zeros_like(block.attn.rel_pos_h))
    assert block.attn.forward.__func__ is AscendRelPosAttention.forward
    if not vllm_version_is("0.28.0"):
        assert block.attn.use_triton_attention is (window_size == 0)


def test_rel_pos_attention_preserves_full_positional_constructor_order():
    if vllm_version_is("0.28.0"):
        attention = AscendRelPosAttention(16, 2, False, True, False, (3, 5))
    else:
        attention = AscendRelPosAttention(16, 2, False, True, True, False, (3, 5))
        assert attention.use_triton_attention is True
    assert attention.rel_pos_h.shape == (5, 8)
    assert attention.rel_pos_w.shape == (9, 8)
    assert attention.qkv.bias is None
