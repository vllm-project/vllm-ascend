from unittest.mock import MagicMock

import torch

from vllm_ascend.patch.worker.patch_qwen2_5_vl import (
    qwen2_5_vision_attention_forward,
)


def test_fused_qkv_rope_pad_short_circuits_generic_path():
    token_count = 7
    hidden_size = 32
    projected_size = 3 * 8 * 72
    projected = torch.randn(token_count, 1, projected_size)
    fused_context = torch.randn(1, token_count, 8, 72)
    expected = torch.randn(token_count, 1, hidden_size)

    layer = MagicMock()
    layer.qkv.return_value = (projected, None)
    layer.attn.forward_qkv_rope_pad_fia.return_value = fused_context
    layer.proj.return_value = (expected, None)

    output = qwen2_5_vision_attention_forward(
        layer,
        torch.randn(token_count, 1, hidden_size),
        torch.tensor([0, token_count], dtype=torch.int32),
        torch.randn(token_count, 36),
        torch.randn(token_count, 36),
        torch.tensor(token_count),
        None,
    )

    torch.testing.assert_close(output, expected)
    layer.apply_rotary_emb.assert_not_called()
    layer.attn.assert_not_called()
