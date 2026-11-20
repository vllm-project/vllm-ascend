from threading import Lock
from unittest.mock import MagicMock

import torch
from torch import nn
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionAttention, Qwen2_5_VisionBlock

from vllm_ascend.patch.worker.patch_qwen2_5_vl import (
    qwen2_5_vision_attention_forward,
)


class _VisionAttentionBackend(nn.Module):
    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.sequence_lock = Lock()

    def forward(self, *, value: torch.Tensor, **kwargs) -> torch.Tensor:
        return value

    def forward_qkv_rope_pad_fia(
        self,
        qkv: torch.Tensor,
        _cos: torch.Tensor,
        _sin: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # Exercise the same Python-only metadata work as the eager adapter.
        with self.sequence_lock:
            token_count = cu_seqlens.tolist()[-1]
        value = qkv.chunk(3, dim=-1)[2].unflatten(-1, (self.num_heads, self.head_size))
        return value.transpose(0, 1) + token_count


class _ResidualNorm(nn.Module):
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + residual
        return x, x


def test_compiled_vision_block_uses_generic_attention_with_fusion_available():
    num_heads, head_size = 8, 72
    token_count = 7

    # Use the real block and patched attention forwards with CPU stand-ins.
    attention = Qwen2_5_VisionAttention.__new__(Qwen2_5_VisionAttention)
    nn.Module.__init__(attention)
    attention.num_attention_heads_per_partition = num_heads
    attention.hidden_size_per_attention_head = head_size
    attention.qkv = lambda x: (x.repeat(1, 1, 3), None)
    attention.proj = lambda x: (x, None)
    attention.apply_rotary_emb = lambda qk, cos, sin: qk
    attention.attn = _VisionAttentionBackend(num_heads, head_size)

    block = Qwen2_5_VisionBlock.__new__(Qwen2_5_VisionBlock)
    nn.Module.__init__(block)
    block.attn = attention
    block.norm1 = nn.Identity()
    block.norm2 = _ResidualNorm()
    block.mlp = nn.Identity()

    x = torch.randint(0, 4, (token_count, 1, num_heads * head_size)).to(torch.bfloat16)
    args = (
        x,
        torch.tensor([0, 2, token_count], dtype=torch.int32),
        torch.ones(token_count, head_size // 2, dtype=x.dtype),
        torch.zeros(token_count, head_size // 2, dtype=x.dtype),
        torch.tensor(token_count - 2, dtype=torch.int32),
    )

    # The identity MLP doubles the sum of the input and the attention output.
    torch.testing.assert_close(block.forward(*args), 4 * x + 2 * token_count)
    compiled_forward = torch.compile(block.forward, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled_forward(*args), 4 * x)


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


def test_fused_qkv_rope_pad_relays_multi_sequence_cu_seqlens():
    token_count = 400
    hidden_size = 32
    projected_size = 3 * 8 * 72
    projected = torch.randn(token_count, 1, projected_size)
    fused_context = torch.randn(1, token_count, 8, 72)
    expected = torch.randn(token_count, 1, hidden_size)
    cu_seqlens = torch.tensor([0, 100, 240, 400], dtype=torch.int32)

    layer = MagicMock()
    layer.qkv.return_value = (projected, None)
    layer.attn.forward_qkv_rope_pad_fia.return_value = fused_context
    layer.proj.return_value = (expected, None)

    output = qwen2_5_vision_attention_forward(
        layer,
        torch.randn(token_count, 1, hidden_size),
        cu_seqlens,
        torch.randn(token_count, 36),
        torch.randn(token_count, 36),
        torch.tensor(token_count),
        None,
    )

    torch.testing.assert_close(output, expected)
    layer.attn.forward_qkv_rope_pad_fia.assert_called_once()
    assert layer.attn.forward_qkv_rope_pad_fia.call_args.kwargs["cu_seqlens"] is cu_seqlens
