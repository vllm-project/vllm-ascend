from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.patch.worker import patch_qwen3vl


def test_qwen3_vision_metadata_expands_rope_once(monkeypatch):
    cos = torch.randn(5, 36)
    sin = torch.randn_like(cos)
    original = MagicMock(
        return_value={
            "rotary_pos_emb_cos": cos,
            "rotary_pos_emb_sin": sin,
        }
    )
    monkeypatch.setattr(patch_qwen3vl, "_QWEN3_VISION_PREPARE_METADATA_ORIGINAL", original)

    metadata = patch_qwen3vl._qwen3_vision_prepare_metadata_with_expanded_rope(
        SimpleNamespace(), [[1, 2, 2]]
    )

    original.assert_called_once()
    torch.testing.assert_close(metadata["rotary_pos_emb_cos"], torch.cat((cos, cos), dim=-1))
    torch.testing.assert_close(metadata["rotary_pos_emb_sin"], torch.cat((sin, sin), dim=-1))


def test_qwen_vision_attention_uses_preexpanded_rope(monkeypatch):
    seq_len, batch_size, num_heads, head_dim = 5, 1, 16, 72
    qkv = torch.randn(seq_len, batch_size, 3 * num_heads * head_dim, dtype=torch.bfloat16)
    rotary_mul = MagicMock(side_effect=lambda qk, cos, sin, rotary_mode: qk)
    monkeypatch.setattr(patch_qwen3vl.torch_npu, "npu_rotary_mul", rotary_mul)

    attention = MagicMock(return_value=torch.zeros(batch_size, seq_len, num_heads, head_dim))
    projection = MagicMock(side_effect=lambda x: (x, None))
    layer = SimpleNamespace(
        hidden_size_per_attention_head=head_dim,
        num_attention_heads_per_partition=num_heads,
        qkv=MagicMock(return_value=(qkv, None)),
        attn=attention,
        proj=projection,
    )
    cos = torch.randn(seq_len, head_dim)
    sin = torch.randn_like(cos)

    output = patch_qwen3vl._qwen_vision_attention_forward_with_preexpanded_rope(
        layer,
        torch.empty(0),
        torch.tensor([0, seq_len], dtype=torch.int32),
        cos,
        sin,
        torch.tensor(seq_len),
        None,
    )

    rotary_mul.assert_called_once()
    assert rotary_mul.call_args.kwargs["rotary_mode"] == "half"
    assert rotary_mul.call_args.args[1].shape == (1, seq_len, 1, head_dim)
    assert attention.call_args.kwargs["query"].shape == (batch_size, seq_len, num_heads, head_dim)
    assert output.shape == (seq_len, batch_size, num_heads * head_dim)


def test_qwen_vision_attention_keeps_upstream_fallback(monkeypatch):
    fallback = MagicMock(return_value=torch.tensor([1.0]))
    monkeypatch.setattr(patch_qwen3vl, "_QWEN_VISION_ATTENTION_ORIGINAL_FORWARD", fallback)
    args = (
        SimpleNamespace(hidden_size_per_attention_head=72),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0, 36),
        torch.empty(0, 36),
        torch.empty(0),
        None,
    )

    output = patch_qwen3vl._qwen_vision_attention_forward_with_preexpanded_rope(*args)

    fallback.assert_called_once_with(*args)
    torch.testing.assert_close(output, torch.tensor([1.0]))
