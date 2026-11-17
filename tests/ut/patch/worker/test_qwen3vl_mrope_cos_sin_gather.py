# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""M-RoPE cos/sin lookup in the patched Qwen3 attention forward.

The fused split-qkv/rmsnorm/mrope kernel must receive exactly
``cos_sin_cache[positions]``: same values, shape, dtype and device.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.patch.worker import patch_qwen3vl

NUM_TOKENS = 7
MAX_POSITION = 64
ROTARY_DIM = 16
NUM_HEADS = 2
NUM_KV_HEADS = 1
HEAD_DIM = 16
QKV_SIZE = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM


def _make_attention(qkv_dtype: torch.dtype):
    rotary_emb = MagicMock(spec=patch_qwen3vl.AscendMRotaryEmbedding)
    rotary_emb.cos_sin_cache = torch.randn(MAX_POSITION, ROTARY_DIM, dtype=torch.float32)
    rotary_emb.mrope_section = [2, 3, 3]
    rotary_emb.mrope_interleaved = True
    rotary_emb.rotary_dim = ROTARY_DIM

    qkv = torch.randn(NUM_TOKENS, QKV_SIZE, dtype=qkv_dtype)
    norm = SimpleNamespace(weight=torch.ones(HEAD_DIM, dtype=qkv_dtype), variance_epsilon=1e-6)
    attention = SimpleNamespace(
        qkv_proj=MagicMock(return_value=(qkv, None)),
        rotary_emb=rotary_emb,
        q_norm=norm,
        k_norm=norm,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        attn=MagicMock(side_effect=lambda q, k, v: q),
        o_proj=MagicMock(side_effect=lambda x: (x, None)),
    )
    return attention, qkv


@pytest.mark.parametrize(
    "positions_shape",
    [(3, NUM_TOKENS), (NUM_TOKENS,)],
    ids=["mrope_3d_positions", "text_1d_positions"],
)
@pytest.mark.parametrize("qkv_dtype", [torch.float32, torch.bfloat16])
def test_kernel_receives_cos_sin_cache_rows_for_positions(monkeypatch, positions_shape, qkv_dtype):
    attention, qkv = _make_attention(qkv_dtype)
    positions = torch.randint(0, MAX_POSITION, positions_shape)
    captured = {}

    def fake_split_qkv_rmsnorm_mrope(**kwargs):
        captured.update(kwargs)
        q = torch.zeros(NUM_TOKENS, NUM_HEADS * HEAD_DIM, dtype=qkv_dtype)
        kv = torch.zeros(NUM_TOKENS, NUM_KV_HEADS * HEAD_DIM, dtype=qkv_dtype)
        return q, kv, kv, None

    monkeypatch.setattr(torch.ops.vllm, "triton_split_qkv_rmsnorm_mrope", fake_split_qkv_rmsnorm_mrope, raising=False)

    patch_qwen3vl.forward_with_split_qkv_rmsnorm_mrope(attention, positions, torch.empty(NUM_TOKENS, 1))

    expected = attention.rotary_emb.cos_sin_cache[positions].to(qkv_dtype)
    cos_sin = captured["cos_sin"]
    assert cos_sin.shape == (*positions_shape, ROTARY_DIM)
    assert cos_sin.dtype == qkv.dtype
    assert cos_sin.device == qkv.device
    torch.testing.assert_close(cos_sin, expected, rtol=0, atol=0)
