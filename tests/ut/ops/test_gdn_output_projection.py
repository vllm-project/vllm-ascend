# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

from vllm_ascend.ops import gdn
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


class _Projection(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", values)

    def forward(self, hidden_states):
        return self.values, None


class _GatedNorm(nn.Module):
    def forward(self, x, z):
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + 1e-6)
        return (normalized * torch.nn.functional.silu(z.float())).to(x.dtype)


class _OutputProjection(nn.Module):
    def __init__(self, width, dtype):
        super().__init__()
        self.register_buffer("weight", torch.arange(width * 3, dtype=dtype).reshape(width, 3) / 10)

    def forward(self, x):
        return x @ self.weight, None


@pytest.mark.parametrize("receiver_cls", [QwenGatedDeltaNetAttention, AscendGatedDeltaNetAttention])
@pytest.mark.parametrize("layout", ["separate", "packed", "interleaved"])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("valid_tokens", [0, 3, 4])
@pytest.mark.parametrize("with_output", [False, True])
def test_forward_reuses_upstream_output_projection(
    monkeypatch, receiver_cls, layout, tp_size, dtype, valid_tokens, with_output
):
    # Keep the actual receiver class and method binding, without allocating
    # model weights or requiring a distributed process group in this CPU UT.
    layer = object.__new__(receiver_cls)
    nn.Module.__init__(layer)
    monkeypatch.setattr(receiver_cls, "forward", AscendGatedDeltaNetAttention.forward)
    monkeypatch.setattr(receiver_cls, "_split_ba_for_tp", AscendGatedDeltaNetAttention._split_ba_for_tp, raising=False)
    layer.tp_size = tp_size
    layer.num_k_heads = layer.num_v_heads = 2
    layer.head_k_dim = layer.head_v_dim = 2
    layer.key_dim = layer.value_dim = 4
    layer.prefix = "layers.0.linear_attn"
    layer.gqa_interleaved_layout = layout == "interleaved"
    tokens = 4
    local_width = layer.value_dim // tp_size
    packed = torch.arange(tokens * local_width * 4, dtype=dtype).reshape(tokens, -1) / 10
    qkv, z_flat = packed.split([local_width * 3, local_width], dim=-1)
    z = z_flat.reshape(tokens, -1, layer.head_v_dim)
    assert not z.is_contiguous()
    ba = torch.arange(tokens * (layer.num_v_heads // tp_size) * 2, dtype=dtype).reshape(tokens, -1)
    b, a = ba.chunk(2, dim=-1)
    split_ba = Mock(return_value=(b, a))
    monkeypatch.setattr(layer, "split_ba", split_ba, raising=False)
    layer.in_proj_ba = _Projection(ba)
    if layout == "separate":
        layer.in_proj_qkv = _Projection(qkv)
        layer.in_proj_z = _Projection(z_flat)
    else:
        layer.in_proj_qkvz = _Projection(packed)
    split_interleaved = Mock(return_value=(qkv, z, b, a))
    monkeypatch.setattr(gdn, "fused_qkvzba_split_reshape_cat", split_interleaved)
    layer.norm = _GatedNorm()
    layer.out_proj = _OutputProjection(local_width, dtype)
    expected_core = torch.zeros_like(z)
    expected_core[:valid_tokens] = qkv[:valid_tokens, :local_width].reshape(valid_tokens, local_width // 2, 2)

    def attention_core(mixed_qkv, actual_b, actual_a, output, prefix, use_aiter):
        assert prefix == layer.prefix
        assert use_aiter is False
        torch.testing.assert_close(mixed_qkv, qkv)
        torch.testing.assert_close(actual_b, b)
        torch.testing.assert_close(actual_a, a)
        assert torch.count_nonzero(output) == 0
        output.copy_(expected_core)

    monkeypatch.setattr(torch.ops.vllm, "qwen_gdn_attention_core", attention_core)
    original_projection = QwenGatedDeltaNetAttention._output_projection
    projection_calls = []

    def projection(self, core, gate):
        assert self is layer
        projection_calls.append((core.shape, gate.shape))
        return original_projection(self, core, gate)

    monkeypatch.setattr(QwenGatedDeltaNetAttention, "_output_projection", projection)
    # Reference is the former Ascend post-core path, independent of the
    # upstream helper under test. Device norm/linear kernels are not tested here.
    normalized = layer.norm(expected_core.reshape(-1, 2), z.reshape(-1, 2)).reshape(z.shape)
    expected, _ = layer.out_proj(normalized.reshape(tokens, local_width))
    output = torch.full((tokens + 2, 3), -99, dtype=dtype) if with_output else None
    result = layer(torch.zeros(tokens, 3, dtype=dtype), output=output)
    assert projection_calls == [(expected_core.shape, z.shape)]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if output is not None:
        torch.testing.assert_close(output[:tokens], expected, rtol=0, atol=0)
        assert torch.all(output[tokens:] == -99)
        assert result.data_ptr() != output.data_ptr()
    if layout == "interleaved":
        split_interleaved.assert_called_once()
        split_ba.assert_not_called()
    else:
        split_ba.assert_called_once_with(ba)
