# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


def _fixture(tp_size, tokens, *, replicated=False, column_stride=False):
    heads, dim = 32 // tp_size, 128
    qkv_width = 8192 // tp_size
    packed_width = qkv_width + heads * dim
    packed = torch.arange(tokens * packed_width, dtype=torch.float32).reshape(tokens, packed_width)
    ba_width = 64 if replicated else 2 * heads
    ba = torch.arange(tokens * ba_width, dtype=torch.float32).reshape(tokens, ba_width)
    if column_stride:
        storage = torch.empty(tokens, packed_width * 2)
        storage[:, ::2] = packed
        packed = storage[:, ::2]
    rank = 1 if tp_size > 1 else 0

    def split_ba(value):
        b, a = value.chunk(2, dim=-1)
        if replicated:
            b = b[:, rank * heads : (rank + 1) * heads]
            a = a[:, rank * heads : (rank + 1) * heads]
        return b, a

    layer = SimpleNamespace(
        key_dim=2048,
        value_dim=4096,
        num_v_heads=32,
        head_v_dim=128,
        tp_size=tp_size,
        gqa_interleaved_layout=False,
        in_proj_qkvz=lambda _: (packed, None),
        in_proj_ba=lambda _: (ba, None),
        _split_ba_for_tp=split_ba,
        norm=lambda core, z: core + z,
        out_proj=lambda value: (value, None),
        prefix="test.gdn",
    )
    return layer, packed, ba, qkv_width


def _cpu_prepare(packed, ba, qkv_width, heads, dim):
    # Independent CPU oracle for the device-only data movement primitive.
    return (
        packed[:, :qkv_width].clone(memory_format=torch.contiguous_format),
        packed[:, qkv_width:].reshape(packed.shape[0], heads, dim),
        ba[:, :heads].clone(memory_format=torch.contiguous_format),
        ba[:, heads:].clone(memory_format=torch.contiguous_format),
        torch.zeros((packed.shape[0], heads, dim), dtype=packed.dtype),
    )


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("tokens", [0, 1, 4])
def test_packed_forward_fuses_preparation_and_preserves_padding(tp_size, tokens):
    # Catches bypassing fusion, duplicate zero-fill, swapped B/A and padding drift.
    layer, packed, ba, width = _fixture(tp_size, tokens)
    heads = 32 // tp_size
    live_tokens = max(0, tokens - 1)
    prepared = {}

    def prepare_inputs(*args):
        values = _cpu_prepare(*args)
        prepared["core"] = values[-1]
        return values

    def core(qkv, b, a, output, prefix, is_prefill):
        assert output is prepared["core"]
        assert qkv.is_contiguous() and b.is_contiguous() and a.is_contiguous()
        torch.testing.assert_close(qkv, packed[:, :width], rtol=0, atol=0)
        torch.testing.assert_close(b, ba[:, :heads], rtol=0, atol=0)
        torch.testing.assert_close(a, ba[:, heads:], rtol=0, atol=0)
        assert torch.count_nonzero(output) == 0
        output[:live_tokens] = 3

    destination = torch.full((tokens + 1, heads * 128), -17.0)
    with (
        patch("vllm_ascend.ops.gdn.fused_gdn_prepare", side_effect=prepare_inputs, create=True) as prepare,
        patch("torch.ops.vllm.qwen_gdn_attention_core", side_effect=core),
    ):
        result = AscendGatedDeltaNetAttention.forward(layer, torch.empty(tokens, 2048), destination)
    prepare.assert_called_once()
    expected = packed[:, width:].clone()
    expected[:live_tokens] += 3
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(destination[:tokens], expected, rtol=0, atol=0)
    assert torch.all(destination[tokens] == -17)


@pytest.mark.parametrize("kind", ["replicated", "column_stride", "ba_column_stride", "dtype_mismatch"])
def test_unsupported_packed_layout_preserves_existing_split(kind):
    layer, packed, ba, width = _fixture(4, 3, replicated=kind == "replicated", column_stride=kind == "column_stride")
    if kind == "ba_column_stride":
        storage = torch.empty(ba.shape[0], ba.shape[1] * 2)
        storage[:, ::2] = ba
        ba = storage[:, ::2]
        layer.in_proj_ba = lambda _: (ba, None)
    if kind == "dtype_mismatch":
        packed = packed.to(torch.float64)
        layer.in_proj_qkvz = lambda _: (packed, None)
    expected_b, expected_a = layer._split_ba_for_tp(ba)

    def core(qkv, b, a, output, *_):
        torch.testing.assert_close(qkv, packed[:, :width], rtol=0, atol=0)
        torch.testing.assert_close(b, expected_b, rtol=0, atol=0)
        torch.testing.assert_close(a, expected_a, rtol=0, atol=0)
        assert torch.count_nonzero(output) == 0
        output[:] = 2

    with (
        patch(
            "vllm_ascend.ops.gdn.fused_gdn_prepare", side_effect=AssertionError("unsupported layout fused"), create=True
        ),
        patch("torch.ops.vllm.qwen_gdn_attention_core", side_effect=core),
    ):
        result = AscendGatedDeltaNetAttention.forward(layer, torch.empty(3, 2048))
    torch.testing.assert_close(result, packed[:, width:] + 2, rtol=0, atol=0)
