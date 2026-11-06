# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the INT2 KV-cache quantization mixin (module level).

Pins the packed-INT2 codec contract: 4 values in [-2, 1] per uint8 byte,
byte-exact round trip, last-dim divisibility fail-loud, and the per-shard
scale preparation from the checkpoint scale tensors. The full
quantize->pack->dequantize numeric path against the OSCAR reference is
covered by tests/ut/oscar/test_int2_quant.py (encoder side).
"""

import pytest
import torch

from vllm_ascend.attention.int2.quantcache import Int2QuantCacheMixin


class TestPackUnpackCodec:
    def test_per_byte_identity(self):
        # one byte == exactly one codeword: byte = (v0+2)|((v1+2)<<2)|
        # ((v2+2)<<4)|((v3+2)<<6); unpack of one byte is its exact inverse
        for _ in range(20):
            x = torch.randint(-2, 2, (7, 4), dtype=torch.int8)
            packed = Int2QuantCacheMixin._pack_int2_to_uint8(x)
            assert packed.shape == (7, 1) and packed.dtype == torch.uint8
            back = Int2QuantCacheMixin._unpack_uint8_to_int2(packed)
            assert back.dtype == torch.int8
            assert torch.equal(back, x)

    def test_known_byte_pattern(self):
        # (v0+2) | ((v1+2)<<2) | ((v2+2)<<4) | ((v3+2)<<6)
        x = torch.tensor([[[-2, 1, 0, -1]]], dtype=torch.int8)
        packed = Int2QuantCacheMixin._pack_int2_to_uint8(x)
        expect = (0) | (3 << 2) | (2 << 4) | (1 << 6)
        assert packed.item() == expect

    def test_multibyte_layout_is_byte_interleaved(self):
        # multi-byte unpack returns values grouped BY LANE (all v0s, then all
        # v1s, ...) — the byte-order convention the dequantize consumer
        # relies on; pin it so any silent layout change fails loud
        x = torch.randint(-2, 2, (3, 5, 12), dtype=torch.int8)
        packed = Int2QuantCacheMixin._pack_int2_to_uint8(x)
        assert packed.shape == (3, 5, 3)
        back = Int2QuantCacheMixin._unpack_uint8_to_int2(packed)
        expect = x.view(3, 5, 3, 4).transpose(-2, -1).reshape(3, 5, 12)
        assert torch.equal(back, expect)

    def test_every_codepoint_round_trips_per_byte(self):
        # all 4^4 = 256 byte patterns are valid codewords
        p = torch.arange(256, dtype=torch.uint8).reshape(256, 1)
        v = Int2QuantCacheMixin._unpack_uint8_to_int2(p)
        assert v.shape == (256, 4)
        assert Int2QuantCacheMixin._pack_int2_to_uint8(v).flatten().tolist() == list(range(256))
        assert int(v.min()) == -2 and int(v.max()) == 1

    def test_head_size_not_divisible_fails_loud(self):
        x = torch.zeros(1, 6, dtype=torch.int8)
        with pytest.raises(AssertionError):
            Int2QuantCacheMixin._pack_int2_to_uint8(x)


class TestScalePreparation:
    class _Self(Int2QuantCacheMixin):
        def __init__(self, nh=4, hd=64):
            self.num_kv_heads = nh
            self.head_size = hd

    def _layer(self, hd=64):
        # exact-shard layout: numel == num_kv_heads*head_size (the TP-slice
        # arm needs an initialized TP group and is covered by device legs)
        return type(
            "Layer",
            (),
            {
                "k_cache_scale": type("S", (), {"data": torch.randn(4, hd)})(),
                "v_cache_scale": type("S", (), {"data": torch.randn(4, hd)})(),
            },
        )()

    def test_scale_shape_and_inv_scale(self):
        s = self._Self(nh=4, hd=64)
        layer = self._layer()
        s._prepare_int2_scales(layer, "cpu")
        assert layer._int2_k_scale.shape == (1, 4, 64)
        assert layer._int2_v_scale.shape == (1, 4, 64)
        assert torch.allclose(layer._int2_k_inv_scale, 1.0 / layer._int2_k_scale)
        assert layer._int2_scales_prepared is True

    def test_idempotent_and_default_scales(self):
        s = self._Self(nh=4, hd=64)
        layer = self._layer()
        s._prepare_int2_scales(layer, "cpu")
        k_ref = layer._int2_k_scale
        s._prepare_int2_scales(layer, "cpu")  # prepared flag short-circuits
        assert layer._int2_k_scale is k_ref
        bare = type("Layer", (), {})()  # no scales -> defaults all-ones
        s._prepare_int2_scales(bare, "cpu")
        assert bare._int2_k_scale.shape == (1, 4, 64)
        assert float(bare._int2_k_scale.abs().min()) == 1.0
