# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

import torch

from vllm_ascend.ops.statetree.codec import PackedWeight, pack, unpack_reference


class TestStateTreeCodec(unittest.TestCase):
    def assert_bits_equal(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        self.assertTrue(torch.equal(actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16)))

    def test_preserves_every_bf16_pattern(self):
        for shuffle in (False, True):
            with self.subTest(shuffle=shuffle):
                weight = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
                if shuffle:
                    weight = weight[torch.randperm(weight.numel(), generator=torch.Generator().manual_seed(7))]
                weight = weight.reshape(256, 256)
                self.assert_bits_equal(unpack_reference(pack(weight, device="cpu")), weight)

    def test_padded_tile_and_payload_round_trip(self):
        weight = torch.randn(208, 256, generator=torch.Generator().manual_seed(65)).to(torch.bfloat16)
        packed = pack(weight, n_tile=192, k_tile=128, device="cpu")
        restored = PackedWeight(
            packed.payload.clone(), packed.n, packed.k, packed.n_tile, packed.k_tile, packed.escape_slots
        )
        restored.validate()
        self.assert_bits_equal(unpack_reference(restored), weight)

    def test_rejects_invalid_weights(self):
        for weight in (
            torch.ones(16, 64),
            torch.ones(16, 65, dtype=torch.bfloat16),
            torch.ones(0, 64, dtype=torch.bfloat16),
        ):
            with self.subTest(shape=weight.shape, dtype=weight.dtype), self.assertRaises(ValueError):
                pack(weight, device="cpu")

    def test_rejects_incomplete_tile_specification(self):
        weight = torch.ones(32, 64, dtype=torch.bfloat16)
        for kwargs in ({"n_tile": 32}, {"k_tile": 64}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "specified together"):
                pack(weight, device="cpu", **kwargs)

    def test_rejects_invalid_payload(self):
        packed = pack(torch.ones(32, 64, dtype=torch.bfloat16), device="cpu")
        for payload in (packed.payload[:, :-1], packed.payload.to(torch.int16)):
            with self.subTest(shape=payload.shape, dtype=payload.dtype), self.assertRaises(ValueError):
                PackedWeight(payload, packed.n, packed.k, packed.n_tile, packed.k_tile, packed.escape_slots).validate()
