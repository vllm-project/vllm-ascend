# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from unittest.mock import patch

import torch

from vllm_ascend.ops.statetree import operator
from vllm_ascend.ops.statetree.codec import pack


class TestStateTreeOperator(unittest.TestCase):
    def test_dispatch_uses_total_rows(self):
        for shape in ((0, 64), (64, 64), (65, 64), (128, 64), (129, 64), (255, 64), (256, 64), (257, 64), (2, 128, 64)):
            with self.subTest(shape=shape):
                x = torch.ones(shape, dtype=torch.bfloat16)
                weight = torch.ones(32, 64, dtype=torch.bfloat16)
                packed = pack(weight, device="cpu")
                with (
                    patch.object(operator.native, "linear", return_value=torch.empty((*shape[:-1], 32))) as fused,
                    patch.object(operator.native, "decode", return_value=weight) as decode,
                ):
                    output = operator._statetree_linear(x, packed.payload, 32, 64, 32, 64, packed.escape_slots)
                self.assertEqual(output.shape, (*shape[:-1], 32))
                if x.numel() // 64 < 256:
                    fused.assert_called_once()
                    decode.assert_not_called()
                else:
                    fused.assert_not_called()
                    decode.assert_called_once()
                    self.assertTrue(torch.equal(output, torch.nn.functional.linear(x, weight)))

    def test_fake_implementation_preserves_metadata(self):
        x = torch.empty((2, 65, 64), dtype=torch.bfloat16, device="meta")
        payload = torch.empty(0, dtype=torch.uint8, device="meta")
        output = operator._statetree_linear_fake(x, payload, 32, 64, 32, 64, 8)
        self.assertEqual(output.shape, (2, 65, 32))
        self.assertEqual(output.dtype, x.dtype)
        self.assertEqual(output.device, x.device)

    def test_load_time_reconstruction_failure_stops_preparation(self):
        weight = torch.ones(32, 64, dtype=torch.bfloat16)
        packed = pack(weight, device="cpu")
        with (
            patch.object(operator, "pack", return_value=packed),
            patch.object(operator.native, "decode", return_value=torch.zeros_like(weight)),
            patch.object(operator.native, "prepare") as prepare,
        ):
            with self.assertRaisesRegex(RuntimeError, "not bit-exact"):
                operator.pack_verify_and_prepare(weight)
            prepare.assert_not_called()
