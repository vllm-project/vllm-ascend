# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for SISO launch arguments; kernels are mocked."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from vllm_ascend.ops.triton import rope


class TestRopeSiso(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.kernel = stack.enter_context(patch.object(rope, "_triton_rope_siso"))
        stack.enter_context(patch.object(rope, "get_vectorcore_num", return_value=40))
        self.padding = stack.enter_context(
            patch.object(
                rope.triton, "next_power_of_2", side_effect=lambda n: 1 << (max(n, 1) - 1).bit_length(), create=True
            )
        )
        self.qk = torch.empty(4, 2, 128, device="cpu")

    def test_default_dimension_is_resolved_before_padding(self):
        for rotary_dim in (64, 96, 128):
            for shape in ((4, rotary_dim // 2), (4, 2, rotary_dim // 4)):
                with self.subTest(rotary_dim=rotary_dim, shape=shape):
                    cos = torch.empty(shape, device="cpu")
                    out = rope.rope_forward_triton_siso(self.qk, cos, torch.empty_like(cos))
                    args = self.kernel.__getitem__.return_value.call_args.args
                    self.assertIs(out, self.qk)
                    self.assertEqual(args[12], rotary_dim)
                    self.assertEqual(args[14], 1 << (rotary_dim - 1).bit_length())
        self.assertNotIn(-1, [call.args[0] for call in self.padding.call_args_list])

    def test_inferred_dimension_cannot_exceed_head_dimension(self):
        cos = torch.empty(4, 80, device="cpu")
        with self.assertRaises(AssertionError):
            rope.rope_forward_triton_siso(self.qk, cos, torch.empty_like(cos))
        self.kernel.__getitem__.assert_not_called()

    def test_explicit_dimension_is_preserved(self):
        cos = torch.empty(4, 64, device="cpu")
        rope.rope_forward_triton_siso(self.qk, cos, torch.empty_like(cos), rope_dim=64)
        args = self.kernel.__getitem__.return_value.call_args.args
        self.assertEqual((args[12], args[14]), (64, 64))

    def test_cache_path_keeps_precedence_and_explicit_dimension(self):
        cache = torch.empty(8, 64, device="cpu")
        positions = torch.arange(4, device="cpu")
        cos = torch.empty(4, 80, device="cpu")
        rope.rope_forward_triton_siso(self.qk, cos, cos, cos_sin_cache=cache, positions=positions, rope_dim=64)
        call = self.kernel.__getitem__.return_value.call_args
        self.assertEqual((call.args[12], call.args[14]), (64, 64))
        self.assertIs(call.args[6], cache)
        self.assertTrue(call.kwargs["USE_COS_SIN"])

    def test_explicit_oversized_dimension_is_rejected(self):
        cos = torch.empty(4, 80, device="cpu")
        with self.assertRaises(AssertionError):
            rope.rope_forward_triton_siso(self.qk, cos, cos, rope_dim=160)
        self.kernel.__getitem__.assert_not_called()
