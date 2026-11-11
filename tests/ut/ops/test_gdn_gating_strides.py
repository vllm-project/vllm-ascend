# SPDX-License-Identifier: Apache-2.0

from unittest import TestCase
from unittest.mock import patch

import torch

from vllm_ascend.ops.triton import fused_gdn_gating as gating


class TestGdnGatingStrides(TestCase):
    def test_passes_independent_input_strides_without_copying(self):
        for rows in (1, 7, 129):
            for layout in ("split", "transposed", "sliced", "expanded"):
                with self.subTest(rows=rows, layout=layout):
                    heads = 8
                    a, b = torch.randn(rows, 2 * heads).chunk(2, dim=-1)
                    if layout == "transposed":
                        b = torch.randn(heads, rows).t()
                    elif layout == "sliced":
                        a = torch.randn(rows * 2, heads * 3)[::2, ::3]
                    elif layout == "expanded":
                        a = torch.randn(1, heads).expand(rows, -1)
                    with (
                        patch.object(gating, "get_vectorcore_num", return_value=8),
                        patch.object(gating, "fused_gdn_gating_kernel") as kernel,
                    ):
                        g, beta = gating.fused_gdn_gating_patch(torch.zeros(heads), a, b, torch.zeros(heads))
                    args = kernel.__getitem__.return_value.call_args.args
                    self.assertIs(args[3], a)
                    self.assertIs(args[4], b)
                    self.assertEqual(args[6:10], (*a.stride(), *b.stride()))
                    self.assertEqual(g.shape, (1, rows, heads))
                    self.assertTrue(g.is_contiguous())
                    self.assertTrue(beta.is_contiguous())

    def test_selects_layout_categories_including_singleton_dimensions(self):
        heads = 8
        split_a, split_b = torch.randn(7, 2 * heads).chunk(2, dim=-1)
        single_a, single_b = torch.randn(1, 2 * heads).chunk(2, dim=-1)
        singleton_head = torch.randn(1, 7).t()
        cases = [
            ("dense", split_a.contiguous(), split_b.contiguous(), True, True),
            ("single_row", single_a, single_b, True, True),
            ("split", split_a, split_b, False, True),
            ("mixed", split_a.contiguous(), split_b, False, True),
            ("transposed", split_a, torch.randn(heads, 7).t(), False, False),
            ("sliced", torch.randn(14, heads * 3)[::2, ::3], split_b, False, False),
            ("expanded", torch.randn(1, heads).expand(7, -1), split_b, False, True),
            ("singleton_head", singleton_head, singleton_head, True, False),
        ]
        for name, a, b, contiguous, unit_head_strides in cases:
            with self.subTest(layout=name):
                with (
                    patch.object(gating, "get_vectorcore_num", return_value=8),
                    patch.object(gating, "fused_gdn_gating_kernel") as kernel,
                ):
                    gating.fused_gdn_gating_patch(torch.zeros(a.shape[1]), a, b, torch.zeros(a.shape[1]))
                call = kernel.__getitem__.return_value.call_args
                self.assertIs(call.args[3], a)
                self.assertIs(call.args[4], b)
                self.assertEqual(call.kwargs["CONTIGUOUS_INPUTS"], contiguous)
                self.assertEqual(call.kwargs["UNIT_HEAD_STRIDES"], unit_head_strides)
