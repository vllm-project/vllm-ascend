# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from vllm_ascend.quantization.utils import get_mxfp_tp_padding, pair_mxfp_scales


class TestMXFPUtils(unittest.TestCase):
    def test_tp_padding_covers_checkpoint_groups(self):
        cases = [
            (4352, 32, 3, (0, 0)),
            (4304, 32, 0, (0, 16)),
            (528, 32, 3, (16, 0)),
            (1076, 32, 0, (0, 12)),
            (1076, 32, 1, (20, 24)),
            (1076, 32, 2, (8, 4)),
            (1076, 32, 3, (28, 16)),
            (1076, 64, 3, (28, 48)),
        ]
        for width, group_size, rank, expected in cases:
            with self.subTest(width=width, group_size=group_size, rank=rank):
                self.assertEqual(get_mxfp_tp_padding(width, group_size, rank), expected)

    def test_scale_pairs_preserve_bytes_and_zero_only_the_tail(self):
        for leading_shape in [(3,), (2, 3)]:
            for groups in (0, 1, 2, 17, 135, 136):
                with self.subTest(leading_shape=leading_shape, groups=groups):
                    scale = torch.randint(0, 256, (*leading_shape, groups), dtype=torch.uint8)
                    original = scale.clone()
                    paired = pair_mxfp_scales(scale)
                    self.assertEqual(paired.shape, (*leading_shape, (groups + 1) // 2, 2))
                    torch.testing.assert_close(paired[..., 0], scale[..., ::2])
                    torch.testing.assert_close(paired[..., : groups // 2, 1], scale[..., 1::2])
                    if groups % 2:
                        torch.testing.assert_close(paired[..., -1, 1], torch.zeros_like(scale[..., -1]))
                    torch.testing.assert_close(scale, original)

    def test_even_contiguous_scales_share_storage(self):
        scale = torch.arange(24, dtype=torch.uint8).reshape(3, 8)
        paired = pair_mxfp_scales(scale)
        self.assertEqual(paired.data_ptr(), scale.data_ptr())
        self.assertEqual(paired.stride(), (8, 2, 1))

    def test_noncontiguous_scales_preserve_group_order(self):
        for groups in (17, 18):
            with self.subTest(groups=groups):
                scale = torch.arange(groups * 3, dtype=torch.uint8).reshape(groups, 3).T
                self.assertFalse(scale.is_contiguous())
                paired = pair_mxfp_scales(scale)
                unpacked = paired.flatten(-2)
                torch.testing.assert_close(unpacked[:, :groups], scale)
