# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression tests for cross-group aliasing, independent of NPU kernels."""

import unittest

from vllm_ascend.core.dflash_cache_layout import (
    DFLASH_FIA_MAX_KERNEL_BLOCKS,
    DFLASH_FIA_MAX_PLANE_ELEMENTS,
    get_dflash_aligned_block_size,
    get_dflash_fia_safe_num_blocks,
)


class TestDFlashCacheLayout(unittest.TestCase):
    def setUp(self):
        # Qwen3.6-27B TP2: BF16 conv, FP32 recurrent state, BF16 K/V.
        self.conv = 5120 * (4 - 1 + 7) * 2
        self.ssm = 24 * 128 * 128 * 4
        self.page = self.conv + 2 * self.ssm
        self.row = 4 * 128 * 2

    def _aligned_block_size(self, **overrides):
        args = dict(
            conv_page_bytes=self.conv,
            ssm_page_bytes=self.ssm,
            common_page_size_bytes=self.page,
            key_row_bytes=self.row,
            value_row_bytes=self.row,
        )
        args.update(overrides)
        return get_dflash_aligned_block_size(**args)

    def _safe_num_blocks(self, **overrides):
        args = dict(
            storage_block_size=1536,
            key_row_bytes=self.row,
            value_row_bytes=self.row,
            key_element_bytes=2,
            value_element_bytes=2,
        )
        args.update(overrides)
        return get_dflash_fia_safe_num_blocks(**args)

    def test_padded_small_swa_pages_overwrite_other_full_attention_block_ids(self):
        # Replay the existing contiguous-tail view equations, not an assumed
        # page-strided layout. All three physical IDs are different/non-null.
        count = 12
        cache = bytearray(count * self.page)
        small_plane = 128 * self.row
        full_v_10 = count * (self.conv + self.ssm) + 10 * self.ssm
        full_v_11 = count * (self.conv + self.ssm) + 11 * self.ssm
        swa_k_1 = count * (self.page - 2 * small_plane) + small_plane
        swa_v_1 = count * (self.page - small_plane) + small_plane
        self.assertEqual(swa_k_1, full_v_10 + small_plane)
        self.assertEqual(swa_v_1, full_v_11 + small_plane)

        cache[full_v_10 : full_v_10 + self.ssm] = bytes([11]) * self.ssm
        cache[full_v_11 : full_v_11 + self.ssm] = bytes([22]) * self.ssm
        self.assertEqual((cache[swa_k_1], cache[swa_v_1]), (11, 22))
        cache[swa_k_1 : swa_k_1 + small_plane] = bytes([33]) * small_plane
        cache[swa_v_1 : swa_v_1 + small_plane] = bytes([44]) * small_plane
        self.assertEqual((cache[full_v_10 + small_plane], cache[full_v_11 + small_plane]), (33, 44))

    def test_alignment_keeps_all_different_physical_block_ids_disjoint(self):
        aligned = self._aligned_block_size()
        self.assertEqual(aligned, 1536)
        plane = aligned * self.row
        for count in (2, 12, 33):
            for first in range(count):
                mamba_and_full = (
                    (first * self.conv, (first + 1) * self.conv),
                    (count * self.conv + first * self.ssm, count * self.conv + (first + 1) * self.ssm),
                    (
                        count * (self.conv + self.ssm) + first * plane,
                        count * (self.conv + self.ssm) + (first + 1) * plane,
                    ),
                )
                for second in range(count):
                    if first == second:
                        continue
                    swa = (
                        (
                            count * (self.page - 2 * plane) + second * plane,
                            count * (self.page - 2 * plane) + (second + 1) * plane,
                        ),
                        (
                            count * (self.page - plane) + second * plane,
                            count * (self.page - plane) + (second + 1) * plane,
                        ),
                    )
                    for start, end in mamba_and_full:
                        for other_start, other_end in swa:
                            self.assertTrue(end <= other_start or other_end <= start)

    def test_alignment_supports_other_head_dimensions_without_model_constants(self):
        self.assertEqual(
            self._aligned_block_size(
                conv_page_bytes=256,
                ssm_page_bytes=128 * 768,
                common_page_size_bytes=256 + 2 * 128 * 768,
                key_row_bytes=768,
                value_row_bytes=768,
            ),
            128,
        )

    def test_incompatible_contiguous_layouts_are_rejected(self):
        cases = (
            dict(conv_page_bytes=-1),
            dict(ssm_page_bytes=0),
            dict(common_page_size_bytes=self.page + 128),
            dict(key_row_bytes=0),
            dict(value_row_bytes=self.row * 2),
            dict(key_row_bytes=5, value_row_bytes=5),
            dict(kernel_block_size=1000),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self._aligned_block_size(**kwargs)

    def test_reproduced_geometry_stays_below_both_fia_boundaries(self):
        limit = self._safe_num_blocks()
        self.assertEqual(limit, 5461)
        self.assertLess(limit * 12 - 1, DFLASH_FIA_MAX_KERNEL_BLOCKS)
        self.assertLessEqual(limit * 1536 * self.row // 2, DFLASH_FIA_MAX_PLANE_ELEMENTS)
        self.assertGreater((limit + 1) * 12, DFLASH_FIA_MAX_KERNEL_BLOCKS)

    def test_more_heads_can_hit_element_boundary_before_block_id_boundary(self):
        self.assertEqual(self._safe_num_blocks(key_row_bytes=2048, value_row_bytes=2048), 2730)
        self.assertEqual(self._safe_num_blocks(value_row_bytes=4096), 1365)

    def test_element_size_and_storage_block_size_are_accounted_for(self):
        self.assertEqual(self._safe_num_blocks(key_element_bytes=1), 2730)
        self.assertEqual(self._safe_num_blocks(storage_block_size=128), 65536)
        self.assertEqual(self._safe_num_blocks(storage_block_size=256), 32768)
        self.assertEqual(self._safe_num_blocks(storage_block_size=1 << 33), 0)

    def test_invalid_fia_geometry_is_rejected(self):
        cases = (
            dict(storage_block_size=0),
            dict(storage_block_size=129),
            dict(kernel_block_size=256),
            dict(key_row_bytes=0),
            dict(value_element_bytes=0),
            dict(key_row_bytes=1025),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self._safe_num_blocks(**kwargs)


if __name__ == "__main__":
    unittest.main()
