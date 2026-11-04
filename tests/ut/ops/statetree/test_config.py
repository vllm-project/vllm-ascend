# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from vllm_ascend.ops.statetree.config import choose_tiles, select_batch_tile, validate_tiles


class TestStateTreeConfig(unittest.TestCase):
    def test_weight_tiles(self):
        for shape, tiles in (((4096, 4096), (256, 128)), ((11008, 4096), (256, 128)), ((96, 192), (32, 64))):
            with self.subTest(shape=shape):
                self.assertEqual(choose_tiles(*shape), tiles)

    def test_batch_tile_boundaries(self):
        for rows, tile in (
            (0, 16),
            (1, 16),
            (16, 16),
            (17, 32),
            (32, 32),
            (33, 64),
            (64, 64),
            (65, 128),
            (128, 128),
            (129, 256),
            (255, 256),
        ):
            with self.subTest(rows=rows):
                self.assertEqual(select_batch_tile(rows), tile)
                validate_tiles(tile, 256, 128, 8)

    def test_rejects_unsupported_tiles(self):
        for tiles in ((512, 128, 256, 8), (16, 31, 256, 8), (16, 128, 65, 8), (16, 128, 256, 7)):
            with self.subTest(tiles=tiles), self.assertRaises(ValueError):
                validate_tiles(*tiles)
