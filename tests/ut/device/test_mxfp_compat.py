# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from vllm_ascend.device.mxfp_compat import (
    MXFP_KV_SCALE_GROUP_SIZE,
    MXFP_SCALE_DTYPE_SIZE,
    mxfp_k_scale_numel,
    mxfp_kv_page_size_bytes,
    mxfp_resolve_kv_cache_layout,
    mxfp_v_scale_numel,
)


class TestMxfpKvPageSizeBytes(unittest.TestCase):
    def test_page_size_matches_kv_and_scale_split(self):
        block_size = 128
        num_kv_heads = 8
        k_dim = 128
        v_dim = 128
        kv_dtype_size = 1
        num_blocks = 16

        page_size = mxfp_kv_page_size_bytes(
            block_size,
            num_kv_heads,
            k_dim,
            v_dim,
            kv_dtype_size,
        )
        self.assertEqual(page_size * num_blocks, num_blocks * page_size)

        kv_bytes = num_blocks * block_size * num_kv_heads * (k_dim + v_dim) * kv_dtype_size
        scale_bytes = (
            mxfp_k_scale_numel(num_blocks, block_size, num_kv_heads, k_dim)
            + mxfp_v_scale_numel(num_blocks, block_size, num_kv_heads, v_dim)
        ) * MXFP_SCALE_DTYPE_SIZE
        self.assertEqual(page_size, kv_bytes // num_blocks + scale_bytes // num_blocks)
        self.assertEqual(page_size * num_blocks, kv_bytes + scale_bytes)

    def test_page_size_typical_gqa_shape(self):
        # block=128, heads=8, head=128, fp8 kv -> 270336 bytes/page
        page_size = mxfp_kv_page_size_bytes(128, 8, 128, 128, kv_dtype_size=1)
        self.assertEqual(page_size, 270336)

    def test_page_size_asymmetric_k_v_dims(self):
        block_size = 64
        num_kv_heads = 4
        k_dim = 128
        v_dim = 64
        page_size = mxfp_kv_page_size_bytes(block_size, num_kv_heads, k_dim, v_dim, kv_dtype_size=1)

        per_page_kv = block_size * num_kv_heads * (k_dim + v_dim)
        per_page_scale = (
            mxfp_k_scale_numel(1, block_size, num_kv_heads, k_dim)
            + mxfp_v_scale_numel(1, block_size, num_kv_heads, v_dim)
        ) * MXFP_SCALE_DTYPE_SIZE
        self.assertEqual(page_size, per_page_kv + per_page_scale)

    def test_page_size_rejects_head_dim_not_divisible_by_64(self):
        with self.assertRaises(ValueError) as ctx:
            mxfp_kv_page_size_bytes(128, 8, 96, 128, kv_dtype_size=1)
        self.assertIn(str(MXFP_KV_SCALE_GROUP_SIZE), str(ctx.exception))

    def test_page_size_rejects_block_size_not_divisible_by_64(self):
        with self.assertRaises(ValueError) as ctx:
            mxfp_kv_page_size_bytes(32, 8, 128, 128, kv_dtype_size=1)
        self.assertIn(str(MXFP_KV_SCALE_GROUP_SIZE), str(ctx.exception))

    def test_resolve_layout_from_reported_runtime_sizes(self):
        # MiniMax-like: 1476 blocks, block=128, 8 heads, head=128
        block_size = 128
        num_kv_heads = 8
        k_dim = 128
        v_dim = 128
        num_blocks = 1476
        raw_k = num_blocks * block_size * num_kv_heads * k_dim
        raw_v = raw_k
        raw_k_scale = mxfp_k_scale_numel(num_blocks, block_size, num_kv_heads, k_dim)
        raw_v_scale = mxfp_v_scale_numel(num_blocks, block_size, num_kv_heads, v_dim)
        self.assertEqual(raw_k_scale, 6045696)

        resolved_blocks, k_shape, v_shape, k_scale_shape, v_scale_shape = mxfp_resolve_kv_cache_layout(
            raw_k_numel=raw_k,
            raw_v_numel=raw_v,
            raw_k_scale_numel=raw_k_scale,
            raw_v_scale_numel=raw_v_scale,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            k_dim=k_dim,
            v_dim=v_dim,
        )
        self.assertEqual(resolved_blocks, 1476)
        self.assertEqual(k_shape, (1476, 128, 8, 128))
        self.assertEqual(k_scale_shape, (1476, 8, 128, 2, 2))
        self.assertEqual(v_scale_shape, (1476, 8, 2, 128, 2))
        self.assertEqual(k_shape[0] * k_shape[1] * k_shape[2] * k_shape[3], raw_k)
        scale_numel = 1
        for dim in k_scale_shape:
            scale_numel *= dim
        self.assertEqual(scale_numel, raw_k_scale)
