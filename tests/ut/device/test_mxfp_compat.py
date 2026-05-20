# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

import torch

from vllm_ascend.device.mxfp_compat import (
    MXFP_KV_SCALE_GROUP_SIZE,
    MXFP8_GROUP_SIZE,
    MXFP_SCALE_DTYPE_SIZE,
    mxfp_k_scale_numel,
    mxfp_kv_page_size_bytes,
    mxfp_k_scale_page_bytes,
    mxfp_v_scale_page_bytes,
    mxfp_resolve_kv_cache_layout,
    mxfp_v_scale_numel,
    scatter_mxfp_v_scale_cache,
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

    def test_k_and_v_scale_page_bytes_unified_formula(self):
        block_size = 128
        num_kv_heads = 8
        head_dim = 128
        expected = num_kv_heads * block_size * head_dim // MXFP8_GROUP_SIZE
        self.assertEqual(expected, 4096)
        self.assertEqual(mxfp_k_scale_page_bytes(num_kv_heads, block_size, head_dim), expected)
        self.assertEqual(mxfp_v_scale_page_bytes(num_kv_heads, block_size, head_dim), expected)
        self.assertEqual(
            mxfp_k_scale_numel(1, block_size, num_kv_heads, head_dim),
            mxfp_v_scale_numel(1, block_size, num_kv_heads, head_dim),
        )
        self.assertEqual(mxfp_k_scale_numel(16, block_size, num_kv_heads, head_dim), 16 * expected)

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

        k_shape, v_shape, k_scale_shape, v_scale_shape = mxfp_resolve_kv_cache_layout(
            raw_k_numel=raw_k,
            raw_v_numel=raw_v,
            raw_k_scale_numel=raw_k_scale,
            raw_v_scale_numel=raw_v_scale,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            k_dim=k_dim,
            v_dim=v_dim,
        )
        self.assertEqual(k_shape, (1476, 128, 8, 128))
        self.assertEqual(k_shape[0], num_blocks)
        self.assertEqual(k_scale_shape, (1476, 8, 128, 2, 2))
        self.assertEqual(v_scale_shape, (1476, 8, 2, 128, 2))
        self.assertEqual(k_shape[0] * k_shape[1] * k_shape[2] * k_shape[3], raw_k)
        scale_numel = 1
        for dim in k_scale_shape:
            scale_numel *= dim
        self.assertEqual(scale_numel, raw_k_scale)

    def test_resolve_layout_rejects_spec_k_dim_mismatch_with_buffers(self):
        block_size = 128
        num_kv_heads = 8
        num_blocks = 1476
        true_k_dim = 128
        wrong_k_dim = 64
        raw_k = num_blocks * block_size * num_kv_heads * true_k_dim
        raw_v = raw_k
        raw_k_scale = mxfp_k_scale_numel(num_blocks, block_size, num_kv_heads, true_k_dim)
        raw_v_scale = mxfp_v_scale_numel(num_blocks, block_size, num_kv_heads, true_k_dim)

        with self.assertRaises(ValueError):
            mxfp_resolve_kv_cache_layout(
                raw_k_numel=raw_k,
                raw_v_numel=raw_v,
                raw_k_scale_numel=raw_k_scale,
                raw_v_scale_numel=raw_v_scale,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                k_dim=wrong_k_dim,
                v_dim=true_k_dim,
                num_blocks_hint=num_blocks,
            )


class TestMxfpVScaleCacheScatter(unittest.TestCase):
    def test_v_scale_slot_matches_qwen3_moe_formula(self):
        block_size = 128
        num_kv_heads = 2
        head_dim = 64
        num_blocks = 4
        groups_per_block = block_size // MXFP_KV_SCALE_GROUP_SIZE

        value_scale_cache = torch.zeros(
            num_blocks,
            num_kv_heads,
            groups_per_block,
            head_dim,
            2,
        )
        num_tokens = 100
        slots = torch.arange(num_tokens, dtype=torch.long)
        v_scale_slots_ref = (slots // MXFP_KV_SCALE_GROUP_SIZE).unique()

        num_scale_groups = (num_tokens + MXFP_KV_SCALE_GROUP_SIZE - 1) // MXFP_KV_SCALE_GROUP_SIZE
        value_scale = torch.arange(
            num_scale_groups * num_kv_heads * head_dim * 2,
            dtype=torch.float32,
        ).view(num_scale_groups, num_kv_heads, head_dim, 2)

        scatter_mxfp_v_scale_cache(
            value_scale,
            value_scale_cache,
            slots,
            block_size,
        )

        block_ids = v_scale_slots_ref // groups_per_block
        cache_group_ids = v_scale_slots_ref % groups_per_block
        write_group_ids = torch.arange(num_tokens) // MXFP_KV_SCALE_GROUP_SIZE
        slot_groups = slots // MXFP_KV_SCALE_GROUP_SIZE
        sort_idx = torch.argsort(slot_groups, stable=True)
        sorted_groups = slot_groups[sort_idx]
        sorted_write_groups = write_group_ids[sort_idx]
        unique_mask = torch.cat(
            (torch.tensor([True]), sorted_groups[1:] != sorted_groups[:-1])
        )
        unique_write_groups = sorted_write_groups[unique_mask]
        expected_by_slot = value_scale[unique_write_groups]

        torch.testing.assert_close(
            value_scale_cache[block_ids, :, cache_group_ids, :, :],
            expected_by_slot,
        )
