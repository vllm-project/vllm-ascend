import unittest

import torch
import torch_npu


class TestPaKvCacheOps(unittest.TestCase):
    def test_scatter_pa_kv_cache_slot_mapping_zero_and_minus_one(self):
        torch.manual_seed(20260709)

        dtype = torch.float16
        block_size = 4
        num_blocks = 2
        num_heads = 1
        head_dim = 8
        slot_mapping = torch.tensor([0, -1, 3], dtype=torch.int32, device="npu")
        key = torch.arange(3 * num_heads * head_dim, dtype=dtype, device="npu").view(3, num_heads, head_dim)
        value = key + 100
        key_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device="npu")
        value_cache = torch.randn_like(key_cache)

        expected_key_cache = key_cache.clone()
        expected_value_cache = value_cache.clone()
        for token_idx, slot in enumerate(slot_mapping.cpu().tolist()):
            if slot < 0:
                continue
            expected_key_cache[slot // block_size, slot % block_size] = key[token_idx]
            expected_value_cache[slot // block_size, slot % block_size] = value[token_idx]

        torch_npu.npu_scatter_pa_kv_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            cache_mode="Norm",
        )
        torch.npu.synchronize()

        torch.testing.assert_close(key_cache, expected_key_cache, atol=0, rtol=0)
        torch.testing.assert_close(value_cache, expected_value_cache, atol=0, rtol=0)

    def test_scatter_pa_kv_cache_all_minus_one_leaves_cache_unchanged(self):
        dtype = torch.float16
        key = torch.randn(2, 1, 8, dtype=dtype, device="npu")
        value = torch.randn_like(key)
        key_cache = torch.randn(2, 4, 1, 8, dtype=dtype, device="npu")
        value_cache = torch.randn_like(key_cache)
        expected_key_cache = key_cache.clone()
        expected_value_cache = value_cache.clone()
        slot_mapping = torch.tensor([-1, -1], dtype=torch.int32, device="npu")

        torch_npu.npu_scatter_pa_kv_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            cache_mode="Norm",
        )
        torch.npu.synchronize()

        torch.testing.assert_close(key_cache, expected_key_cache, atol=0, rtol=0)
        torch.testing.assert_close(value_cache, expected_value_cache, atol=0, rtol=0)
