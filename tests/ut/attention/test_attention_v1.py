import unittest

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl


class DummyNPU:

    @staticmethod
    def npu_scatter_nd_update_(tensor, indices, updates):
        batch = indices.shape[0]
        for i in range(batch):
            b = indices[i, 0, 0].item()
            o = indices[i, 0, 1].item()
            tensor[b, o] = updates[i]


class TestUpdateKVCache(unittest.TestCase):

    def test_basic_update(self):
        block_num, block_size = 3, 2
        num_heads, head_dim = 1, 1

        key_cache = torch.zeros(block_num, block_size, num_heads, head_dim)
        value_cache = torch.zeros_like(key_cache)

        key = torch.tensor([[[1.0]], [[2.0]]])
        value = torch.tensor([[[3.0]], [[4.0]]])

        slot_indices = torch.tensor([1, 3])

        AscendAttentionBackendImpl.update_kv_cache(key, value, key_cache,
                                                   value_cache, slot_indices)

        self.assertEqual(key_cache[0, 1, 0, 0].item(), 1.0)
        self.assertEqual(value_cache[0, 1, 0, 0].item(), 3.0)

        self.assertEqual(key_cache[1, 1, 0, 0].item(), 2.0)
        self.assertEqual(value_cache[1, 1, 0, 0].item(), 4.0)


if __name__ == '__main__':
    unittest.main()
