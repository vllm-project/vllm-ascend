import random
import numpy as np
import torch
import torch_npu
import math
import vllm_ascend
from torch_npu.testing.testcase import TestCase, run_tests

import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401

import vllm_ascend.meta_registration  # type: ignore  # noqa: F401

class TestScatterPaKvCache(TestCase):
    num_tokens = 14
    num_head = 4
    head_size = 16
    block_size = 16
    num_blocks = 100

    def cal_nd(self, key, value, key_cache, value_cache, slot_mapping):
        key_expect = key_cache.clone()
        value_expect = value_cache.clone()
        for i, slot in enumerate(slot_mapping):
            if slot < 0:
                continue
            block_index = slot // self.block_size
            block_offset = slot % self.block_size
            token_key = key[i]
            token_v = value[i]
            key_expect[block_index][block_offset] = token_key
            value_expect[block_index][block_offset] = token_v
        return key_expect, value_expect


    def test_reshape_and_cache(self):
        head_size_k = 256
        head_size_v = head_size_k

        key = torch.rand((self.num_tokens, self.num_head, head_size_k), dtype=torch.float16)
        value = torch.rand((self.num_tokens, self.num_head, head_size_k), dtype=torch.float16)

        num_slots = self.block_size * self.num_blocks
        slot_list = random.sample(range(num_slots), self.num_tokens)
        slot_mapping = np.array(slot_list).astype(np.int32)

        kv_shape = [2, self.num_blocks, self.block_size, self.num_head, head_size_k]
        cache_size = math.prod(kv_shape)
        kv_cache_tensor = torch.zeros(cache_size, device="npu", dtype=torch.float16)
        kv_cache = kv_cache_tensor.view(kv_shape)

        hidden_size = kv_cache.shape[2:].numel()
        kv_cache.as_strided_(
            size=kv_cache.shape,
            stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
        )
        key_cache, value_cache = kv_cache.unbind(0)

        key_expect, value_expect = self.cal_nd(key, value, key_cache, value_cache, slot_mapping)
        key = key.npu()
        value = value.npu()
        key_cache = key_cache.npu()
        value_cache = value_cache.npu()
        slot_mapping = torch.from_numpy(slot_mapping).to(torch.int32).npu()
        
        torch.ops._C_ascend.npu_scatter_pa_kv_cache_vllm(key, value, key_cache, value_cache, slot_mapping, cache_mode="Norm")
        self.assertRtolEqual(key_expect, key_cache)
        self.assertRtolEqual(value_expect, value_cache)


    def test_reshape_and_cache_origin(self):
        head_size_k = np.random.randint(1, 256)
        head_size_v = np.random.randint(1, 256)

        key = torch.rand((self.num_tokens, self.num_head, head_size_k), dtype=torch.float16)
        value = torch.rand((self.num_tokens, self.num_head, head_size_v), dtype=torch.float16)
        num_slots = self.block_size * self.num_blocks
        slot_list = random.sample(range(num_slots), self.num_tokens)
        slot_mapping = np.array(slot_list).astype(np.int32)
        key_cache = torch.zeros((self.num_blocks, self.block_size, self.num_head, head_size_k), dtype=torch.float16)
        value_cache = torch.zeros((self.num_blocks, self.block_size, self.num_head, head_size_v), dtype=torch.float16)
        key_expect, value_expect = self.cal_nd(key, value, key_cache, value_cache, slot_mapping)
        key = key.npu()
        value = value.npu()
        key_cache = key_cache.npu()
        value_cache = value_cache.npu()
        slot_mapping = torch.from_numpy(slot_mapping).to(torch.int32).npu()
        torch.ops._C_ascend.npu_scatter_pa_kv_cache_vllm(key, value, key_cache, value_cache, slot_mapping, cache_mode="Norm")
        self.assertRtolEqual(key_expect, key_cache)
        self.assertRtolEqual(value_expect, value_cache)

if __name__ == '__main__':
    run_tests()
