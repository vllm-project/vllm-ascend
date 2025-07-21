import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.cache import concat_and_cache_mla


class TestCache(TestBase):

    def setUp(self):
        self.num_tokens = 4
        self.num_kv_head = 1
        self.nope = 3
        self.rope = 4
        self.num_blocks = 2
        self.block_size = 2
        self.kv_c_normed = torch.randn(self.num_tokens, self.nope)
        self.k_pe = torch.randn(self.num_tokens, self.num_kv_head, self.rope)
        self.kv_cache = torch.zeros(self.num_blocks, self.block_size,
                                    self.num_kv_head, self.nope + self.rope)
        self.slot_mapping = torch.tensor([0, 1, 2, 3])

    def test_concat_and_cache_mla(self):
        concat_and_cache_mla(self.kv_c_normed, self.k_pe, self.kv_cache,
                             self.slot_mapping)
        kv_cache_reshaped = self.kv_cache.view(
            self.num_blocks * self.block_size, self.num_kv_head, -1)
        expected = torch.cat([self.kv_c_normed.unsqueeze(1), self.k_pe],
                             dim=-1).squeeze(1)
        for i in range(self.num_tokens):
            idx = self.slot_mapping[i].item()
            self.assertTrue(
                torch.allclose(kv_cache_reshaped[idx], expected[i]))
