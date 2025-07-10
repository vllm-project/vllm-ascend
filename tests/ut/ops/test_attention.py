import unittest

import torch

from vllm_ascend.ops.attention import vanilla_chunked_prefill

class TestAttention(unittest.TestCase):
    def setUp(self):
        self.scale = 0.1
        self.num_query_heads = 8
        self.num_kv_heads = 8
        self.num_tokens = 70
        self.num_blocks = 100
        self.block_size = 4
        self.max_num_blocks_per_seq = 30
        self.head_dim = 1024
        self.causal = False
        self.max_seqlen_k = 60
        self.max_seqlen_q = 50
        self.num_batch = 2
        self.alibi_slopes = torch.tensor(0.1, device="npu")
        self.block_tables = torch.randint(0, 100, (self.num_batch, self.max_num_blocks_per_seq), device="npu")
        self.output = torch.randn((self.num_tokens, self.num_query_heads, self.head_dim), device="npu")
        self.query = torch.randn((self.num_tokens, self.num_query_heads, self.head_dim), device="npu")
        self.key_cache = torch.randn((self.num_blocks, self.block_size, self.num_kv_heads, self.head_dim), device="npu")
        self.value_cache = torch.randn((self.num_blocks, self.block_size, self.num_kv_heads, self.head_dim), device="npu")
        self.cu_seqlen_q = torch.tensor([[0],[30],[70]], device="npu")
        self.cu_seqlen_k = torch.tensor([[0],[40],[90]], device="npu")

    def test_vanila_chunked_prefill(self):
        output = vanilla_chunked_prefill(self.output, 
                                        self.query,
                                        self.key_cache,
                                        self.value_cache,
                                        self.block_tables,
                                        self.cu_seqlen_q,
                                        self.cu_seqlen_k,
                                        self.max_seqlen_q,
                                        self.max_seqlen_k,
                                        self.scale,
                                        self.alibi_slopes,
                                        self.causal)
        self.assertEqual(output.shape[0], self.num_tokens)  
        self.assertEqual(output.shape[1], self.num_query_heads)
        self.assertEqual(output.shape[2], self.head_dim)

    
