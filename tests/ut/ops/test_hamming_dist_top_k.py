import torch
import torch_npu
import torchair
import numpy as np
import torch.nn as nn

from torch_npu.testing.testcase import TestCase, run_tests

import logging
from torchair import logger
from vllm_ascend.utils import enable_custom_op

logger.setLevel(logging.DEBUG)
torch._logging.set_logs(graph_code=True)

enable_custom_op()
torch_npu.npu.config.allow_internal_format = True


class TestCustomHammingDistTopK(TestCase):
    
    def setUp(self):
        self.device = 'cpu'
        self.batch_size = 5
        self.num_head = 16
        self.num_kv_head = 1
        self.head_dim = 128
        self.compress_rate = 8
        self.compressed_dim = self.head_dim // self.compress_rate
        self.seqlen_q = 1
        self.sparse_ratio = 0.2
        self.chunk_size_value = 128
        self.device_id = 0
        self.DEVICE_ID = 0
        
        self.seqlen_list = [30720] * self.batch_size
        self.seqlen = torch.tensor(self.seqlen_list, dtype=torch.int32, device=self.device)
        self.max_seq_len = max(self.seqlen_list)
        self.chunk_size = torch.tensor([self.chunk_size_value] * self.batch_size, 
                                      dtype=torch.int32, device=self.device)
        self.top_k_list = [int(seq * self.sparse_ratio // self.chunk_size_value) 
                         for seq in self.seqlen_list]
        self.top_k = torch.tensor(self.top_k_list, dtype=torch.int32, device=self.device)
        print("self.top_k_list", self.top_k_list, self.top_k)
        self.block_size = 128
        self.num_blocks_per_seq = (self.seqlen + self.block_size - 1) // self.block_size
        self.num_blocks = self.num_blocks_per_seq.sum().item() + 5
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.qhash = torch.randint(255, 
                                  (self.batch_size, self.num_head, self.seqlen_q, self.compressed_dim), 
                                  dtype=torch.uint8, device=self.device)
        self.khash = torch.randint(255, 
                                  (self.num_blocks, self.num_kv_head, self.block_size, self.compressed_dim), 
                                  dtype=torch.uint8, device=self.device)
        self.sink = 1
        self.recent = 4

        # 初始化block_table
        max_num_blocks_per_seq = (max(self.seqlen_list) + self.block_size - 1) // self.block_size + 5
        self.block_table = torch.full((len(self.num_blocks_per_seq), max_num_blocks_per_seq), 
                                     fill_value=0, dtype=torch.int32)
        start = 1
        for i, n in enumerate(self.num_blocks_per_seq):
            self.block_table[i, :n] = torch.arange(start, start + n, dtype=torch.int32)
            start += n
        self.block_table = self.block_table.to(device=self.device)
        self.indices = torch.zeros([self.batch_size, self.num_kv_head, 128], dtype=torch.int32)

        self.support_offload = 1
        self.mask = torch.tensor([True, True, False, False, False])
        
        torch.npu.set_device(self.device_id)
        self.npu = f'npu:{self.device_id}'

    def _run_eager_mode(self):
        """运行单算子模式"""
        output_eager = torch.ops._C_ascend.npu_hamming_dist_top_k(
            self.qhash.to(self.npu), self.khash.to(self.npu), None, self.top_k.to(self.npu), 
            self.seqlen.to(self.npu), self.chunk_size.to(self.npu), self.max_seq_len, 
            self.sink, self.recent, self.support_offload, self.block_table.to(self.npu), 
            self.mask.to(self.npu), self.indices.to(self.npu)
        )
        return output_eager

    def _run_graph_mode(self):
        """运行图模式"""
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, qhash, khash, khash_rope, top_k, seqlen, chunk_size, max_seq_len, 
                       sink, recent, support_offload, block_table, mask, indices):
                return torch.ops._C_ascend.npu_hamming_dist_top_k(
                    qhash, khash, None, top_k, seqlen, chunk_size, max_seq_len, 
                    sink, recent, support_offload, block_table, mask, indices
                )

        npu_mode = Network().to(f"npu:{self.DEVICE_ID}")
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        
        npu_mode = torch.compile(npu_mode, backend=npu_backend, dynamic=False)
        npu_out = npu_mode(
            self.qhash.to(self.npu), self.khash.to(self.npu), None, self.top_k.to(self.npu), 
            self.seqlen.to(self.npu), self.chunk_size.to(self.npu), self.max_seq_len, 
            self.sink, self.recent, self.support_offload, self.block_table.to(self.npu), 
            self.mask.to(self.npu), self.indices.to(self.npu)
        )
        return npu_out

    def test_hamming_dist_top_k_compare(self):
        """比较单算子模式和图模式的结果"""
        # 获取单算子模式的结果
        output_eager = self._run_eager_mode()
        
        # 获取图模式的结果
        output_graph = self._run_graph_mode()
        
        # 断言两种模式的结果相等
        print(f"===========output_eager {output_eager} ===================")
        print(f"===========output_graph {output_graph} ===================")
        self.assertEqual(output_eager.shape, output_graph.shape)
        self.assertTrue(torch.allclose(output_eager.float(), output_graph.float(), atol=1e-05))
