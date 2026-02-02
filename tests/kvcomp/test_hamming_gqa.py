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

DEVICE_ID = 0
enable_custom_op()

torch_npu.npu.config.allow_internal_format = True

class TestCustomHammingDistTopK(TestCase):
    
    def test_hamming_dist_top_k_graph(self):

        print('=======================data=======================')
        device = 'cpu'
        batch_size = 5
        num_head = 16    # 16 ok
        num_kv_head = 1
        head_dim = 128
        compress_rate = 8
        compressed_dim = head_dim // compress_rate
        compressed_dim = head_dim // compress_rate

        seqlen_q = 1
        sparse_ratio = 0.2
        chunk_size_value = 128

        seqlen_list = [30720] * batch_size
        seqlen = torch.tensor(seqlen_list, dtype=torch.int32, device=device)

        max_seq_len = max(seqlen_list)

        
        chunk_size_list = [chunk_size_value] * batch_size
        chunk_size = torch.tensor(chunk_size_list, dtype=torch.int32, device=device)

        top_k_list = [seq * sparse_ratio // chunk_size_list[0] for seq in seqlen_list]
        top_k = torch.tensor(top_k_list, dtype=torch.int32, device=device)

        num_chunks = seqlen // chunk_size

        block_size = 128
        num_blocks_per_seq = (seqlen + block_size - 1) // block_size  # 天花板除
        num_blocks = num_blocks_per_seq.sum().item() + 5
        
        qhash = torch.randint(255, (batch_size, num_head, seqlen_q, compressed_dim), dtype=torch.uint8, device=device)

        khash = torch.randint(255, (num_blocks, num_kv_head, block_size, compressed_dim), dtype=torch.uint8, device=device)

        sink = 1
        recent = 4

        print(f'seqlen: {seqlen}')
        print(f'top_k: {top_k}')
        print(f'chunk_size: {chunk_size}')
        print(f'num_chunks: {num_chunks}')
        print(f'block_size: {block_size}')
        print(f'num_blocks_per_seq: {num_blocks_per_seq}')
        print(f'num_blocks: {num_blocks}')

        print(f'qhash shape: {qhash.shape}')  # torch.Size([2, 16, 1, 72])
        print(f'khash shape: {khash.shape}')  # torch.Size([245, 1, 128, 72])

        print(f'max_seq_len: {max_seq_len}')
        print(f'sink: {sink}')
        print(f'recent: {recent}')

        # 初始化block_table
        max_num_blocks_per_seq = (max(seqlen_list) + block_size - 1) // block_size + 5
        block_table = torch.full((len(num_blocks_per_seq), max_num_blocks_per_seq), fill_value=0, dtype=torch.int32)

        shuffle = False
        start = 1  # 1
        for i, n in enumerate(num_blocks_per_seq):
            if shuffle:
                ids = torch.arange(start, start + n, dtype=torch.int32)
                idx  = torch.randperm(n)
                block_table[i, :n] = ids[idx]
            else:
                block_table[i, :n] = torch.arange(start, start + n, dtype=torch.int32)
            start += n
        block_table = block_table.to(device=device)
        print(f'block_table.shape: {block_table.shape}')
        # print(f'block_table: {block_table}')
        indices = torch.zeros([batch_size, num_kv_head, 128], dtype=torch.int32)

        support_offload = 1
        mask = torch.tensor([True, True, False, False, False])
        
        # start run custom ops
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, qhash, khash, khash_rope, top_k, seqlen, chunk_size, max_seq_len, sink, recent, support_offload, block_table, mask, indices):

                out1 = torch.ops._C_ascend.npu_hamming_dist_top_k(qhash, khash, None, top_k, seqlen, chunk_size, max_seq_len, sink, recent, support_offload, block_table, mask, indices)

                return out1
        
        print(f'======================== PTA graph BEGIN ========================')

        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        device_id = 0
        torch.npu.set_device(device_id)
        npu = f'npu:{device_id}'

        npu_mode = torch.compile(npu_mode, backend=npu_backend, dynamic=False)
        npu_out = npu_mode(qhash.to(npu), khash.to(npu), None, top_k.to(npu), seqlen.to(npu), chunk_size.to(npu), max_seq_len, sink, recent, support_offload, block_table.to(npu), mask.to(npu), indices.to(npu))

        print(f'acl graph npu_out = {npu_out}')

        print("test hamming eager.....")
        output_eager = torch.ops._C_ascend.npu_hamming_dist_top_k(qhash.to(npu), khash.to(npu), None, top_k.to(npu), seqlen.to(npu), chunk_size.to(npu), max_seq_len, sink, recent, support_offload, block_table.to(npu), mask.to(npu), indices.to(npu))
        print(f'eager npu_out = {output_eager}')

if __name__ == "__main__":
    run_tests()