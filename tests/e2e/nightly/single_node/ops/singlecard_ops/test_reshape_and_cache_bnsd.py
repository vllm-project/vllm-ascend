import logging
import unittest
from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn
import torchair
from torchair import logger

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
logger.setLevel(logging.DEBUG)
torch._logging.set_logs(graph_code=True)


def run_tests():
    unittest.main()


class TestCustomReshapeAndCacheBnsd(TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

        self.device_id = 0
        torch.npu.set_device(self.device_id)
        self.npu = f"npu:{self.device_id}"

        self.num_blocks = 2
        self.num_kv_heads = 2
        self.block_size = 4
        self.head_size = 16
        self.token_num = 4
        self.bs = 1

        self.hashk_op = torch.randn(self.token_num * self.num_kv_heads, self.head_size, dtype=torch.float16).npu()
        self.hashk_cache_op = torch.randn(
            self.num_blocks, self.num_kv_heads, self.block_size, self.head_size, dtype=torch.float16
        ).npu()
        self.slot_mapping_op = torch.tensor([0, 1, 2, 3], dtype=torch.int32).npu()
        self.seq_lens_op = torch.tensor([self.token_num], dtype=torch.int32).npu()
        self.hashk_op_org = self.hashk_op.reshape(self.num_kv_heads, self.token_num, self.head_size)

    def _run_eager_mode(self):
        return torch.ops._C_ascend.npu_reshape_and_cache_bnsd(
            self.hashk_op, self.hashk_cache_op, self.slot_mapping_op, self.seq_lens_op, self.hashk_cache_op
        )

    def _run_graph_mode(self):
        class Network(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, hashk_op, hashk_cache_op, slot_mapping_op, seq_lens_op, k_cache_out):
                return torch.ops._C_ascend.npu_reshape_and_cache_bnsd(
                    hashk_op, hashk_cache_op, slot_mapping_op, seq_lens_op, k_cache_out
                )

        npu_mode = Network().to(f"npu:{self.device_id}")
        from torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        npu_mode = torch.compile(npu_mode, backend=npu_backend, dynamic=False)
        return npu_mode(self.hashk_op, self.hashk_cache_op, self.slot_mapping_op, self.seq_lens_op, self.hashk_cache_op)

    def _verify_results(self, output):
        for token_id in range(self.token_num):
            slot = self.slot_mapping_op[token_id].item()
            block_idx = slot // self.block_size
            block_offset = slot % self.block_size

            for kv_head_id in range(self.num_kv_heads):
                output_hashk = output[block_idx, kv_head_id, block_offset, :]

        token_id = self.bs - 1
        slot = self.slot_mapping_op[token_id].item() + 1
        block_idx = slot // self.block_size
        block_offset = slot % self.block_size

        for kv_head_id in range(self.num_kv_heads):
            output_hashk = output[block_idx, kv_head_id, block_offset, :]
            self.assertIsNotNone(output_hashk)

    def test_reshape_and_cache_bnsd_compare(self):
        print("=========== test_reshape_and_cache_bnsd_compare begin ===================")
        output_eager = self._run_eager_mode()
        output_graph = self._run_graph_mode()

        self.assertIsNotNone(output_eager, "output_eager should not be None")
        self.assertIsNotNone(output_graph, "output_graph should not be None")

        self.assertEqual(output_eager.shape, output_graph.shape)
        self.assertTrue(torch.allclose(output_eager, output_graph, atol=1e-05))
        print("=========== test_reshape_and_cache_bnsd_compare end ===================")

    def test_reshape_and_cache_bnsd_with_expected_output(self):
        print("=========== test_reshape_and_cache_bnsd_with_expected_output begin ===================")

        num_blocks = 2
        num_kv_heads = 2
        block_size = 4
        head_size = 8
        bs = 1
        seq_lens_list = [8]
        token_num = sum(seq_lens_list)

        key_in = torch.zeros(num_kv_heads, token_num, head_size, dtype=torch.uint8)
        val = 0
        for head_id in range(num_kv_heads):
            for token_id in range(token_num):
                for dim_id in range(head_size):
                    key_in[head_id, token_id, dim_id] = val
                    val += 1
        key_in = key_in.reshape(num_kv_heads * token_num, head_size).npu()
        key_cache_out = torch.randint(
            100, 200, (num_blocks, num_kv_heads, block_size, head_size), dtype=torch.uint8
        ).npu()
        slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32).npu()
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32).npu()
        key_cache_out_cpu = key_cache_out.cpu().clone()

        output = torch.ops._C_ascend.npu_reshape_and_cache_bnsd(
            key_in, key_cache_out, slot_mapping, seq_lens, key_cache_out
        )

        expected = key_cache_out_cpu.clone()
        key_in_cpu = key_in.cpu().view(num_kv_heads, token_num, head_size)
        for token_id in range(token_num):
            slot = slot_mapping[token_id].cpu().item()
            block_idx = slot // block_size
            block_offset = slot % block_size

            for head_id in range(num_kv_heads):
                src_data = key_in_cpu[head_id, token_id, :]
                expected[block_idx, head_id, block_offset, :] = src_data

        self.assertIsNotNone(output, "output should not be None")
        self.assertEqual(output.shape, expected.shape, "output shape should match expected shape")

        output_cpu = output.cpu()
        self.assertTrue(
            torch.equal(output_cpu, expected),
            f"output should match expected values\noutput:\n{output_cpu}\nexpected:\n{expected}",
        )

        print(f"bs: {bs}, seq_lens: {seq_lens_list}, total_tokens: {token_num}")
        print(f"key_in shape: {key_in.shape}")
        print(f"key_cache_out shape: {key_cache_out.shape}")
        print(f"key_cache_out (before):\n{key_cache_out_cpu}")
        print(f"output:\n{output_cpu}")
        print(f"expected:\n{expected}")
        print("=========== test_reshape_and_cache_bnsd_with_expected_output end ===================")

    def test_reshape_and_cache_bnsd_bf16_shape(self):
        print("=========== test_reshape_and_cache_bnsd_bf16_shape begin ===================")

        num_blocks = 1
        num_kv_heads = 2
        block_size = 4
        head_size = 4
        token_num = 4

        key_in = torch.zeros(num_kv_heads, token_num, head_size, dtype=torch.bfloat16)
        for head_id in range(num_kv_heads):
            for token_id in range(token_num):
                start_val = head_id * token_num * head_size + token_id * head_size
                key_in[head_id, token_id, :] = torch.arange(start_val, start_val + head_size, dtype=torch.bfloat16)
        key_in = key_in.reshape(num_kv_heads * token_num, head_size).npu()
        key_cache_out = torch.zeros(num_blocks, num_kv_heads, block_size, head_size, dtype=torch.bfloat16).npu()
        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32).npu()
        seq_lens = torch.tensor([token_num], dtype=torch.int32).npu()
        output = torch.ops._C_ascend.npu_reshape_and_cache_bnsd(
            key_in, key_cache_out, slot_mapping, seq_lens, key_cache_out
        )

        expected_shape = (num_blocks, num_kv_heads, block_size, head_size)

        self.assertIsNotNone(output, "output should not be None")
        self.assertEqual(output.shape, expected_shape, "output shape should match expected shape")

        print(f"output shape: {output.shape}")
        print(f"output:\n{output.cpu()}")
        print("=========== test_reshape_and_cache_bnsd_bf16_shape end ===================")


if __name__ == "__main__":
    run_tests()
