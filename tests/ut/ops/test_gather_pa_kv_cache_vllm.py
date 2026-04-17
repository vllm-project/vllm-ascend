import random
import numpy as np
import torch
import torch_npu
import math
import vllm_ascend
from torch_npu.testing.testcase import TestCase, run_tests

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

class TestGatherPaKvCacheVllm(TestCase):
    num_heads = 4
    head_size_k = 128
    head_size_v = 128
    block_size = 16
    num_blocks = 128
    batch = 4

    def _build_block_tables(self, batch, per_batch_lens, block_size, num_blocks,
                            seq_offset=None):
        """Allocate unique physical blocks for each batch's sequence."""
        blocks_needed = []
        for i in range(batch):
            offset = int(seq_offset[i]) if seq_offset is not None else 0
            n = (per_batch_lens[i] + offset + block_size - 1) // block_size
            blocks_needed.append(n)

        max_blocks_per_seq = max(blocks_needed)
        total_needed = sum(blocks_needed)
        assert total_needed <= num_blocks, (
            f"Need {total_needed} unique blocks but only {num_blocks} available"
        )

        all_blocks = list(range(num_blocks))
        random.shuffle(all_blocks)
        block_tables = torch.zeros((batch, max_blocks_per_seq), dtype=torch.int32)
        idx = 0
        for i in range(batch):
            for b in range(blocks_needed[i]):
                block_tables[i][b] = all_blocks[idx]
                idx += 1
        return block_tables

    def cal_reference(self, key_cache, value_cache, block_tables, seq_lens,
                      is_seq_lens_cumsum=False, seq_offset=None):
        """
        CPU reference: gather key/value from paged KV cache.
        For each batch, iterate over tokens and copy from the physical
        block/offset indicated by block_tables into contiguous output.
        """
        batch_size = block_tables.shape[0]
        block_size = key_cache.shape[1]
        num_heads_k = key_cache.shape[2]
        hsk = key_cache.shape[3]
        num_heads_v = value_cache.shape[2]
        hsv = value_cache.shape[3]

        if is_seq_lens_cumsum:
            batch_lens = [
                int(seq_lens[i + 1] - seq_lens[i]) for i in range(batch_size)
            ]
            batch_start = [int(seq_lens[i]) for i in range(batch_size)]
            total_tokens = int(seq_lens[-1])
        else:
            batch_lens = [int(seq_lens[i]) for i in range(batch_size)]
            batch_start = []
            acc = 0
            for s in batch_lens:
                batch_start.append(acc)
                acc += s
            total_tokens = acc

        key_out = torch.zeros(
            (total_tokens, num_heads_k, hsk), dtype=key_cache.dtype
        )
        value_out = torch.zeros(
            (total_tokens, num_heads_v, hsv), dtype=value_cache.dtype
        )

        for i in range(batch_size):
            table_offset = (int(seq_offset[i]) // block_size) if seq_offset is not None else 0
            for j in range(batch_lens[i]):
                bt_idx = j // block_size + table_offset
                blk_off = j % block_size
                phys_block = int(block_tables[i][bt_idx])
                key_out[batch_start[i] + j] = key_cache[phys_block][blk_off]
                value_out[batch_start[i] + j] = value_cache[phys_block][blk_off]

        return key_out, value_out

    # ======================== cumsum=True ========================
    def test_gather_pa_kv_cache_cumsum(self):
        """Fixed params, is_seq_lens_cumsum=True"""
        per_batch_lens = [12, 8, 16, 4]
        total_tokens = sum(per_batch_lens)

        cumsum = [0]
        for length in per_batch_lens:
            cumsum.append(cumsum[-1] + length)
        seq_lens = torch.tensor(cumsum, dtype=torch.int32)

        block_tables = self._build_block_tables(
            self.batch, per_batch_lens, self.block_size, self.num_blocks
        )
        key_cache = torch.rand(
            (self.num_blocks, self.block_size, self.num_heads, self.head_size_k),
            dtype=torch.float16,
        )
        value_cache = torch.rand(
            (self.num_blocks, self.block_size, self.num_heads, self.head_size_v),
            dtype=torch.float16,
        )
        key_ref = torch.zeros(
            (total_tokens, self.num_heads, self.head_size_k), dtype=torch.float16
        )
        value_ref = torch.zeros(
            (total_tokens, self.num_heads, self.head_size_v), dtype=torch.float16
        )

        key_expected, value_expected = self.cal_reference(
            key_cache, value_cache, block_tables, seq_lens,
            is_seq_lens_cumsum=True,
        )

        result_key, result_value = torch.ops._C_ascend.npu_gather_pa_kv_cache_vllm(
            key_cache.npu(), value_cache.npu(),
            block_tables.npu(), seq_lens.npu(),
            key_ref.npu(), value_ref.npu(),
            None, "Norm", True,
        )

        self.assertRtolEqual(key_expected, result_key.cpu())
        self.assertRtolEqual(value_expected, result_value.cpu())

    # ======================== cumsum=False ========================
    def test_gather_pa_kv_cache_non_cumsum(self):
        """Fixed params, is_seq_lens_cumsum=False"""
        per_batch_lens = [12, 8, 16, 4]
        total_tokens = sum(per_batch_lens)

        seq_lens = torch.tensor(per_batch_lens, dtype=torch.int32)

        block_tables = self._build_block_tables(
            self.batch, per_batch_lens, self.block_size, self.num_blocks
        )
        key_cache = torch.rand(
            (self.num_blocks, self.block_size, self.num_heads, self.head_size_k),
            dtype=torch.float16,
        )
        value_cache = torch.rand(
            (self.num_blocks, self.block_size, self.num_heads, self.head_size_v),
            dtype=torch.float16,
        )
        key_ref = torch.zeros(
            (total_tokens, self.num_heads, self.head_size_k), dtype=torch.float16
        )
        value_ref = torch.zeros(
            (total_tokens, self.num_heads, self.head_size_v), dtype=torch.float16
        )

        key_expected, value_expected = self.cal_reference(
            key_cache, value_cache, block_tables, seq_lens,
            is_seq_lens_cumsum=False,
        )

        result_key, result_value = torch.ops._C_ascend.npu_gather_pa_kv_cache_vllm(
            key_cache.npu(), value_cache.npu(),
            block_tables.npu(), seq_lens.npu(),
            key_ref.npu(), value_ref.npu(),
            None, "Norm", False,
        )

        self.assertRtolEqual(key_expected, result_key.cpu())
        self.assertRtolEqual(value_expected, result_value.cpu())

    # ======================== with seq_offset ========================
    def test_gather_pa_kv_cache_with_seq_offset(self):
        """Test with seq_offset provided"""
        per_batch_lens = [8, 6, 10, 4]
        seq_offset_vals = [16, 0, 32, 16]
        total_tokens = sum(per_batch_lens)

        seq_lens = torch.tensor(per_batch_lens, dtype=torch.int32)
        seq_offset = torch.tensor(seq_offset_vals, dtype=torch.int32)

        block_tables = self._build_block_tables(
            self.batch, per_batch_lens, self.block_size, self.num_blocks,
            seq_offset=seq_offset,
        )
        key_cache = torch.rand(
            (self.num_blocks, self.block_size, self.num_heads, self.head_size_k),
            dtype=torch.float16,
        )
        value_cache = torch.rand(
            (self.num_blocks, self.block_size, self.num_heads, self.head_size_v),
            dtype=torch.float16,
        )
        key_ref = torch.zeros(
            (total_tokens, self.num_heads, self.head_size_k), dtype=torch.float16
        )
        value_ref = torch.zeros(
            (total_tokens, self.num_heads, self.head_size_v), dtype=torch.float16
        )

        key_expected, value_expected = self.cal_reference(
            key_cache, value_cache, block_tables, seq_lens,
            is_seq_lens_cumsum=False, seq_offset=seq_offset,
        )

        result_key, result_value = torch.ops._C_ascend.npu_gather_pa_kv_cache_vllm(
            key_cache.npu(), value_cache.npu(),
            block_tables.npu(), seq_lens.npu(),
            key_ref.npu(), value_ref.npu(),
            seq_offset.npu(), "Norm", False,
        )

        self.assertRtolEqual(key_expected, result_key.cpu())
        self.assertRtolEqual(value_expected, result_value.cpu())

    # ======================== random params ========================
    def test_gather_pa_kv_cache_random(self):
        """Random dimension test"""
        batch = np.random.randint(1, 9)
        num_heads = np.random.randint(1, 17)
        head_size_k = int(np.random.choice([64, 128, 144]))
        head_size_v = int(np.random.choice([64, 128, 144]))
        block_size = int(np.random.choice([16, 32, 64, 128]))
        num_blocks = 256

        per_batch_lens = [int(np.random.randint(1, 65)) for _ in range(batch)]
        total_tokens = sum(per_batch_lens)

        cumsum = [0]
        for length in per_batch_lens:
            cumsum.append(cumsum[-1] + length)
        seq_lens = torch.tensor(cumsum, dtype=torch.int32)

        block_tables = self._build_block_tables(
            batch, per_batch_lens, block_size, num_blocks
        )
        key_cache = torch.rand(
            (num_blocks, block_size, num_heads, head_size_k), dtype=torch.float16
        )
        value_cache = torch.rand(
            (num_blocks, block_size, num_heads, head_size_v), dtype=torch.float16
        )
        key_ref = torch.zeros(
            (total_tokens, num_heads, head_size_k), dtype=torch.float16
        )
        value_ref = torch.zeros(
            (total_tokens, num_heads, head_size_v), dtype=torch.float16
        )

        key_expected, value_expected = self.cal_reference(
            key_cache, value_cache, block_tables, seq_lens,
            is_seq_lens_cumsum=True,
        )

        result_key, result_value = torch.ops._C_ascend.npu_gather_pa_kv_cache_vllm(
            key_cache.npu(), value_cache.npu(),
            block_tables.npu(), seq_lens.npu(),
            key_ref.npu(), value_ref.npu(),
            None, "Norm", True,
        )

        self.assertRtolEqual(key_expected, result_key.cpu())
        self.assertRtolEqual(value_expected, result_value.cpu())


    # ======================== non-contiguous KV cache (stride) ========================
    def test_gather_pa_kv_cache_non_contiguous(self):
        """Test with non-contiguous KV Cache using K/V interleaved layout"""
        per_batch_lens = [12, 8, 16, 4]
        total_tokens = sum(per_batch_lens)
        head_size = self.head_size_k

        cumsum = [0]
        for length in per_batch_lens:
            cumsum.append(cumsum[-1] + length)
        seq_lens = torch.tensor(cumsum, dtype=torch.int32)

        block_tables = self._build_block_tables(
            self.batch, per_batch_lens, self.block_size, self.num_blocks
        )

        kv_shape = [2, self.num_blocks, self.block_size, self.num_heads, head_size]
        kv_cache_flat_cpu = torch.rand(math.prod(kv_shape), dtype=torch.float16)
        kv_cache_cpu = kv_cache_flat_cpu.view(kv_shape)
        hidden_size = kv_cache_cpu.shape[2:].numel()
        kv_cache_cpu.as_strided_(
            size=kv_cache_cpu.shape,
            stride=(hidden_size, 2 * hidden_size, *kv_cache_cpu.stride()[2:]),
        )
        key_cache_cpu, value_cache_cpu = kv_cache_cpu.unbind(0)

        key_expected, value_expected = self.cal_reference(
            key_cache_cpu, value_cache_cpu, block_tables, seq_lens,
            is_seq_lens_cumsum=True,
        )

        kv_cache_flat_npu = kv_cache_flat_cpu.npu()
        kv_cache_npu = kv_cache_flat_npu.view(kv_shape)
        kv_cache_npu.as_strided_(
            size=kv_cache_npu.shape,
            stride=(hidden_size, 2 * hidden_size, *kv_cache_npu.stride()[2:]),
        )
        key_cache_npu, value_cache_npu = kv_cache_npu.unbind(0)

        self.assertFalse(key_cache_npu.is_contiguous())
        self.assertFalse(value_cache_npu.is_contiguous())

        key_ref = torch.zeros(
            (total_tokens, self.num_heads, head_size), dtype=torch.float16
        )
        value_ref = torch.zeros(
            (total_tokens, self.num_heads, head_size), dtype=torch.float16
        )

        result_key, result_value = torch.ops._C_ascend.npu_gather_pa_kv_cache_vllm(
            key_cache_npu, value_cache_npu,
            block_tables.npu(), seq_lens.npu(),
            key_ref.npu(), value_ref.npu(),
            None, "Norm", True,
        )

        self.assertRtolEqual(key_expected, result_key.cpu())
        self.assertRtolEqual(value_expected, result_value.cpu())


if __name__ == '__main__':
    run_tests()
