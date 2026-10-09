import unittest

import torch
import torch_npu

from vllm_ascend.worker.kv_cache_layout import reshape_combined_attention_kv_cache


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

    def test_padded_shared_pages_scatter_and_fia_preserve_other_states(self):
        logical_block_size = 128
        head_dim = 128
        num_kv_heads = 1
        num_query_heads = 2
        num_physical_pages = 2
        num_tokens = 4
        region_offset_bytes = 128
        guard_bytes = 128
        mamba_state_elements = 8
        generator = torch.Generator().manual_seed(20261009)

        for dtype in (torch.float16, torch.bfloat16):
            for padding_bytes in (0, 4096):
                for kernel_blocks_per_page in (1, 2):
                    with self.subTest(dtype=dtype, padding=padding_bytes, splits=kernel_blocks_per_page):
                        kernel_block_size = logical_block_size // kernel_blocks_per_page
                        page_stride_bytes = (
                            2 * logical_block_size * num_kv_heads * head_dim * dtype.itemsize + padding_bytes
                        )
                        region_bytes = num_physical_pages * page_stride_bytes
                        raw = torch.zeros(
                            region_offset_bytes + region_bytes + guard_bytes,
                            dtype=torch.uint8,
                            device="npu",
                        )
                        raw[:region_offset_bytes].fill_(83)
                        raw[-guard_bytes:].fill_(97)
                        region = raw[region_offset_bytes : region_offset_bytes + region_bytes]
                        cache_shape = (
                            2,
                            num_physical_pages * kernel_blocks_per_page,
                            kernel_block_size,
                            num_kv_heads,
                            head_dim,
                        )
                        key_cache, value_cache = reshape_combined_attention_kv_cache(
                            region, cache_shape, dtype, page_stride_bytes, kernel_blocks_per_page
                        )

                        # 模拟另一物理页中的 Mamba 状态，与本次写入的 Attention 页隔离。
                        mamba_state = torch.as_strided(
                            region.view(torch.float32),
                            size=(num_physical_pages, mamba_state_elements),
                            stride=(page_stride_bytes // torch.float32.itemsize, 1),
                        )
                        mamba_state[1].fill_(3)
                        key_cpu = (torch.randn(num_tokens, num_kv_heads, head_dim, generator=generator) * 0.25).to(
                            dtype
                        )
                        value_cpu = (torch.randn(num_tokens, num_kv_heads, head_dim, generator=generator) * 0.25).to(
                            dtype
                        )
                        query_cpu = (torch.randn(1, num_query_heads, head_dim, generator=generator) * 0.25).to(dtype)
                        kernel_block_id = kernel_blocks_per_page - 1
                        slots = torch.arange(
                            kernel_block_id * kernel_block_size,
                            kernel_block_id * kernel_block_size + num_tokens,
                            dtype=torch.int32,
                        )

                        # 在 CPU 上按相同物理页更新黄金字节，逐字节检查填充和首尾保护区。
                        before_raw = raw.cpu()
                        expected_raw = before_raw.clone()
                        expected_region = expected_raw[region_offset_bytes : region_offset_bytes + region_bytes]
                        expected_key_cache, expected_value_cache = reshape_combined_attention_kv_cache(
                            expected_region, cache_shape, dtype, page_stride_bytes, kernel_blocks_per_page
                        )
                        expected_key_cache[kernel_block_id, :num_tokens] = key_cpu
                        expected_value_cache[kernel_block_id, :num_tokens] = value_cpu
                        torch_npu.npu_scatter_pa_kv_cache(
                            key=key_cpu.to("npu"),
                            value=value_cpu.to("npu"),
                            key_cache=key_cache,
                            value_cache=value_cache,
                            slot_mapping=slots.to("npu"),
                            cache_mode="Norm",
                        )
                        torch.npu.synchronize()

                        torch.testing.assert_close(raw.cpu(), expected_raw, atol=0, rtol=0)
                        torch.testing.assert_close(key_cache.cpu(), expected_key_cache, atol=0, rtol=0)
                        torch.testing.assert_close(value_cache.cpu(), expected_value_cache, atol=0, rtol=0)
                        torch.testing.assert_close(
                            mamba_state[1].cpu(), torch.full((mamba_state_elements,), 3.0), atol=0, rtol=0
                        )
                        second_page_start = region_offset_bytes + page_stride_bytes
                        torch.testing.assert_close(
                            raw[second_page_start : second_page_start + page_stride_bytes].cpu(),
                            before_raw[second_page_start : second_page_start + page_stride_bytes],
                            atol=0,
                            rtol=0,
                        )

                        # FIA 通过块表读取刚写入的子块，参考值使用量化后的输入做 FP32 计算。
                        key_golden = key_cpu.float().repeat_interleave(num_query_heads // num_kv_heads, dim=1)
                        value_golden = value_cpu.float().repeat_interleave(num_query_heads // num_kv_heads, dim=1)
                        scores = (
                            torch.matmul(
                                query_cpu.float().transpose(0, 1), key_golden.transpose(0, 1).transpose(-1, -2)
                            )
                            * head_dim**-0.5
                        )
                        expected_output = torch.matmul(scores.softmax(dim=-1), value_golden.transpose(0, 1)).transpose(
                            0, 1
                        )
                        attention_output, _ = torch_npu.npu_fused_infer_attention_score(
                            query=query_cpu.to("npu"),
                            key=key_cache.view(cache_shape[1], kernel_block_size, -1),
                            value=value_cache.view(cache_shape[1], kernel_block_size, -1),
                            block_table=torch.tensor([[kernel_block_id]], dtype=torch.int32, device="npu"),
                            actual_seq_lengths=[1],
                            actual_seq_lengths_kv=[num_tokens],
                            num_heads=num_query_heads,
                            num_key_value_heads=num_kv_heads,
                            input_layout="TND",
                            block_size=kernel_block_size,
                            scale=head_dim**-0.5,
                            sparse_mode=0,
                        )
                        torch.npu.synchronize()
                        tolerance = 0.005 if dtype == torch.float16 else 0.02
                        torch.testing.assert_close(
                            attention_output.cpu().float(), expected_output, atol=tolerance, rtol=tolerance
                        )
