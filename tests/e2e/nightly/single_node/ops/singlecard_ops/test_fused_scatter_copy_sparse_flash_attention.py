# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend correctness coverage for fused copy SFA with dynamic TND."""

from __future__ import annotations

import math
import unittest

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.utils import enable_custom_op  # noqa: E402

enable_custom_op()


BLOCK_SIZE = 128
TOPK = 2048
CKV_DIM = 512
KPE_DIM = 64
MISS_CAPACITY = 32768


def prefix_sum(values: list[int]) -> list[int]:
    total = 0
    result: list[int] = []
    for value in values:
        total += value
        result.append(total)
    return result


def logical_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    request: int,
    logical_slots: torch.Tensor,
) -> torch.Tensor:
    physical_blocks = block_table[request, logical_slots // BLOCK_SIZE].to(torch.int64)
    offsets = logical_slots % BLOCK_SIZE
    return cache[physical_blocks, offsets]


class TestFusedScatterCopySparseFlashAttentionDynamicTnd(unittest.TestCase):
    def run_case(
        self,
        query_counts: list[int],
        heads: int,
        *,
        first_fill: bool = False,
        cache_budgets: list[int] | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        torch.manual_seed(20260901 + heads + sum(query_counts))
        device = torch.device("npu:0")
        batch_size = len(query_counts)
        total_query_tokens = sum(query_counts)
        if cache_budgets is None:
            cache_budgets = [TOPK] * batch_size
        if len(cache_budgets) != batch_size:
            raise ValueError("cache budgets must match batch size")
        if any(budget < TOPK or budget % BLOCK_SIZE for budget in cache_budgets):
            raise ValueError("cache budgets must be block aligned and >= TopK")
        tail_tokens = 2
        max_logical_tokens = max(budget + tail_tokens + count for budget, count in zip(cache_budgets, query_counts))
        blocks_per_request = math.ceil(max_logical_tokens / BLOCK_SIZE)

        hbm_block_table_cpu = torch.arange(batch_size * blocks_per_request, dtype=torch.int32).view(
            batch_size, blocks_per_request
        )
        hbm_blocks = batch_size * blocks_per_request
        initial_hbm_kpe_cpu = torch.randn(hbm_blocks, BLOCK_SIZE, KPE_DIM, dtype=dtype)
        initial_hbm_ckv_cpu = torch.randn(hbm_blocks, BLOCK_SIZE, CKV_DIM, dtype=dtype)
        expected_hbm_kpe_cpu = initial_hbm_kpe_cpu.clone()
        expected_hbm_ckv_cpu = initial_hbm_ckv_cpu.clone()
        query_cpu = torch.randn(total_query_tokens, heads, CKV_DIM, dtype=dtype)
        query_rope_cpu = torch.randn(total_query_tokens, heads, KPE_DIM, dtype=dtype)

        sparse_slots_cpu = (
            torch.arange(TOPK, dtype=torch.int32).view(1, 1, TOPK).expand(total_query_tokens, 1, TOPK).contiguous()
        )
        topk_src_ids_cpu = torch.full_like(sparse_slots_cpu, -777) if first_fill else sparse_slots_cpu.clone()
        topk_miss_counts_cpu = torch.full((total_query_tokens,), TOPK if first_fill else 0, dtype=torch.int32)
        actual_q_cpu = torch.tensor(prefix_sum(query_counts), dtype=torch.int32)
        # The last route sees this full length; earlier routes hide their
        # request-local future speculative rows.
        actual_kv_cpu = torch.tensor(
            [budget + tail_tokens + count for budget, count in zip(cache_budgets, query_counts)],
            dtype=torch.int32,
        )
        cache_tokens_cpu = torch.tensor(cache_budgets, dtype=torch.int32)

        miss_src_ids_cpu = torch.full((batch_size, MISS_CAPACITY), -1, dtype=torch.int32)
        miss_dst_slots_cpu = torch.full_like(miss_src_ids_cpu, -1)
        miss_counts_cpu = torch.zeros(batch_size, dtype=torch.int32)
        dram_blocks_per_request = math.ceil(max(cache_budgets) / BLOCK_SIZE) if first_fill else 1
        dram_block_table_cpu = torch.arange(batch_size * dram_blocks_per_request, dtype=torch.int32).view(
            batch_size, dram_blocks_per_request
        )
        dram_blocks = batch_size * dram_blocks_per_request
        dram_kpe_cpu = torch.randn(dram_blocks, BLOCK_SIZE, KPE_DIM, dtype=dtype)
        dram_ckv_cpu = torch.randn(dram_blocks, BLOCK_SIZE, CKV_DIM, dtype=dtype)
        if first_fill:
            first_fill_budget = cache_budgets[0]
            miss_src_ids_cpu[0, :first_fill_budget] = torch.arange(first_fill_budget, dtype=torch.int32)
            miss_dst_slots_cpu[0, :first_fill_budget] = torch.arange(first_fill_budget, dtype=torch.int32)
            miss_counts_cpu[0] = first_fill_budget
            logical_slots = torch.arange(first_fill_budget, dtype=torch.int64)
            source_blocks = dram_block_table_cpu[0, logical_slots // BLOCK_SIZE].to(torch.int64)
            source_offsets = logical_slots % BLOCK_SIZE
            destination_blocks = hbm_block_table_cpu[0, logical_slots // BLOCK_SIZE].to(torch.int64)
            destination_offsets = logical_slots % BLOCK_SIZE
            expected_hbm_kpe_cpu[destination_blocks, destination_offsets] = dram_kpe_cpu[source_blocks, source_offsets]
            expected_hbm_ckv_cpu[destination_blocks, destination_offsets] = dram_ckv_cpu[source_blocks, source_offsets]

        scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)
        expected_rows: list[torch.Tensor] = []
        global_row = 0
        for request, query_count in enumerate(query_counts):
            cache_tokens = cache_budgets[request]
            for route in range(query_count):
                visible_kv = int(actual_kv_cpu[request]) - (query_count - 1 - route)
                logical_slots = torch.cat(
                    (
                        torch.arange(TOPK, dtype=torch.int64),
                        torch.arange(cache_tokens, visible_kv, dtype=torch.int64),
                    )
                )
                selected_ckv = logical_rows(
                    expected_hbm_ckv_cpu,
                    hbm_block_table_cpu,
                    request,
                    logical_slots,
                ).float()
                selected_kpe = logical_rows(
                    expected_hbm_kpe_cpu,
                    hbm_block_table_cpu,
                    request,
                    logical_slots,
                ).float()
                scores = (
                    query_cpu[global_row].float() @ selected_ckv.T + query_rope_cpu[global_row].float() @ selected_kpe.T
                ) * scale
                expected_rows.append(torch.softmax(scores, dim=-1) @ selected_ckv)
                global_row += 1
        expected = torch.stack(expected_rows)

        def to_npu(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device)

        query = to_npu(query_cpu)
        query_rope = to_npu(query_rope_cpu)
        hbm_kpe = to_npu(initial_hbm_kpe_cpu).view(hbm_blocks, BLOCK_SIZE, 1, KPE_DIM)
        hbm_ckv = to_npu(initial_hbm_ckv_cpu).view(hbm_blocks, BLOCK_SIZE, 1, CKV_DIM)
        actual_q = to_npu(actual_q_cpu)
        actual_kv = to_npu(actual_kv_cpu)
        cache_tokens_tensor = to_npu(cache_tokens_cpu)
        sparse_slots = to_npu(sparse_slots_cpu)
        topk_src_ids = to_npu(topk_src_ids_cpu)
        topk_miss_counts = to_npu(topk_miss_counts_cpu)
        miss_src_ids = to_npu(miss_src_ids_cpu)
        miss_dst_slots = to_npu(miss_dst_slots_cpu)
        miss_counts = to_npu(miss_counts_cpu)
        hbm_block_table = to_npu(hbm_block_table_cpu)
        dram_block_table = to_npu(dram_block_table_cpu)
        dram_kpe = to_npu(dram_kpe_cpu)
        dram_ckv = to_npu(dram_ckv_cpu)
        output = torch.empty_like(query)
        torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention.default(
            query_rope,
            query,
            actual_q,
            actual_kv,
            cache_tokens_tensor,
            sparse_slots,
            topk_src_ids,
            topk_miss_counts,
            miss_src_ids,
            miss_dst_slots,
            miss_counts,
            hbm_block_table,
            dram_block_table,
            hbm_kpe,
            hbm_ckv,
            dram_kpe,
            dram_ckv,
            scale,
            output,
        )
        torch.npu.synchronize()

        actual = output.cpu().float()
        max_abs = float((actual - expected).abs().max())
        self.assertLess(
            max_abs,
            0.08,
            msg=(f"dynamic TND mismatch: query_counts={query_counts}, heads={heads}, max_abs={max_abs}"),
        )
        torch.testing.assert_close(
            hbm_kpe.cpu().view_as(expected_hbm_kpe_cpu),
            expected_hbm_kpe_cpu,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            hbm_ckv.cpu().view_as(expected_hbm_ckv_cpu),
            expected_hbm_ckv_cpu,
            rtol=0,
            atol=0,
        )

        graph_hbm_kpe = to_npu(initial_hbm_kpe_cpu).view(hbm_blocks, BLOCK_SIZE, 1, KPE_DIM)
        graph_hbm_ckv = to_npu(initial_hbm_ckv_cpu).view(hbm_blocks, BLOCK_SIZE, 1, CKV_DIM)
        graph_topk_src_ids = topk_src_ids.clone()
        graph_topk_miss_counts = topk_miss_counts.clone()
        graph_miss_counts = miss_counts.clone()
        if first_fill:
            # Capture the steady branch, then switch the same graph buffers to
            # first-fill before replay. This proves routing is device-data
            # driven rather than frozen by host-side graph capture.
            graph_topk_src_ids.copy_(sparse_slots)
            graph_topk_miss_counts.zero_()
            graph_miss_counts.zero_()
        graph_output = torch.empty_like(query)
        graph = torch.npu.NPUGraph()
        pool = torch.npu.graph_pool_handle()
        with torch.npu.graph(graph, pool=pool):
            torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention.default(
                query_rope,
                query,
                actual_q,
                actual_kv,
                cache_tokens_tensor,
                sparse_slots,
                graph_topk_src_ids,
                graph_topk_miss_counts,
                miss_src_ids,
                miss_dst_slots,
                graph_miss_counts,
                hbm_block_table,
                dram_block_table,
                graph_hbm_kpe,
                graph_hbm_ckv,
                dram_kpe,
                dram_ckv,
                scale,
                graph_output,
            )
        torch.npu.synchronize()
        if first_fill:
            graph_hbm_kpe.copy_(to_npu(initial_hbm_kpe_cpu).view_as(graph_hbm_kpe))
            graph_hbm_ckv.copy_(to_npu(initial_hbm_ckv_cpu).view_as(graph_hbm_ckv))
            graph_topk_src_ids.copy_(topk_src_ids)
            graph_topk_miss_counts.copy_(topk_miss_counts)
            graph_miss_counts.copy_(miss_counts)
            torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        graph_actual = graph_output.cpu().float()
        graph_max_abs = float((graph_actual - expected).abs().max())
        self.assertLess(
            graph_max_abs,
            0.08,
            msg=(f"dynamic TND graph mismatch: query_counts={query_counts}, heads={heads}, max_abs={graph_max_abs}"),
        )
        torch.testing.assert_close(graph_output, output, rtol=0, atol=0)
        torch.testing.assert_close(
            graph_hbm_kpe.cpu().view_as(expected_hbm_kpe_cpu),
            expected_hbm_kpe_cpu,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            graph_hbm_ckv.cpu().view_as(expected_hbm_ckv_cpu),
            expected_hbm_ckv_cpu,
            rtol=0,
            atol=0,
        )

    def test_heterogeneous_mtp2_mtp3_mtp8_mtp15_n8(self) -> None:
        self.run_case(
            [3, 4, 9, 16],
            heads=8,
            cache_budgets=[2048, 2176, 2304, 2432],
        )

    def test_n128(self) -> None:
        self.run_case([1], heads=128)

    def test_fp16(self) -> None:
        self.run_case([1], heads=8, dtype=torch.float16)

    def test_intermediate_supported_head_counts(self) -> None:
        for heads in (16, 32, 64):
            with self.subTest(heads=heads):
                self.run_case([1], heads=heads)

    def test_heterogeneous_first_fill_graph(self) -> None:
        self.run_case(
            [3, 4, 9],
            heads=8,
            first_fill=True,
            cache_budgets=[2048, 2176, 2304],
        )


if __name__ == "__main__":
    unittest.main()
