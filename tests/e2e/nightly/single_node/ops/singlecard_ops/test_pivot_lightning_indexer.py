# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

BLOCK_SIZE = 128
HEAD_SIZE = 128
INDEXER_HEADS = 32
KV_LENGTH = 65536
SPARSE_COUNT = 2048


def _call(op, query, key, weights, query_lengths, key_lengths, block_table):
    result = op(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=query_lengths,
        actual_seq_lengths_key=key_lengths,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
    )
    return result[0] if isinstance(result, tuple) else result


def _require_ops(*names):
    if not hasattr(torch.ops, "_C_ascend") or any(not hasattr(torch.ops._C_ascend, name) for name in names):
        pytest.skip(f"requires the {', '.join(names)} custom operators")
    soc_version = torch_npu.npu.get_soc_version()
    if not 220 <= soc_version <= 225:
        pytest.skip("PIVOT reuse requires an arch22 A2/A3 NPU")


@torch.inference_mode()
def test_pivot_lightning_indexer_multi_batch_matches_independent_requests():
    _require_ops("npu_pivot_lightning_indexer")

    torch.manual_seed(20260924)
    qlen, batch = 4, 2
    blocks_per_request = KV_LENGTH // BLOCK_SIZE
    query = torch.randn(
        qlen * batch,
        INDEXER_HEADS,
        HEAD_SIZE,
        dtype=torch.bfloat16,
        device="npu",
    )
    key = torch.randn(
        blocks_per_request * batch,
        BLOCK_SIZE,
        1,
        HEAD_SIZE,
        dtype=torch.bfloat16,
        device="npu",
    )
    weights = torch.rand(qlen * batch, INDEXER_HEADS, dtype=torch.bfloat16, device="npu")
    query_lengths = torch.tensor([qlen, qlen * batch], dtype=torch.int32, device="npu")
    key_lengths = torch.tensor([KV_LENGTH, KV_LENGTH], dtype=torch.int32, device="npu")
    block_table = torch.arange(blocks_per_request * batch, dtype=torch.int32, device="npu").reshape(
        batch, blocks_per_request
    )

    op = torch.ops._C_ascend.npu_pivot_lightning_indexer
    batched = _call(op, query, key, weights, query_lengths, key_lengths, block_table)
    independent = []
    for request in range(batch):
        start = request * qlen
        independent.append(
            _call(
                op,
                query[start : start + qlen],
                key,
                weights[start : start + qlen],
                torch.tensor([qlen], dtype=torch.int32, device="npu"),
                key_lengths[request : request + 1],
                block_table[request : request + 1],
            )
        )
    torch.testing.assert_close(batched, torch.cat(independent), rtol=0, atol=0)

    gc.collect()
    torch.npu.empty_cache()


@torch.inference_mode()
def test_pivot_lightning_indexer_fallback_matches_baseline():
    _require_ops("npu_lightning_indexer", "npu_pivot_lightning_indexer")

    torch.manual_seed(20260924)
    qlen, batch = 1, 2
    blocks_per_request = KV_LENGTH // BLOCK_SIZE
    query = torch.randn(
        qlen * batch,
        INDEXER_HEADS,
        HEAD_SIZE,
        dtype=torch.bfloat16,
        device="npu",
    )
    key = torch.randn(
        blocks_per_request * batch,
        BLOCK_SIZE,
        1,
        HEAD_SIZE,
        dtype=torch.bfloat16,
        device="npu",
    )
    weights = torch.rand(qlen * batch, INDEXER_HEADS, dtype=torch.bfloat16, device="npu")
    query_lengths = torch.tensor([qlen, qlen * batch], dtype=torch.int32, device="npu")
    key_lengths = torch.tensor([KV_LENGTH, KV_LENGTH], dtype=torch.int32, device="npu")
    block_table = torch.arange(blocks_per_request * batch, dtype=torch.int32, device="npu").reshape(
        batch, blocks_per_request
    )

    baseline = _call(
        torch.ops._C_ascend.npu_lightning_indexer,
        query,
        key,
        weights,
        query_lengths,
        key_lengths,
        block_table,
    )
    fallback = _call(
        torch.ops._C_ascend.npu_pivot_lightning_indexer,
        query,
        key,
        weights,
        query_lengths,
        key_lengths,
        block_table,
    )
    torch.testing.assert_close(fallback, baseline, rtol=0, atol=0)

    gc.collect()
    torch.npu.empty_cache()
