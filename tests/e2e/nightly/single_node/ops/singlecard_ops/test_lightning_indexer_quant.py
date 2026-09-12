# SPDX-License-Identifier: Apache-2.0

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

CHUNK_TOKENS = 1024
LOCAL_TOKENS = 128
PREFIX_TOKENS = 2048
FULL_KEY_LEN = PREFIX_TOKENS + CHUNK_TOKENS
NUM_HEADS = 32
HEAD_DIM = 128
BLOCK_SIZE = 128
SPARSE_COUNT = 2048


def _run_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    block_table: torch.Tensor,
    key_len: int,
) -> torch.Tensor:
    query_len = query.shape[0]
    return torch.ops._C_ascend.npu_lightning_indexer_quant(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=torch.tensor(
            [query_len], dtype=torch.int32, device=query.device
        ),
        actual_seq_lengths_key=torch.tensor(
            [key_len], dtype=torch.int32, device=query.device
        ),
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
    )


def test_equal_score_topk_is_invariant_to_query_slicing() -> None:
    """TopK selection must not depend on how the Query rows are tiled."""
    torch.manual_seed(20260826)
    bootstrap_custom_op_env()
    torch.npu.set_device(0)
    device = torch.device("npu:0")

    query = torch.randint(
        -64,
        64,
        (CHUNK_TOKENS, NUM_HEADS, HEAD_DIM),
        dtype=torch.int8,
        device=device,
    )
    key = torch.zeros(
        FULL_KEY_LEN // BLOCK_SIZE,
        BLOCK_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.int8,
        device=device,
    )
    # Keep only a small non-zero prefix. The remaining valid keys have exactly
    # equal zero scores, so sparse_count cuts through a large tie group.
    key[:3].random_(-64, 64)
    query_scale = (
        torch.rand(CHUNK_TOKENS, NUM_HEADS, device=device) * 0.018 + 0.002
    ).half()
    key_scale = (
        torch.rand(FULL_KEY_LEN // BLOCK_SIZE, BLOCK_SIZE, 1, device=device)
        * 0.018
        + 0.002
    ).half()
    weights = (torch.rand(CHUNK_TOKENS, NUM_HEADS, device=device) * 0.05).half()
    block_table = torch.arange(
        FULL_KEY_LEN // BLOCK_SIZE, dtype=torch.int32, device=device
    ).view(1, -1)

    full_topk = _run_indexer(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        block_table,
        FULL_KEY_LEN,
    )

    for rank in range(CHUNK_TOKENS // LOCAL_TOKENS):
        start = rank * LOCAL_TOKENS
        end = start + LOCAL_TOKENS
        local_topk = _run_indexer(
            query[start:end],
            key,
            weights[start:end],
            query_scale[start:end],
            key_scale,
            block_table,
            PREFIX_TOKENS + end,
        )
        torch.testing.assert_close(
            local_topk,
            full_topk[start:end],
            rtol=0,
            atol=0,
            msg=lambda msg, rank=rank: f"rank={rank}: {msg}",
        )
