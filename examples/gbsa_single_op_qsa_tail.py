import argparse
import math

import torch

import vllm_ascend.vllm_ascend_C  # noqa: F401
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.ops.gbsa import (
    generic_block_sparse_attention,
    generic_block_sparse_attention_metadata,
)


PAGE_SIZE = 128
SPARSE_BLOCK_SIZE = 4
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 128


def reference_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    logical_tokens = torch.arange(sequence_length, device=key_cache.device)
    logical_pages = torch.div(logical_tokens, PAGE_SIZE, rounding_mode="floor")
    page_offsets = logical_tokens.remainder(PAGE_SIZE)
    physical_pages = block_table[0, logical_pages]
    keys = key_cache[physical_pages, page_offsets].float()
    values = value_cache[physical_pages, page_offsets].float()
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    keys = keys.repeat_interleave(group_size, dim=1)
    values = values.repeat_interleave(group_size, dim=1)
    scores = torch.einsum("thd,shd->ths", query.float(), keys)
    probabilities = (scores / math.sqrt(HEAD_DIM)).softmax(dim=-1)
    return torch.einsum("ths,shd->thd", probabilities, values)


def run_case(dtype: torch.dtype, tail_tokens: int) -> None:
    sequence_length = 8 + tail_tokens
    explicit_blocks = sequence_length // SPARSE_BLOCK_SIZE

    generator = torch.Generator(device="cpu").manual_seed(20261009 + tail_tokens)
    query = torch.randn((1, NUM_Q_HEADS, HEAD_DIM), dtype=dtype, generator=generator).npu()
    host_key = torch.randn((3, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=dtype, generator=generator)
    host_value = torch.randn(host_key.shape, dtype=dtype, generator=generator)
    key_cache = host_key.npu()
    value_cache = host_value.npu()

    block_table = torch.tensor([[2]], dtype=torch.int32, device="npu")
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int64, device="npu")
    seqused_kv = torch.tensor([sequence_length], dtype=torch.int32, device="npu")
    sparse_block_idx = torch.arange(explicit_blocks, dtype=torch.int32, device="npu")
    sparse_block_idx = sparse_block_idx.view(1, 1, -1).expand(NUM_KV_HEADS, 1, -1).contiguous()
    sparse_block_count = torch.full(
        (NUM_KV_HEADS, 1), explicit_blocks, dtype=torch.int32, device="npu"
    )

    common = dict(
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=4,
        mask_mode=1,
        quant_mode=0,
        softmax_precision=0,
        residual_block_mode=1,
        is_consistent_topk=True,
    )
    metadata = generic_block_sparse_attention_metadata(
        sparse_block_idx,
        sparse_block_count,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        [1, SPARSE_BLOCK_SIZE],
        max_seqlen_q=1,
        max_seqlen_kv=sequence_length,
        **common,
    )
    actual, _ = generic_block_sparse_attention(
        query,
        key_cache,
        value_cache,
        sparse_block_idx,
        sparse_block_count,
        [1, SPARSE_BLOCK_SIZE],
        metadata=metadata,
        block_table=block_table,
        softmax_scale=HEAD_DIM**-0.5,
        **common,
    )
    expected = reference_attention(query, key_cache, value_cache, block_table, sequence_length)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    max_abs = (actual.float() - expected).abs().max().item()
    print(
        f"PASS dtype={dtype} seq={sequence_length} tail={tail_tokens} "
        f"explicit_blocks={explicit_blocks} max_abs={max_abs:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "all"), default="all")
    args = parser.parse_args()
    dtypes = (torch.float16, torch.bfloat16) if args.dtype == "all" else (getattr(torch, args.dtype),)
    for dtype in dtypes:
        for tail_tokens in range(4):
            run_case(dtype, tail_tokens)


if __name__ == "__main__":
    main()
