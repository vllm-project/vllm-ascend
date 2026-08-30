# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.ops.fia_v2_sink import (
    FIA_V2_SINK_METADATA_SIZE,
    build_fia_v2_sink_metadata,
    fused_infer_attention_score_v2_sink,
)
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)

BLOCK_SIZE = 128
NUM_QUERY_HEADS = 4
NUM_KV_HEADS = 1


def _paged_block_table(batch_size: int, blocks_per_request: int) -> torch.Tensor:
    return torch.arange(
        batch_size * blocks_per_request,
        dtype=torch.int32,
        device="npu",
    ).view(batch_size, blocks_per_request)


def _build_gqa_metadata(
    actual_seq_qlen: torch.Tensor,
    actual_seq_kvlen: torch.Tensor,
    *,
    sparse_mode: int,
    output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    return build_fia_v2_sink_metadata(
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        num_query_heads=NUM_QUERY_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim_qk=128,
        head_dim_v=128,
        input_layout="TND",
        input_layout_kv="BnBsH",
        sparse_mode=sparse_mode,
        block_size=BLOCK_SIZE,
        output_buffer=output_buffer,
    )


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size,kv_len", [(2, 255), (4, 2048)])
def test_fia_v2_sink_matches_fia_for_paged_gqa(dtype: torch.dtype, batch_size: int, kv_len: int):
    torch.manual_seed(20260830)
    query_tokens_per_request = 8
    query_tokens = batch_size * query_tokens_per_request
    head_dim = 128
    blocks_per_request = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table = _paged_block_table(batch_size, blocks_per_request)
    key = torch.randn(
        batch_size * blocks_per_request,
        BLOCK_SIZE,
        NUM_KV_HEADS * head_dim,
        dtype=dtype,
        device="npu",
    )
    value = torch.randn_like(key)
    query = torch.randn(
        query_tokens,
        NUM_QUERY_HEADS,
        head_dim,
        dtype=dtype,
        device="npu",
    )
    actual_seq_qlen = torch.arange(
        query_tokens_per_request,
        query_tokens + 1,
        query_tokens_per_request,
        dtype=torch.int64,
        device="npu",
    )
    actual_seq_kvlen = torch.full((batch_size,), kv_len, dtype=torch.int64, device="npu")
    scale = head_dim**-0.5

    expected, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        block_table=block_table,
        input_layout="TND",
        block_size=BLOCK_SIZE,
        actual_seq_lengths=actual_seq_qlen.cpu().tolist(),
        actual_seq_lengths_kv=actual_seq_kvlen.cpu().tolist(),
        num_key_value_heads=NUM_KV_HEADS,
        num_heads=NUM_QUERY_HEADS,
        scale=scale,
        sparse_mode=0,
    )
    actual = fused_infer_attention_score_v2_sink(
        query,
        key,
        value,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        block_table=block_table,
        metadata=_build_gqa_metadata(actual_seq_qlen, actual_seq_kvlen, sparse_mode=0),
        num_query_heads=NUM_QUERY_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        softmax_scale=scale,
        input_layout="TND",
        sparse_mode=0,
        block_size=BLOCK_SIZE,
    )

    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


@torch.inference_mode()
def test_fia_v2_sink_replays_with_live_sequence_lengths():
    """Replay the multi-layer causal FIA shape used by K3 DSpark."""
    torch.manual_seed(20260830)
    batch_size = 4
    query_tokens = 32
    head_dim = 128
    blocks_per_request = 16
    block_table = _paged_block_table(batch_size, blocks_per_request)
    key = torch.randn(
        batch_size * blocks_per_request,
        BLOCK_SIZE,
        NUM_KV_HEADS * head_dim,
        dtype=torch.bfloat16,
        device="npu",
    )
    value = torch.randn_like(key)
    query = torch.randn(
        query_tokens,
        NUM_QUERY_HEADS,
        head_dim,
        dtype=torch.bfloat16,
        device="npu",
    )
    actual_seq_qlen = torch.tensor([8, 16, 24, 32], dtype=torch.int64, device="npu")
    # Full-graph capture uses the uniform decode width as its synthetic KV
    # length; replay then receives the live cache lengths.
    actual_seq_kvlen = torch.full((batch_size,), 8, dtype=torch.int64, device="npu")
    atten_mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).npu()
    scale = head_dim**-0.5
    metadata = torch.zeros(FIA_V2_SINK_METADATA_SIZE, dtype=torch.int32, device="npu")
    _build_gqa_metadata(actual_seq_qlen, actual_seq_kvlen, sparse_mode=3, output_buffer=metadata)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        for _ in range(16):
            actual = fused_infer_attention_score_v2_sink(
                query,
                key,
                value,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                block_table=block_table,
                metadata=metadata,
                num_query_heads=NUM_QUERY_HEADS,
                num_key_value_heads=NUM_KV_HEADS,
                softmax_scale=scale,
                input_layout="TND",
                sparse_mode=3,
                block_size=BLOCK_SIZE,
                atten_mask=atten_mask,
            )

    replay_qlens = ([8, 16, 24, 32], [5, 13, 22, 32])
    replay_kvlens = ([2048, 2048, 2048, 2048], [1920, 1984, 2016, 2048])
    for qlens, kvlens in zip(replay_qlens, replay_kvlens):
        actual_seq_qlen.copy_(torch.tensor(qlens, dtype=torch.int64, device="npu"))
        actual_seq_kvlen.copy_(torch.tensor(kvlens, dtype=torch.int64, device="npu"))
        _build_gqa_metadata(actual_seq_qlen, actual_seq_kvlen, sparse_mode=3, output_buffer=metadata)
        graph.replay()
        torch.npu.synchronize()

        expected, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            block_table=block_table,
            input_layout="TND",
            block_size=BLOCK_SIZE,
            actual_seq_lengths=qlens,
            actual_seq_lengths_kv=kvlens,
            num_key_value_heads=NUM_KV_HEADS,
            num_heads=NUM_QUERY_HEADS,
            scale=scale,
            sparse_mode=3,
            atten_mask=atten_mask,
        )
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fia_v2_sink_matches_fia_for_paged_mla(dtype: torch.dtype):
    torch.manual_seed(20260830)
    query_tokens = 4
    kv_lora_rank = 512
    rope_head_dim = 64
    blocks_per_request = 2
    block_table = _paged_block_table(2, blocks_per_request)
    key = torch.randn(
        2 * blocks_per_request,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        kv_lora_rank,
        dtype=dtype,
        device="npu",
    )
    key_rope = torch.randn(
        2 * blocks_per_request,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        rope_head_dim,
        dtype=dtype,
        device="npu",
    )
    query = torch.randn(
        query_tokens,
        NUM_QUERY_HEADS,
        kv_lora_rank,
        dtype=dtype,
        device="npu",
    )
    query_rope = torch.randn(
        query_tokens,
        NUM_QUERY_HEADS,
        rope_head_dim,
        dtype=dtype,
        device="npu",
    )
    actual_seq_qlen = torch.tensor([2, 4], dtype=torch.int64, device="npu")
    actual_seq_kvlen = torch.tensor([192, 255], dtype=torch.int64, device="npu")
    scale = (kv_lora_rank + rope_head_dim) ** -0.5

    common_kwargs = {
        "query_rope": query_rope,
        "key_rope": key_rope,
        "num_query_heads": NUM_QUERY_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "input_layout": "TND_NTD",
        "sparse_mode": 0,
        "softmax_scale": scale,
        "block_table": block_table,
        "block_size": BLOCK_SIZE,
        "actual_seq_qlen": actual_seq_qlen.cpu().tolist(),
        "actual_seq_kvlen": actual_seq_kvlen.cpu().tolist(),
    }
    expected, _ = torch_npu.npu_fused_infer_attention_score_v2(query, key, key, **common_kwargs)
    actual = fused_infer_attention_score_v2_sink(
        query,
        key,
        key,
        query_rope=query_rope,
        key_rope=key_rope,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        block_table=block_table,
        metadata=build_fia_v2_sink_metadata(
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            num_query_heads=NUM_QUERY_HEADS,
            num_key_value_heads=NUM_KV_HEADS,
            head_dim_qk=kv_lora_rank,
            head_dim_v=kv_lora_rank,
            input_layout="TND_NTD",
            input_layout_kv="BnNBsD",
            sparse_mode=0,
            block_size=BLOCK_SIZE,
            rope_head_dim=rope_head_dim,
        ),
        num_query_heads=NUM_QUERY_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        softmax_scale=scale,
        input_layout="TND_NTD",
        sparse_mode=0,
        block_size=BLOCK_SIZE,
    )

    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


@torch.inference_mode()
def test_fia_v2_sink_handles_consecutive_mla_batch_shapes():
    """Do not reuse a 16-request descriptor for the following 14-request call."""
    torch.manual_seed(20260830)
    max_batch_size = 16
    tokens_per_request = 8
    kv_lora_rank = 512
    rope_head_dim = 64
    blocks_per_request = 2
    block_table = _paged_block_table(max_batch_size, blocks_per_request)
    key = torch.randn(
        max_batch_size * blocks_per_request,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        kv_lora_rank,
        dtype=torch.bfloat16,
        device="npu",
    )
    key_rope = torch.randn(
        max_batch_size * blocks_per_request,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        rope_head_dim,
        dtype=torch.bfloat16,
        device="npu",
    )
    query = torch.randn(
        max_batch_size * tokens_per_request,
        NUM_QUERY_HEADS,
        kv_lora_rank,
        dtype=torch.bfloat16,
        device="npu",
    )
    query_rope = torch.randn(
        max_batch_size * tokens_per_request,
        NUM_QUERY_HEADS,
        rope_head_dim,
        dtype=torch.bfloat16,
        device="npu",
    )
    actual_seq_qlen = torch.arange(
        tokens_per_request,
        max_batch_size * tokens_per_request + 1,
        tokens_per_request,
        dtype=torch.int64,
        device="npu",
    )
    actual_seq_kvlen = torch.full(
        (max_batch_size,),
        BLOCK_SIZE,
        dtype=torch.int64,
        device="npu",
    )
    scale = (kv_lora_rank + rope_head_dim) ** -0.5

    for batch_size in (16, 14):
        num_tokens = batch_size * tokens_per_request
        qlens = actual_seq_qlen[:batch_size].clone()
        kvlens = actual_seq_kvlen[:batch_size].clone()
        blocks = block_table[:batch_size]
        expected, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query[:num_tokens],
            key,
            key,
            query_rope=query_rope[:num_tokens],
            key_rope=key_rope,
            num_query_heads=NUM_QUERY_HEADS,
            num_key_value_heads=NUM_KV_HEADS,
            input_layout="TND_NTD",
            sparse_mode=0,
            softmax_scale=scale,
            block_table=blocks,
            block_size=BLOCK_SIZE,
            actual_seq_qlen=qlens.cpu().tolist(),
            actual_seq_kvlen=kvlens.cpu().tolist(),
        )
        actual = fused_infer_attention_score_v2_sink(
            query[:num_tokens],
            key,
            key,
            query_rope=query_rope[:num_tokens],
            key_rope=key_rope,
            actual_seq_qlen=qlens,
            actual_seq_kvlen=kvlens,
            block_table=blocks,
            metadata=build_fia_v2_sink_metadata(
                actual_seq_qlen=qlens,
                actual_seq_kvlen=kvlens,
                num_query_heads=NUM_QUERY_HEADS,
                num_key_value_heads=NUM_KV_HEADS,
                head_dim_qk=kv_lora_rank,
                head_dim_v=kv_lora_rank,
                input_layout="TND_NTD",
                input_layout_kv="BnNBsD",
                sparse_mode=0,
                block_size=BLOCK_SIZE,
                rope_head_dim=rope_head_dim,
            ),
            num_query_heads=NUM_QUERY_HEADS,
            num_key_value_heads=NUM_KV_HEADS,
            softmax_scale=scale,
            input_layout="TND_NTD",
            sparse_mode=0,
            block_size=BLOCK_SIZE,
        )
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
