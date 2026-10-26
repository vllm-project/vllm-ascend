# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Bundled GBSA dispatcher/Meta and paged sparse attention regression tests."""

import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

# CANN-generated direct-kernel libraries query the SoC during import.
# Initialize the available NPU first; Meta-only CPU checks need no device.
bootstrap_custom_op_env(include_vendor_lib=True)
if torch.npu.is_available():
    torch.npu.init()
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401


@pytest.mark.parametrize(
    "layout,shape,lse_shape",
    [("TND", (5, 8, 128), (5, 8, 1)), ("BNSD", (2, 8, 5, 128), (2, 8, 5, 1)), ("BSND", (2, 5, 8, 128), (2, 8, 5, 1))],
)
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gbsa_meta_outputs(layout, shape, lse_shape, return_lse, dtype):
    q = torch.empty(shape, device="meta", dtype=dtype)
    idx = torch.empty((1, 5, 4), device="meta", dtype=torch.int32)
    count = torch.empty((1, 5), device="meta", dtype=torch.int32)
    metadata = torch.ops._C_ascend.npu_generic_block_sparse_attention_metadata(idx, count, 8, 1, 128, [1, 128])
    assert metadata.shape == (1024,)
    assert metadata.dtype == torch.int32
    out, lse = torch.ops._C_ascend.npu_generic_block_sparse_attention(
        q, q, q, idx, count, [1, 128], layout_q=layout, return_softmax_lse=return_lse
    )
    assert out.shape == q.shape and out.dtype == dtype
    assert lse.shape == (lse_shape if return_lse else (0,))
    assert lse.dtype == torch.float32


def test_gbsa_meta_quantized_output_dtype():
    q = torch.empty((4, 8, 128), device="meta", dtype=torch.float8_e4m3fn)
    idx = torch.empty((1, 4, 4), device="meta", dtype=torch.int32)
    count = torch.empty((1, 4), device="meta", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="attention_out_dtype must be specified"):
        torch.ops._C_ascend.npu_generic_block_sparse_attention(q, q, q, idx, count, [1, 128], quant_mode=5)
    out, _ = torch.ops._C_ascend.npu_generic_block_sparse_attention(
        q, q, q, idx, count, [1, 128], quant_mode=5, attention_out_dtype=torch.bfloat16
    )
    assert out.dtype == torch.bfloat16


def test_gbsa_meta_symbolic_export():
    class Attention(torch.nn.Module):
        def forward(self, q, k, idx, count):
            return torch.ops._C_ascend.npu_generic_block_sparse_attention(q, k, k, idx, count, [1, 128])[0]

    args = (
        torch.empty(5, 8, 128),
        torch.empty(4, 128, 1, 128),
        torch.empty(1, 5, 4, dtype=torch.int32),
        torch.empty(1, 5, dtype=torch.int32),
    )
    tokens = torch.export.Dim("tokens", min=1, max=32)
    exported = torch.export.export(Attention(), args, dynamic_shapes=({0: tokens}, None, {1: tokens}, {1: tokens}))
    output = next(node for node in exported.graph.nodes if node.op == "output").args[0][0]
    assert str(output.meta["val"].shape[0]) != "5"


def _make_case(dtype, q_lens, kv_lens, mask_mode):
    block_size, num_heads, head_dim, table_width = 128, 8, 128, 4
    generator = torch.Generator().manual_seed(20261022)
    q_cpu = torch.randn(sum(q_lens), num_heads, head_dim, generator=generator).to(dtype)
    k_cpu = torch.randn(len(q_lens) * table_width, block_size, 1, head_dim, generator=generator).to(dtype)
    v_cpu = torch.randn(k_cpu.shape, generator=generator).to(dtype)
    table = torch.arange(k_cpu.shape[0], dtype=torch.int32).reshape(len(q_lens), table_width).flip(1).contiguous()
    idx = torch.zeros((1, sum(q_lens), table_width), dtype=torch.int32)
    count = torch.zeros((1, sum(q_lens)), dtype=torch.int32)
    expected, prefix = [], [0]
    for batch, (qlen, kvlen) in enumerate(zip(q_lens, kv_lens)):
        for local_q in range(qlen):
            token = prefix[-1] + local_q
            visible = kvlen - qlen + local_q + 1 if mask_mode == 1 else kvlen
            # Keep nonadjacent logical blocks and the final block to exercise sparsity.
            blocks = list(range(0, math.ceil(visible / block_size), 2))
            last_block = (visible - 1) // block_size
            if last_block not in blocks:
                blocks.append(last_block)
            idx[0, token, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            count[0, token] = len(blocks)
            logical_tokens = torch.cat(
                [torch.arange(b * block_size, min((b + 1) * block_size, visible)) for b in blocks]
            )
            physical = table[batch, logical_tokens // block_size].long()
            offsets = logical_tokens % block_size
            keys = k_cpu[physical, offsets, 0].float()
            values = v_cpu[physical, offsets, 0].float()
            scores = q_cpu[token].float() @ keys.T / math.sqrt(head_dim)
            expected.append(torch.softmax(scores, dim=-1) @ values)
        prefix.append(prefix[-1] + qlen)
    inputs = [
        q_cpu,
        k_cpu,
        v_cpu,
        idx,
        count,
        table,
        torch.tensor(prefix, dtype=torch.int64),
        torch.tensor(kv_lens, dtype=torch.int32),
    ]
    return [tensor.npu() for tensor in inputs], torch.stack(expected)


def _run_case(inputs, max_q, mask_mode):
    q, k, v, idx, count, table, cu_q, used_kv = inputs
    metadata = torch.ops._C_ascend.npu_generic_block_sparse_attention_metadata(
        idx,
        count,
        q.shape[1],
        1,
        q.shape[2],
        [1, 128],
        cu_seqlens_q=cu_q,
        seqused_kv=used_kv,
        max_seqlen_q=max_q,
        mask_mode=mask_mode,
        softmax_precision=0,
    )
    out, _ = torch.ops._C_ascend.npu_generic_block_sparse_attention(
        q,
        k,
        v,
        idx,
        count,
        [1, 128],
        metadata=metadata,
        cu_seqlens_q=cu_q,
        seqused_kv=used_kv,
        block_table=table,
        layout_q="TND",
        layout_kv="PA_BBND",
        softmax_scale=1 / math.sqrt(q.shape[-1]),
        mask_mode=mask_mode,
        softmax_precision=0,
    )
    return out


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires a free Ascend NPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "q_lens,kv_lens", [((1, 1), (257, 383)), ((4, 4), (257, 383)), ((1, 4), (129, 257)), ((4, 0, 1), (257, 0, 129))]
)
@pytest.mark.parametrize("mask_mode", [0, 1])
def test_gbsa_eager_against_dense_reference(dtype, q_lens, kv_lens, mask_mode):
    inputs, expected = _make_case(dtype, q_lens, kv_lens, mask_mode)
    out = _run_case(inputs, max(q_lens), mask_mode)
    torch.npu.synchronize()
    torch.testing.assert_close(out.float().cpu(), expected, atol=1e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires a free Ascend NPU")
@pytest.mark.parametrize("q_len", [1, 4])
def test_gbsa_graph_replay_changed_inputs(q_len):
    inputs, _ = _make_case(torch.bfloat16, (q_len, q_len), (257, 383), 1)
    for _ in range(3):
        _run_case(inputs, q_len, 1)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = _run_case(inputs, q_len, 1)
    for lengths in [(258, 384), (259, 385)]:
        inputs[-1].copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
        inputs[0].mul_(0.75)
        eager = _run_case(inputs, q_len, 1)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out, eager, atol=1e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires a free Ascend NPU")
@pytest.mark.parametrize("q_len", [1, 4])
def test_gbsa_model_metadata_reuse(q_len):
    # Match plugin startup order when running without the full E2E conftest.
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _run_generic_block_sparse_attention

    inputs, expected = _make_case(torch.bfloat16, (q_len, q_len), (257, 383), 1)
    q, k, v, idx, count, table, cu_q, used_kv = inputs
    cache = {}
    kwargs = dict(num_kv_heads=1, sm_scale=1 / math.sqrt(q.shape[-1]), block_size=128, max_query_len=q_len)
    first = _run_generic_block_sparse_attention(
        q, k, v, idx, table, count, cu_q, used_kv, gbsa_metadata_cache=cache, **kwargs
    )
    torch.testing.assert_close(first.float().cpu(), expected, atol=1e-2, rtol=2e-2)
    assert len(cache) == 1
    schedule = next(iter(cache.values()))
    next_q = q * 0.75
    cached = _run_generic_block_sparse_attention(
        next_q, k, v, idx, table, count, cu_q, used_kv, gbsa_metadata_cache=cache, **kwargs
    )
    fresh = _run_generic_block_sparse_attention(next_q, k, v, idx, table, count, cu_q, used_kv, **kwargs)
    torch.npu.synchronize()
    torch.testing.assert_close(cached, fresh, atol=1e-2, rtol=2e-2)
    assert next(iter(cache.values())) is schedule
