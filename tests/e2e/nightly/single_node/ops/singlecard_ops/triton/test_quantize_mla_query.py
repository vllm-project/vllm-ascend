# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.ops.triton.mla_query_rope import (
    quantize_mla_query,
    quantize_mla_query_and_kv,
    scale_mla_query_rope,
)
from vllm_ascend.ops.triton.quantize_mla_kv import quantize_mla_kv


@pytest.mark.parametrize("tokens,heads", [(3, 12), (32, 12), (64, 96)])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_quantize_mla_query_graph(tokens, heads, strided):
    torch.manual_seed(813)
    if strided:
        backing = torch.randn((tokens, heads * 2, 640), dtype=torch.bfloat16, device="npu")
        source = backing[:, ::2, :576]
        query, rope = source[..., :512], source[..., 512:]
    else:
        query = torch.randn((tokens, heads, 512), dtype=torch.bfloat16, device="npu")
        rope = torch.randn((tokens, heads, 64), dtype=torch.bfloat16, device="npu")
    query[0, 0].zero_()
    kv_scale = torch.tensor([0.02], dtype=torch.float32, device="npu")

    def run():
        return quantize_mla_query(query, rope, kv_scale)

    def check(actual):
        torch.npu.synchronize()
        quantized, scale = torch_npu.npu_dynamic_quant(query.contiguous(), dst_type=torch.float8_e4m3fn)
        scaled_rope = scale_mla_query_rope(rope.contiguous(), scale, kv_scale)
        torch.npu.synchronize()
        torch.testing.assert_close(actual[0].float(), quantized.float(), rtol=0, atol=0)
        torch.testing.assert_close(actual[1], scaled_rope, rtol=0, atol=0)
        torch.testing.assert_close(actual[2], scale, rtol=0, atol=0)
        assert torch.isfinite(actual[1]).all()

    query_before, rope_before = query.clone(), rope.clone()
    check(run())
    torch.testing.assert_close(query, query_before, rtol=0, atol=0)
    torch.testing.assert_close(rope, rope_before, rtol=0, atol=0)
    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    graph.replay()
    check(captured)
    query.mul_(2)
    rope.mul_(-0.5)
    kv_scale.fill_(0.1)
    graph.replay()
    check(captured)
    query.zero_()
    graph.replay()
    check(captured)


@torch.inference_mode()
def test_quantize_mla_query_native_quantization():
    query = torch.randn((32, 12, 512), dtype=torch.bfloat16, device="npu")
    rope = torch.randn((32, 12, 64), dtype=torch.bfloat16, device="npu")
    kv_scale = torch.tensor([0.02], dtype=torch.float32, device="npu")
    quantized, scaled_rope, scale = quantize_mla_query(query, rope, kv_scale)
    reference_q, reference_scale = torch_npu.npu_dynamic_quant(query, dst_type=torch.float8_e4m3fn)
    reference_rope = scale_mla_query_rope(rope, reference_scale, kv_scale)
    torch.npu.synchronize()
    torch.testing.assert_close(quantized.float(), reference_q.float(), rtol=0, atol=0)
    torch.testing.assert_close(scaled_rope, reference_rope, rtol=0, atol=0)
    torch.testing.assert_close(scale, reference_scale, rtol=0, atol=0)


@pytest.mark.parametrize("tokens,heads", [(3, 12), (64, 96)])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_quantize_query_and_owned_kv_graph(tokens, heads, paged, strided):
    torch.manual_seed(16468)
    source_q = torch.randn((tokens, heads, 580), device="npu", dtype=torch.bfloat16)
    query, query_rope = source_q[..., :512], source_q[..., 516:]
    if not strided:
        query, query_rope = query.contiguous(), query_rope.contiguous()
    query[0, 0].zero_()
    scale = torch.tensor([0.03125], device="npu", dtype=torch.float32)
    reciprocal = scale.reciprocal()
    if paged:
        source = torch.randn((7, 128, 1, 600), device="npu", dtype=torch.bfloat16)[::2]
        latent, rope = source[..., :512], source[..., 536:]
        source_slots = torch.arange(tokens, dtype=torch.int64) * 5 + 17
        source_slots[0] = -1
        source_slots = source_slots.npu()
    else:
        source = torch.randn((tokens, 600), device="npu", dtype=torch.bfloat16)
        latent, rope, source_slots = source[..., :512], source[..., 536:], None
    # Independent first-axis strides and untouched neighboring pages.
    key_ref = torch.full((9, 128, 1, 512), 2.0, device="npu", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    rope_ref = torch.full((9, 128, 1, 64), 3.0, device="npu", dtype=torch.bfloat16)
    key_got, rope_got = key_ref.clone(), rope_ref.clone()
    slots = torch.arange(tokens, dtype=torch.int64) * 7 + 1
    slots[torch.arange(tokens) % 8 != 0] = -1
    slots = slots.npu()

    def run():
        return quantize_mla_query_and_kv(
            query,
            query_rope,
            scale,
            latent,
            rope,
            key_got[::2],
            rope_got[::2],
            slots,
            reciprocal,
            source_slots=source_slots,
        )

    def bits(t):
        return t.cpu().contiguous().view(torch.uint8)

    def check(actual):
        quantize_mla_kv(
            latent,
            rope,
            key_ref[::2],
            rope_ref[::2],
            slots,
            reciprocal,
            source_slots=source_slots,
        )
        expected = quantize_mla_query(query, query_rope, scale)
        torch.npu.synchronize()
        for result, reference in zip((*actual, key_got, rope_got), (*expected, key_ref, rope_ref)):
            assert torch.equal(bits(result), bits(reference))

    check(run())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    graph.replay()
    check(captured)
    query.mul_(2)
    query_rope.mul_(-0.5)
    scale.fill_(0.02)
    reciprocal.copy_(scale.reciprocal())
    graph.replay()
    check(captured)


@pytest.mark.parametrize("rank", [0, 3, 7])
@torch.inference_mode()
def test_quantize_query_kv_local_bf16_graph(rank):
    torch.manual_seed(16468)
    query = torch.randn((64, 96, 512), dtype=torch.bfloat16, device="npu")
    query_rope = torch.randn((64, 96, 64), dtype=torch.bfloat16, device="npu")
    query[0, rank * 12].zero_()
    scale = torch.tensor([0.03125], dtype=torch.float32, device="npu")
    reciprocal = scale.reciprocal()
    current = torch.randn((1, 128, 1, 576), dtype=torch.bfloat16, device="npu")
    source_slots = torch.arange(64, dtype=torch.int64, device="npu")
    slots = source_slots.clone()
    slots[slots % 8 != 0] = -1
    key_ref = torch.full((2, 128, 1, 512), 2.0, dtype=torch.bfloat16, device="npu").to(torch.float8_e4m3fn)
    rope_ref = torch.full((2, 128, 1, 64), 3.0, dtype=torch.bfloat16, device="npu")
    key_got, rope_got = key_ref.clone(), rope_ref.clone()
    local_query = torch.empty((64, 12, 576), dtype=torch.bfloat16, device="npu")

    def run(with_local):
        return quantize_mla_query_and_kv(
            query,
            query_rope,
            scale,
            current[..., :512],
            current[..., 512:],
            key_got if with_local else key_ref,
            rope_got if with_local else rope_ref,
            slots,
            reciprocal,
            source_slots=source_slots,
            local_query=local_query if with_local else None,
            local_head_start=rank * 12,
        )

    def bits(value):
        return value.cpu().contiguous().view(torch.uint8)

    def check(actual):
        expected = run(False)
        reference_local = torch.cat(
            (query[:, rank * 12 : (rank + 1) * 12], query_rope[:, rank * 12 : (rank + 1) * 12]), dim=-1
        )
        torch.npu.synchronize()
        assert len(actual) == 3
        for got, ref in zip((*actual, key_got, rope_got, local_query), (*expected, key_ref, rope_ref, reference_local)):
            assert torch.equal(bits(got), bits(ref))

    check(run(True))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run(True)
    query.mul_(0.9)
    query_rope.add_(0.25)
    local_query.fill_(3)
    graph.replay()
    check(captured)
    # Verify that the current query preserves BF16 bits rather than round-tripping
    # through the FP32 quantization arithmetic, including signed zero/NaN payload.
    special = torch.tensor([-32768, 0, 32705, 32640, -128, 32639], dtype=torch.int16, device="npu")
    query.view(torch.int16)[0, rank * 12, :6] = special
    query_rope.view(torch.int16)[0, rank * 12, :6] = special
    graph.replay()
    check(captured)
