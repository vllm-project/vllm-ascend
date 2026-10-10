# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch

from vllm_ascend.ops.triton.linearnorm.minimax_qknorm_rope_cache import minimax_qknorm_rope_cache


def reference(packed, cos_sin, positions, weights, q_heads, kv_heads, index_heads, eps):
    head_dim = weights[0].numel()
    sizes = [q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim, index_heads * head_dim, head_dim]
    parts = list(packed.split(sizes, -1))
    half = cos_sin.shape[-1] // 2
    cs = cos_sin[positions].float().unsqueeze(1)
    for part, weight in zip((0, 1, 3, 4), weights):
        x = parts[part].reshape(packed.shape[0], -1, head_dim).float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (1 + weight.float())
        x = x.to(packed.dtype).float()
        a, b = x[..., :half].clone(), x[..., half : 2 * half].clone()
        x[..., :half] = a * cs[..., :half] - b * cs[..., half:]
        x[..., half : 2 * half] = b * cs[..., :half] + a * cs[..., half:]
        parts[part] = x.reshape(packed.shape[0], -1)
    return parts


@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32, 64, 128, 192, 256, 513])
@pytest.mark.parametrize("heads", [(8, 1, 1), (16, 2, 2), (5, 1, 3)])
@pytest.mark.parametrize("rotary_dim", [64, 128])
@torch.inference_mode()
def test_minimax_qknorm_rope_cache(tokens, heads, rotary_dim):
    torch.manual_seed(17)
    q_heads, kv_heads, index_heads = heads
    head_dim, eps, block_size = 128, 1e-6, 16
    total_heads = q_heads + 2 * kv_heads + index_heads + 1
    # A non-contiguous token stride exercises fused projections with padding.
    packed = torch.randn((tokens, total_heads * head_dim + 32), dtype=torch.bfloat16)[:, : total_heads * head_dim]
    weights = [torch.randn(head_dim, dtype=torch.bfloat16) * 0.1 for _ in range(4)]
    angles = torch.randn((257, rotary_dim // 2))
    cos_sin = torch.cat((angles.cos(), angles.sin()), -1).to(torch.bfloat16)
    positions = torch.randint(257, (tokens,))
    actual = max(1, tokens - 3)
    blocks = (2 * tokens + block_size - 1) // block_size
    slots = torch.randperm(blocks * block_size)[:actual].long()
    index_slots = torch.randperm(blocks * block_size)[:actual].long()
    slots[::5] = -1
    index_slots[::7] = -1
    cache_shape = (blocks, block_size, kv_heads, head_dim)
    key_cache = torch.full(cache_shape, -3.0, dtype=torch.bfloat16, device="npu")
    value_cache = torch.full_like(key_cache, -4.0)
    index_cache = torch.full((blocks, block_size, head_dim), -5.0, dtype=torch.bfloat16, device="npu")
    packed_npu = torch.empty((tokens, total_heads * head_dim + 32), dtype=packed.dtype, device="npu")
    packed_npu = packed_npu[:, : total_heads * head_dim]
    packed_npu.copy_(packed)
    # Exercise independent main/index projections without concatenating them.
    main_width = (q_heads + 2 * kv_heads) * head_dim
    separate_index = tokens % 2 == 0
    q, iq = minimax_qknorm_rope_cache(
        packed_npu[:, :main_width] if separate_index else packed_npu,
        cos_sin.npu(),
        positions.npu(),
        *[w.npu() for w in weights],
        key_cache,
        value_cache,
        index_cache,
        slots.npu(),
        index_slots.npu(),
        actual,
        q_heads,
        kv_heads,
        index_heads,
        eps,
        index_packed=packed_npu[:, main_width:] if separate_index else None,
    )
    expected = reference(packed, cos_sin, positions, weights, *heads, eps)
    torch.testing.assert_close(q.cpu().float(), expected[0].to(q.dtype).float(), atol=0.032, rtol=0.01)
    torch.testing.assert_close(iq.cpu().float(), expected[3].to(iq.dtype).float(), atol=0.032, rtol=0.01)
    for cache, part, mapping, sentinel in (
        (key_cache, expected[1], slots, -3.0),
        (value_cache, expected[2], slots, -4.0),
        (index_cache, expected[4], index_slots, -5.0),
    ):
        gold = torch.full(cache.shape, sentinel, dtype=cache.dtype).view(blocks * block_size, -1)
        valid = mapping >= 0
        gold[mapping[valid]] = part[:actual][valid].to(cache.dtype)
        torch.testing.assert_close(cache.cpu().float().reshape_as(gold), gold.float(), atol=0.032, rtol=0.01)
        # Untouched rows must remain bit-exact, including padding and negative slots.
        untouched = torch.ones(blocks * block_size, dtype=torch.bool)
        untouched[mapping[valid]] = False
        assert torch.equal(cache.cpu().view(blocks * block_size, -1)[untouched], gold[untouched])


@torch.inference_mode()
def test_empty_minimax_qknorm_rope_cache():
    device = "npu"
    weight = torch.zeros(128, dtype=torch.bfloat16, device=device)
    cache = torch.zeros((1, 16, 1, 128), dtype=torch.bfloat16, device=device)
    index_cache = torch.zeros((1, 16, 128), dtype=torch.bfloat16, device=device)
    slots = torch.empty(0, dtype=torch.int64, device=device)
    q, iq = minimax_qknorm_rope_cache(
        torch.empty((0, 12 * 128), dtype=torch.bfloat16, device=device),
        torch.empty((16, 64), dtype=torch.bfloat16, device=device),
        slots,
        weight,
        weight,
        weight,
        weight,
        cache,
        cache.clone(),
        index_cache,
        slots,
        slots,
        0,
        8,
        1,
        1,
        1e-6,
    )
    assert q.shape == (0, 1024) and iq.shape == (0, 128)
    assert torch.count_nonzero(cache).item() == 0


@pytest.mark.parametrize("blocks,tokens", [(3, 8), (64, 192), (5392, 192), (8192, 256)])
@torch.inference_mode()
def test_dynamic_cache_capacity_and_strided_positions(blocks, tokens):
    torch.manual_seed(71)
    hd, block_size, max_position = 128, 128, 263000
    packed = torch.randn(tokens, 12 * hd, dtype=torch.bfloat16)
    weights = [torch.randn(hd, dtype=packed.dtype) * 0.1 for _ in range(4)]
    # A padded cos/sin row and a strided positions view must not add a copy.
    cs = torch.randn(max_position, 80, dtype=packed.dtype)[:, :64]
    cs_npu = torch.empty((max_position, 80), dtype=packed.dtype, device="npu")[:, :64]
    cs_npu.copy_(cs)
    positions = torch.randint(max_position, (tokens,))
    positions[0], positions[-1] = 0, max_position - 1
    positions_npu = torch.empty((tokens, 2), dtype=torch.int64, device="npu")[:, 0]
    positions_npu.copy_(positions)
    # Exercise the end of each dynamic allocation, not only its first pages.
    slots = torch.arange(blocks * block_size - tokens, blocks * block_size)
    index_slots = slots.flip(0)
    key = torch.zeros((blocks, block_size, 1, hd), dtype=packed.dtype, device="npu")
    value = torch.zeros_like(key)
    index = torch.zeros((blocks, block_size, hd), dtype=packed.dtype, device="npu")
    q, iq = minimax_qknorm_rope_cache(
        packed[:, : 10 * hd].contiguous().npu(),
        cs_npu,
        positions_npu,
        *[w.npu() for w in weights],
        key,
        value,
        index,
        slots.npu(),
        index_slots.npu(),
        tokens,
        8,
        1,
        1,
        1e-6,
        index_packed=packed[:, 10 * hd :].contiguous().npu(),
    )
    expected = reference(packed, cs, positions, weights, 8, 1, 1, 1e-6)
    for actual, gold in (
        (q, expected[0]),
        (iq, expected[3]),
        (key.view(-1, hd)[slots.npu()], expected[1]),
        (value.view(-1, hd)[slots.npu()], expected[2]),
        (index.view(-1, hd)[index_slots.npu()], expected[4]),
    ):
        torch.testing.assert_close(actual.cpu().float(), gold.to(packed.dtype).float(), atol=0.032, rtol=0.01)
    assert torch.count_nonzero(key[0].float()).item() == 0


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("tokens", [8, 192])
@torch.inference_mode()
def test_cache_graph_replay_updates_slots_and_preserves_strides(fp8, tokens):
    if fp8 and not any(tag in torch.npu.get_device_name(0).lower() for tag in ("950", "910_95")):
        pytest.skip("FP8 cache conversion requires A5")
    torch.manual_seed(29)
    qh, kh, ih, hd, actual = 8, 1, 1, 128, 5
    cache_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    packed = torch.randn(tokens, 12 * hd, dtype=torch.bfloat16, device="npu")
    if fp8:
        # V saturation is independent of normalization.
        packed[:, 9 * hd : 10 * hd] *= 1000
    weights = [torch.randn(hd, dtype=packed.dtype, device="npu") * 0.1 for _ in range(4)]
    cs = torch.randn(tokens + 32, 64, dtype=packed.dtype, device="npu")
    positions = torch.arange(tokens, device="npu")
    # Different main/index block sizes and padded block strides.
    key = torch.zeros((3, 17, kh, hd), dtype=cache_dtype, device="npu")[:, :16]
    value = torch.zeros((3, 18, kh, hd), dtype=cache_dtype, device="npu")[:, :16]
    index = torch.zeros((6, 9, hd), dtype=cache_dtype, device="npu")[:, :8]
    slots = torch.full((tokens,), -1, dtype=torch.int64, device="npu")
    index_slots = slots.clone()

    def run():
        return minimax_qknorm_rope_cache(
            packed,
            cs,
            positions,
            *weights,
            key,
            value,
            index,
            slots,
            index_slots,
            actual,
            qh,
            kh,
            ih,
            1e-6,
        )

    # An all-padding warmup must leave every cache row untouched.
    run()
    assert torch.count_nonzero(key.float()).item() == 0
    assert torch.count_nonzero(value.float()).item() == 0
    assert torch.count_nonzero(index.float()).item() == 0
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        q, iq = run()
    # Change device data after capture: positions and slots must not be baked in.
    positions.copy_(torch.arange(tokens, device="npu") + 11)
    slots[:8].copy_(torch.tensor([17, -1, 0, 35, 7, 20, 21, 22], device="npu"))
    index_slots[:8].copy_(torch.tensor([4, 18, -1, 9, 34, 30, 31, 32], device="npu"))
    graph.replay()
    torch.npu.synchronize()
    expected = reference(packed.cpu(), cs.cpu(), positions.cpu(), [w.cpu() for w in weights], qh, kh, ih, 1e-6)
    torch.testing.assert_close(q.cpu().float(), expected[0].to(q.dtype).float(), atol=0.032, rtol=0.01)
    expected_iq = expected[3].clamp(-448, 448) if fp8 else expected[3]
    torch.testing.assert_close(
        iq.cpu().float(), expected_iq.to(iq.dtype).float(), atol=0.25 if fp8 else 0.032, rtol=0.01
    )
    for cache, part, mapping in (
        (key, expected[1], slots),
        (value, expected[2], slots),
        (index, expected[4], index_slots),
    ):
        gold = torch.zeros(cache.shape, dtype=cache_dtype)
        part = part.to(packed.dtype) if cache is not index else part
        if fp8:
            part = part.float().clamp(-448, 448)
        for token, slot in enumerate(mapping.cpu().tolist()[:actual]):
            if slot >= 0:
                gold[slot // cache.shape[1], slot % cache.shape[1]] = part[token].reshape_as(gold[0, 0]).to(cache_dtype)
        torch.testing.assert_close(cache.cpu().float(), gold.float(), atol=0.25 if fp8 else 0.032, rtol=0.01)
    # Replay repeatedly while another stream generates memory traffic.
    traffic = torch.empty(1 << 20, device="npu")
    stream = torch.npu.Stream()
    saved_q, saved_iq = q.clone(), iq.clone()
    saved_key, saved_index = key.clone(), index.clone()
    for _ in range(100):
        with torch.npu.stream(stream):
            traffic.fill_(1.0)
        graph.replay()
    torch.npu.synchronize()
    assert torch.equal(q.cpu(), saved_q.cpu())
    assert torch.equal(iq.cpu().float(), saved_iq.cpu().float())
    assert torch.equal(key.cpu().float(), saved_key.cpu().float())
    assert torch.equal(index.cpu().float(), saved_index.cpu().float())
