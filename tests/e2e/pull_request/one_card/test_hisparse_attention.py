# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the actual SFA bridge against an independent FP32 CPU reference."""

import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401


@pytest.fixture(autouse=True)
def _require_npu_sfa():
    if not torch.npu.is_available():
        pytest.skip("HiSparse SFA tests require an NPU")
    from vllm_ascend.device.device_config import is_310p

    if is_310p():
        pytest.skip("The native SFA operation is not registered for Ascend 310P")
    torch.npu.set_device(0)
    from vllm_ascend.utils import bootstrap_custom_op_env

    # Load the required SFA extension without changing custom-op selection.
    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # noqa: F401


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("num_queries", [1, 7, 35])
def test_sparse_attention_reference(num_queries, strided, capture):
    from vllm_ascend.attention.sfa_hisparse import _ATTENTION_CHUNK_ROWS, AscendHiSparseImpl

    torch.manual_seed(47)
    device = torch.device("npu:0")
    topk, latent_dim, rope_dim, heads = 2048, 512, 64, 8
    block_size, blocks = 128, 16
    capacity = _ATTENTION_CHUNK_ROWS
    scale = 1 / math.sqrt(latent_dim + rope_dim)
    workspace = (
        torch.empty((capacity, topk, 1, latent_dim), device=device, dtype=torch.bfloat16),
        torch.empty((capacity, topk, 1, rope_dim), device=device, dtype=torch.bfloat16),
        torch.arange(capacity * topk // block_size, dtype=torch.int32, device=device).view(capacity, -1),
        torch.arange(topk, dtype=torch.int32, device=device).view(1, 1, topk),
        torch.arange(1, capacity + 1, dtype=torch.int32, device=device),
        torch.full((capacity,), topk, dtype=torch.int32, device=device),
    )
    impl = SimpleNamespace(kv_lora_rank=latent_dim, qk_rope_head_dim=rope_dim, scale=scale, enable_sparse_sfa_c8=False)
    impl._workspace = lambda dtype: workspace
    # Quantize before taking the FP32 reference so both paths see the same inputs.
    source = torch.randn((blocks, block_size, latent_dim + rope_dim)).to(torch.bfloat16)
    query = torch.randn((num_queries, heads, latent_dim)).to(torch.bfloat16)
    query_rope = torch.randn((num_queries, heads, rope_dim)).to(torch.bfloat16)
    physical = torch.randint(0, blocks * block_size, (num_queries, topk), dtype=torch.int32)
    physical[:, 3::7] = -1
    physical[:, 1] = physical[:, 0]
    # A short row makes duplicate multiplicity and invalid-padding mistakes large.
    physical[0] = -1
    physical[0, :3] = torch.tensor([0, 0, 1], dtype=torch.int32)
    if num_queries > 1:
        physical[-1] = -1
    if strided:
        backing = torch.empty((blocks, 2, block_size, latent_dim + rope_dim), device=device, dtype=source.dtype)
        cache = backing[:, 1]
        cache.copy_(source)
    else:
        cache = source.to(device)
    query_npu, rope_npu, physical_npu = query.to(device), query_rope.to(device), physical.to(device)
    forward = partial(AscendHiSparseImpl._attention_physical, impl, query_npu, rope_npu, cache, physical_npu)

    graph = None
    try:
        if capture:
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(2):
                    forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                captured = forward()
            # Change both payload and selection length after graph capture.
            query.neg_()
            physical[0, :5] = torch.tensor([3, 3, 8, 12, 14], dtype=torch.int32)
            query_npu.copy_(query)
            physical_npu.copy_(physical)
            for _ in range(3):
                graph.replay()
            actual = captured.float().cpu()
        else:
            actual = forward().float().cpu()

        selected = source.flatten(0, 1)[physical.clamp_min(0).long()].float()
        logits = torch.einsum("thd,tkd->thk", query.float(), selected[..., :latent_dim])
        logits += torch.einsum("thd,tkd->thk", query_rope.float(), selected[..., latent_dim:])
        logits *= scale
        logits.masked_fill_(~(physical >= 0)[:, None, :], -torch.inf)
        probabilities = torch.softmax(logits, dim=-1).nan_to_num(0)
        expected = torch.einsum("thk,tkd->thd", probabilities, selected[..., :latent_dim])
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.025)
        if num_queries > 1:
            assert torch.count_nonzero(actual[-1]) == 0
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_c8_attention_uses_existing_native_operator(strided, capture):
    from vllm_ascend.device.device_config import is_950

    if not is_950():
        pytest.skip("This case uses the native A5 E4M3FN C8 payload; A2/A3 use an INT8 format")
    from vllm_ascend.attention.sfa_v1 import custom_kv_rmsnorm_rope
    from vllm_ascend.ops.hisparse_c8 import c8_physical_attention, pack_c8_components

    torch.manual_seed(71)
    rows, heads, blocks, block, topk = 3, 8, 2, 128, 128
    device = "npu:0"
    values = torch.randn(blocks * block, 1, 1, 576, dtype=torch.bfloat16, device=device)
    gamma = torch.ones(512, dtype=torch.bfloat16, device=device)
    cos = torch.ones(blocks * block, 1, 1, 64, dtype=torch.bfloat16, device=device)
    sin = torch.zeros_like(cos)
    rope, nope, scales = custom_kv_rmsnorm_rope(
        values, gamma, cos, sin, 512, 64, epsilon=1e-5, dst_type=torch.float8_e4m3fn, tile_size=128
    )
    packed = pack_c8_components(nope.contiguous(), rope.contiguous(), scales.contiguous())
    backing = torch.empty((blocks, 2 if strided else 1, block, 656), dtype=torch.float8_e4m3fn, device=device)
    cache = backing[:, -1]
    cache.view(torch.uint8).copy_(packed.reshape(blocks, block, 656))
    q = torch.randn(rows, heads, 512, dtype=torch.bfloat16, device=device) * 0.25
    qr = torch.randn(rows, heads, 64, dtype=torch.bfloat16, device=device) * 0.25
    indices = torch.randint(0, blocks * block, (rows, topk), dtype=torch.int32)
    indices[:, 3::5] = -1
    indices[0] = -1
    indices[0, :3] = torch.tensor([3, 3, 7], dtype=torch.int32)
    indices[-1] = -1
    selected = indices.to(device)
    query_end, key_length = (torch.empty(1, dtype=torch.int32, device=device) for _ in range(2))

    def forward():
        return c8_physical_attention(q, qr, cache, selected, 1 / math.sqrt(576), query_end, key_length)

    graph = None
    try:
        if capture:
            forward()
            forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = forward()
            indices[0, :4] = torch.tensor([9, 2, 9, 0], dtype=torch.int32)
            selected.copy_(indices)
            q.neg_()
            graph.replay()
        else:
            output = forward()
        torch.npu.synchronize()
        scale = scales.contiguous().view(torch.float32).cpu().reshape(-1, 4).repeat_interleave(128, dim=-1)
        latent = nope.float().cpu().reshape(-1, 512) * scale
        rope_cpu = rope.contiguous().view(torch.bfloat16).cpu().reshape(-1, 64).float()
        keys = latent[indices.clamp_min(0).long()]
        rope_keys = rope_cpu[indices.clamp_min(0).long()]
        logits = torch.einsum("thd,tkd->thk", q.float().cpu(), keys)
        logits += torch.einsum("thd,tkd->thk", qr.float().cpu(), rope_keys)
        logits *= 1 / math.sqrt(576)
        logits.masked_fill_(~(indices >= 0)[:, None, :], -torch.inf)
        probability = logits.softmax(-1).nan_to_num(0)
        expected = torch.einsum("thk,tkd->thd", probability, keys)
        torch.testing.assert_close(output.float().cpu(), expected, atol=0.01, rtol=0.05)
        assert torch.count_nonzero(output[-1].cpu()) == 0
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
