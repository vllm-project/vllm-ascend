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
def _require_ascend950_sfa():
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse SFA integration currently targets Ascend 950")
    torch.npu.set_device(0)
    from vllm_ascend.utils import bootstrap_custom_op_env

    # Load the required SFA extension without enabling all A5 custom paths.
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
    impl = SimpleNamespace(kv_lora_rank=latent_dim, qk_rope_head_dim=rope_dim, scale=scale)
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
def test_valid_order_duplicates_and_interleaved_padding(monkeypatch, capture):
    from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl
    from vllm_ascend.ops import hisparse

    torch.manual_seed(61)
    device, topk, latent, rope, capacity = "npu:0", 2048, 512, 64, 32
    workspace = (
        torch.empty((capacity, topk, 1, latent), dtype=torch.bfloat16, device=device),
        torch.empty((capacity, topk, 1, rope), dtype=torch.bfloat16, device=device),
        torch.arange(capacity * topk // 128, dtype=torch.int32, device=device).view(capacity, -1),
        torch.arange(topk, dtype=torch.int32, device=device).view(1, 1, topk),
        torch.arange(1, capacity + 1, dtype=torch.int32, device=device),
        torch.full((capacity,), topk, dtype=torch.int32, device=device),
    )
    impl = SimpleNamespace(
        topk=topk, kv_lora_rank=latent, qk_rope_head_dim=rope, scale=0.0625, _workspace=lambda _: workspace
    )
    query = torch.randn(3, 8, latent, dtype=torch.bfloat16, device=device)
    query_rope = torch.randn(3, 8, rope, dtype=torch.bfloat16, device=device)
    cache = torch.randn(1, 128, latent + rope, dtype=torch.bfloat16, device=device)
    indices = torch.full((3, topk), -1, dtype=torch.int32)
    indices[0, :6] = torch.tensor([7, -1, 2, 7, -1, 0])
    indices[2, :6] = torch.tensor([0, 5, -1, 3, 5, 1])
    device_indices = indices.to(device)
    observed = torch.empty_like(device_indices)
    original_pack = hisparse.pack_attention_rows

    def observe(cache, physical, selected_latent, selected_rope, latent_dim):
        observed.copy_(physical)
        return original_pack(cache, physical, selected_latent, selected_rope, latent_dim)

    monkeypatch.setattr(hisparse, "pack_attention_rows", observe)

    def forward():
        return AscendHiSparseImpl._attention_physical(impl, query, query_rope, cache, device_indices)

    graph = None
    try:
        if capture:
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                forward()
                forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                result = forward()
            indices[0, :6] = torch.tensor([-1, 9, 1, 9, 4, -1])
            device_indices.copy_(indices)
            for _ in range(3):
                graph.replay()
        else:
            result = forward()
        expected = torch.full_like(indices, -1)
        for row in range(3):
            valid = indices[row][indices[row] >= 0]
            expected[row, : valid.numel()] = valid
        torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
        assert torch.isfinite(result.cpu()).all()
        assert torch.count_nonzero(result[1].cpu()) == 0
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
