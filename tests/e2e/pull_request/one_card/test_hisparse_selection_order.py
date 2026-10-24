# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SFA bridge must preserve valid selection order across graph replay."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401


@pytest.mark.parametrize("capture", [False, True])
def test_valid_order_duplicates_and_interleaved_padding(monkeypatch, capture):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse SFA integration currently targets Ascend 950")
    torch.npu.set_device(0)
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # noqa: F401

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
