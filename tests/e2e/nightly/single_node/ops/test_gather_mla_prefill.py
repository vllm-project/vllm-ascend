# SPDX-License-Identifier: Apache-2.0
"""NPU accuracy gate for fused C8 history gather/dequantization."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.gather_mla_prefill import gather_dequant_mla_prefill


@pytest.mark.parametrize("lengths", [(1, 129, 0), (129, 257, 17), (128, 128, 128)])
@pytest.mark.parametrize("scale", [0.03125, 0.13])
@pytest.mark.parametrize("row_stride", [1, 2])
@torch.inference_mode()
def test_fused_gather_preserves_page_offsets_strides_and_static_scale(lengths, scale, row_stride):
    torch.manual_seed(891)
    latent_cpu = (torch.randn(8, 3, 128 * row_stride, 1, 512) * 7).to(torch.float8_e4m3fn)
    rope_cpu = torch.randn(8, 2, 128 * row_stride, 1, 64).to(torch.bfloat16)
    latent_storage, rope_storage = latent_cpu.npu(), rope_cpu.npu()
    latent_cache, rope_cache = latent_storage[:, 1, ::row_stride], rope_storage[:, 1, ::row_stride]
    blocks = torch.tensor([[7, 1, 5, 3], [6, 0, 4, 2], [2, 5, 0, 7]], dtype=torch.int32)
    table_storage = torch.full((3, 8), -1, dtype=torch.int32)
    table_storage[:, ::2] = blocks
    table = table_storage.npu()[:, ::2]
    starts = (0, 127, 128)
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    scale_tensor = torch.tensor([scale], dtype=torch.float32)
    actual_latent, actual_rope = gather_dequant_mla_prefill(
        latent_cache,
        rope_cache,
        table,
        torch.tensor(cumulative, dtype=torch.int32, device="npu"),
        torch.tensor(lengths, dtype=torch.int32, device="npu"),
        torch.tensor(starts, dtype=torch.int32, device="npu"),
        scale_tensor.npu(),
        num_tokens=sum(lengths),
        max_seq_len=max(lengths),
    )
    expected_latent, expected_rope = [], []
    for request, (start, length) in enumerate(zip(starts, lengths)):
        logical = torch.arange(start, start + length)
        pages = blocks[request, logical // 128].long()
        expected_latent.append(
            (latent_cpu[:, 1, ::row_stride].float()[pages, logical % 128] * scale_tensor).to(torch.bfloat16)
        )
        expected_rope.append(rope_cpu[:, 1, ::row_stride][pages, logical % 128])
    torch.testing.assert_close(actual_latent.cpu(), torch.cat(expected_latent), rtol=0, atol=0)
    torch.testing.assert_close(actual_rope.cpu(), torch.cat(expected_rope), rtol=0, atol=0)
    torch.testing.assert_close(latent_storage.cpu().float(), latent_cpu.float(), rtol=0, atol=0)
    torch.testing.assert_close(rope_storage.cpu(), rope_cpu, rtol=0, atol=0)


@torch.inference_mode()
def test_empty_fused_gather_does_not_launch():
    latent, rope = gather_dequant_mla_prefill(
        torch.empty(1, 128, 1, 512, dtype=torch.float8_e4m3fn, device="npu"),
        torch.empty(1, 128, 1, 64, dtype=torch.bfloat16, device="npu"),
        torch.empty(0, 1, dtype=torch.int32, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
        torch.empty(0, dtype=torch.int32, device="npu"),
        torch.empty(0, dtype=torch.int32, device="npu"),
        torch.ones(1, device="npu"),
        num_tokens=0,
        max_seq_len=0,
    )
    assert latent.shape == (0, 1, 512) and rope.shape == (0, 1, 64)


@torch.inference_mode()
def test_fused_gather_graph_replay_reads_live_start_and_scale():
    torch.manual_seed(37)
    latent_cpu = torch.randn(3, 128, 1, 512).to(torch.float8_e4m3fn)
    rope_cpu = torch.randn(3, 128, 1, 64).to(torch.bfloat16)
    latent_cache, rope_cache = latent_cpu.npu(), rope_cpu.npu()
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device="npu")
    cumulative = torch.tensor([0, 33], dtype=torch.int32, device="npu")
    lengths = torch.tensor([33], dtype=torch.int32, device="npu")
    starts = torch.tensor([0], dtype=torch.int32, device="npu")
    scale = torch.tensor([0.25], device="npu")

    def gather():
        return gather_dequant_mla_prefill(
            latent_cache, rope_cache, table, cumulative, lengths, starts, scale, num_tokens=33, max_seq_len=33
        )

    for _ in range(2):
        gather()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        latent, rope = gather()
    starts.fill_(127)
    scale.fill_(0.125)
    graph.replay()
    logical = torch.arange(127, 160)
    pages = torch.tensor([2, 0, 1])[logical // 128]
    expected = (latent_cpu.float()[pages, logical % 128] * 0.125).bfloat16()
    torch.testing.assert_close(latent.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(rope.cpu(), rope_cpu[pages, logical % 128], rtol=0, atol=0)
