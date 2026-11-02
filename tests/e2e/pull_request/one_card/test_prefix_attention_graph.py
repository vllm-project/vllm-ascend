# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendMetadata


def prefix_reference(query, key, value, lengths, boundaries):
    outputs, masks = [], []
    start = 0
    for end, boundary in zip(lengths, boundaries):
        length = end - start
        # Two bidirectional blocks: state queries cannot read question keys.
        mask = torch.tensor([[q < boundary <= k for k in range(length)] for q in range(length)])
        q, k, v = [x[start:end].float().transpose(0, 1) for x in (query, key, value)]
        repeats = q.shape[0] // k.shape[0]
        k, v = k.repeat_interleave(repeats, 0), v.repeat_interleave(repeats, 0)
        scores = (q @ k.transpose(-1, -2) * query.shape[-1] ** -0.5).masked_fill(mask, float("-inf"))
        outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
        masks.append(mask)
        start = end
    return torch.cat(outputs), masks


@pytest.mark.e2e_coverage(
    arch="classification",
    feature="pooling,aclgraph,mixed_lengths",
    parallel="",
    deploy="",
    hardware="A2",
    quantization="",
    graph_mode="aclgraph",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads,num_kv_heads,head_size", [(4, 4, 128), (24, 4, 256)])
@pytest.mark.parametrize("padding", [0, 7])
@torch.inference_mode()
def test_prefix_attention_changed_boundary_replay(dtype, num_heads, num_kv_heads, head_size, padding):
    torch.npu.set_device(0)
    torch.manual_seed(37)
    lengths = [17, 48]
    cpu = [torch.randn(48 + padding, heads, head_size).to(dtype) for heads in (num_heads, num_kv_heads, num_kv_heads)]
    device = [x.npu() for x in cpu]
    expected, masks = prefix_reference(*cpu, lengths, [8, 19])
    metadata = AscendMetadata(actual_seq_lengths_q=lengths, prefix_masks=[m.npu() for m in masks])
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.num_heads, impl.num_kv_heads, impl.scale = num_heads, num_kv_heads, head_size**-0.5
    impl.sliding_window = None
    output = torch.empty_like(device[0])

    def forward():
        return impl._forward_prefix_attention(*device, metadata, output)

    atol = 0.008 if dtype == torch.bfloat16 else 0.001
    torch.testing.assert_close(forward()[:48].cpu().float(), expected, atol=atol, rtol=0.01)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = forward()
    for seed, boundaries in [(41, [3, 7]), (43, [15, 23])]:
        torch.manual_seed(seed)
        cpu = [torch.randn_like(x) for x in cpu]
        expected, masks = prefix_reference(*cpu, lengths, boundaries)
        for static, changed in zip(device, cpu):
            static.copy_(changed)
        for static, changed in zip(metadata.prefix_masks, masks):
            static.copy_(changed)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured[:48].cpu().float(), expected, atol=atol, rtol=0.01)
