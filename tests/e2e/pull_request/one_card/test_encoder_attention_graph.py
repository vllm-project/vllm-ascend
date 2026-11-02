# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl


def encoder_reference(query, key, value, lengths, sliding_window):
    """Independent FP32 dense attention, with no cross-request visibility."""
    outputs = []
    start = 0
    for end in lengths:
        q, k, v = [x[start:end].float().transpose(0, 1) for x in (query, key, value)]
        repeats = q.shape[0] // k.shape[0]
        k, v = k.repeat_interleave(repeats, 0), v.repeat_interleave(repeats, 0)
        scores = q @ k.transpose(-1, -2) * query.shape[-1] ** -0.5
        if sliding_window is not None:
            positions = torch.arange(end - start)
            mask = (positions[:, None] - positions[None, :]).abs() >= sliding_window
            scores.masked_fill_(mask, float("-inf"))
        weights = scores.softmax(-1)
        outputs.append((weights @ v).transpose(0, 1))
        start = end
    return torch.cat(outputs)


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
@pytest.mark.parametrize("num_heads,num_kv_heads,head_size", [(4, 4, 128), (8, 2, 128), (24, 4, 256)])
@pytest.mark.parametrize("padding", [0, 7])
@pytest.mark.parametrize("sliding_window", [None, 8, 64])
@torch.inference_mode()
def test_encoder_attention_changed_input_replay(dtype, num_heads, num_kv_heads, head_size, padding, sliding_window):
    torch.npu.set_device(0)
    torch.manual_seed(37)
    lengths = [17, 48]
    shape_tokens = lengths[-1] + padding
    cpu = [torch.randn(shape_tokens, heads, head_size).to(dtype) for heads in (num_heads, num_kv_heads, num_kv_heads)]
    device = [x.npu() for x in cpu]
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.num_heads, impl.num_kv_heads, impl.scale = num_heads, num_kv_heads, head_size**-0.5
    impl.sliding_window = sliding_window
    metadata = SimpleNamespace(actual_seq_lengths_q=lengths)
    output = torch.empty_like(device[0])

    def forward():
        return impl._forward_encoder_attention(*device, metadata, output)

    atol = 0.008 if dtype == torch.bfloat16 else 0.001
    expected = encoder_reference(cpu[0], cpu[1], cpu[2], lengths, sliding_window)
    actual = forward()
    torch.testing.assert_close(actual[: lengths[-1]].cpu().float(), expected, atol=atol, rtol=0.01)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = forward()
    for seed in (41, 43):
        torch.manual_seed(seed)
        cpu = [torch.randn_like(x) for x in cpu]
        for static, changed in zip(device, cpu):
            static.copy_(changed)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            captured[: lengths[-1]].cpu().float(),
            encoder_reference(cpu[0], cpu[1], cpu[2], lengths, sliding_window),
            atol=atol,
            rtol=0.01,
        )
