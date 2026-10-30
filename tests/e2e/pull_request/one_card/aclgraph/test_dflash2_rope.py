# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise DFlash's paired Q/K RoPE through Dynamo, AOT and ACL replay.

The AOT eager backend checks fake-tensor tracing and functionalization while
retaining the production NPU operators. It does not test compiler fusion or
performance; those require the model-level Ascend backend tests.
"""

import pytest
import torch
import torch_npu
from torch._dynamo.backends.registry import lookup_backend

import vllm_ascend.ops.register_custom_ops  # noqa: F401
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

_HEAD_SIZE = 128
_QUERY_HEADS = 2
_KV_HEADS = 1
_MAX_POSITION = 64
_EPSILON = 1e-6


class DraftQKRotary(torch.nn.Module):
    def __init__(self, rotary_dim):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.register_buffer("weight", torch.ones(_HEAD_SIZE, dtype=torch.bfloat16, device="npu"))

    def forward(self, qkv, positions, cos_sin_cache):
        q, k, _ = qkv.split([_QUERY_HEADS * _HEAD_SIZE, _KV_HEADS * _HEAD_SIZE, _KV_HEADS * _HEAD_SIZE], dim=-1)
        q_normalized, _ = torch_npu.npu_rms_norm(q.view(-1, _QUERY_HEADS, _HEAD_SIZE), self.weight, _EPSILON)
        k_normalized, _ = torch_npu.npu_rms_norm(k.view(-1, _KV_HEADS, _HEAD_SIZE), self.weight, _EPSILON)
        return torch.ops.vllm.npu_rotary_embedding(
            positions,
            q_normalized.view(q.shape),
            k_normalized.view(k.shape),
            cos_sin_cache,
            _HEAD_SIZE,
            self.rotary_dim,
            True,
        )


def _reference(qkv, positions, cache, rotary_dim):
    q, k, _ = qkv.float().split([_QUERY_HEADS * _HEAD_SIZE, _KV_HEADS * _HEAD_SIZE, _KV_HEADS * _HEAD_SIZE], dim=-1)
    cos, sin = cache[positions].float().unsqueeze(1).chunk(2, dim=-1)
    outputs = []
    for tensor in (q, k):
        heads = tensor.reshape(tensor.shape[0], -1, _HEAD_SIZE)
        normalized = (heads * torch.rsqrt(heads.square().mean(-1, keepdim=True) + _EPSILON)).bfloat16().float()
        left, right = normalized[..., :rotary_dim].chunk(2, dim=-1)
        rotated = torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)
        outputs.append(torch.cat((rotated, normalized[..., rotary_dim:]), dim=-1).reshape(tensor.shape).bfloat16())
    return outputs


@pytest.mark.e2e_coverage(
    arch="dense",
    feature="spec_decode,aclgraph",
    parallel="",
    deploy="",
    hardware="",
    quantization="",
    graph_mode="full_graph",
)
@pytest.mark.parametrize("num_speculative_tokens", [4, 5, 6])
@pytest.mark.parametrize("rotary_dim", [64, _HEAD_SIZE])
@torch.inference_mode()
def test_compiled_draft_qk_rope_graph_replay(num_speculative_tokens, rotary_dim):
    torch.npu.set_device(0)
    init_device_properties_triton()
    model = DraftQKRotary(rotary_dim)
    compiled_graphs = []
    aot_eager = lookup_backend("aot_eager")

    def backend(graph, example_inputs):
        assert any(
            node.op == "call_function" and "vllm.npu_rotary_embedding" in str(node.target) for node in graph.graph.nodes
        )
        compiled_graphs.append(graph)
        return aot_eager(graph, example_inputs)

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
    frequencies = 10000.0 ** (-torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    angles = torch.arange(_MAX_POSITION).float()[:, None] * frequencies[None, :]
    cache_cpu = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()
    cache = cache_cpu.npu()
    generator = torch.Generator().manual_seed(19)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        # Reuse the compiled callable at different token counts, then capture
        # one graph per shape as the production graph manager does.
        for num_reqs in (2, 3):
            num_tokens = num_reqs * (num_speculative_tokens + 1)
            qkv = (
                torch.randn(num_tokens, (_QUERY_HEADS + 2 * _KV_HEADS) * _HEAD_SIZE, generator=generator)
                .bfloat16()
                .npu()
            )
            positions = torch.arange(num_tokens, device="npu")
            compiled(qkv, positions, cache)
            torch.npu.synchronize()
            assert compiled_graphs
            compilation_count = len(compiled_graphs)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                captured = compiled(qkv, positions, cache)
            for offset in (0, 9):
                input_cpu = torch.randn(qkv.shape, generator=generator).bfloat16()
                positions_cpu = torch.arange(num_tokens) + offset
                qkv.copy_(input_cpu.npu())
                positions.copy_(positions_cpu.npu())
                graph.replay()
                eager = model(qkv, positions, cache)
                torch.npu.synchronize()
                expected = _reference(input_cpu, positions_cpu, cache_cpu, rotary_dim)
                for actual, eager_output, reference in zip(captured, eager, expected):
                    torch.testing.assert_close(actual.cpu(), eager_output.cpu(), rtol=0.01, atol=0.01)
                    torch.testing.assert_close(actual.cpu(), reference, rtol=0.02, atol=0.02)
            assert len(compiled_graphs) == compilation_count
