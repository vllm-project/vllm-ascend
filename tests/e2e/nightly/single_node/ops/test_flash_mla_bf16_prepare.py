# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import statistics

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401


def make_inputs(tokens, heads, rope_heads, strided):
    multiplier = 2 if strided else 1
    projection = torch.randn(tokens * multiplier, heads, 256, device="npu", dtype=torch.bfloat16)[::multiplier]
    key, value = projection.split(128, dim=-1)
    rope = torch.randn(tokens * multiplier, rope_heads, 64, device="npu", dtype=torch.bfloat16)[::multiplier]
    return key, value, rope


def reference(key, value, rope):
    return torch.cat((key, rope.expand(-1, key.shape[1], -1)), dim=-1), value.contiguous()


@pytest.mark.parametrize("tokens,heads", [(0, 12), (1, 1), (13, 12), (257, 12), (16384, 12), (13, 128)])
@pytest.mark.parametrize("rope_per_head", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_flash_mla_bf16_prepare_bitwise(tokens, heads, rope_per_head, strided):
    inputs = make_inputs(tokens, heads, heads if rope_per_head else 1, strided)
    expected = reference(*inputs)
    actual = torch.ops._C_ascend.flash_mla_bf16_prepare(*inputs)
    for output, golden in zip(actual, expected):
        assert output.is_contiguous()
        assert torch.equal(output.view(torch.int16), golden.view(torch.int16))


@torch.inference_mode()
def test_flash_mla_bf16_prepare_strided_heads_and_special_bit_patterns():
    patterns = torch.tensor([0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7F81, 1, 0xFFFF], dtype=torch.int32)
    bits = patterns.to(torch.int16).view(torch.bfloat16).npu()
    projection = bits.repeat(26 * 24 * 256 // bits.numel()).reshape(26, 24, 256)[::2, ::2]
    rope = bits.flip(0).repeat(26 * 24 * 64 // bits.numel()).reshape(26, 24, 64)[::2, ::2]
    inputs = (*projection.split(128, dim=-1), rope)
    for output, expected in zip(torch.ops._C_ascend.flash_mla_bf16_prepare(*inputs), reference(*inputs)):
        # View as integers: signed zero and NaN payload bits must survive too.
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


@torch.inference_mode()
def test_flash_mla_bf16_prepare_changed_input_graph_replay():
    inputs = make_inputs(137, 12, 1, True)
    torch.ops._C_ascend.flash_mla_bf16_prepare(*inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = torch.ops._C_ascend.flash_mla_bf16_prepare(*inputs)
    for _ in range(3):
        for tensor in inputs:
            tensor.copy_(torch.randn_like(tensor))
        graph.replay()
        for output, expected in zip(outputs, reference(*inputs)):
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


@torch.inference_mode()
def benchmark(tokens, heads, repeats, trials):
    inputs = make_inputs(tokens, heads, 1, False)
    methods = {
        "cat_contiguous": lambda: reference(*inputs),
        "native": lambda: torch.ops._C_ascend.flash_mla_bf16_prepare(*inputs),
    }
    graphs = {}
    for name, method in methods.items():
        for _ in range(5):
            method()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(repeats):
                output = method()
        graphs[name] = graph, output
        for _ in range(5):
            graph.replay()
    torch.npu.synchronize()
    samples = {name: [] for name in methods}
    names = tuple(methods)
    for trial in range(trials):
        for name in names if trial % 2 == 0 else names[::-1]:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / repeats)
    for observed, expected in zip(graphs["native"][1], graphs["cat_contiguous"][1]):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    return {name: {"median_us": statistics.median(values), "samples_us": values} for name, values in samples.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--trials", type=int, default=11)
    arguments = parser.parse_args()
    print(json.dumps(benchmark(arguments.tokens, arguments.heads, arguments.repeats, arguments.trials), indent=2))
