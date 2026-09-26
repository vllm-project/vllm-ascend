# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Llama 3 native and Ascend fused RoPE using NPU graph replay.

Example: python benchmarks/llama3_rotary_embedding.py --tokens 4 1024
Synthetic BF16 Q/K use Llama 70B geometry. Timings include input copies for
both paths because the fused operator may mutate its input. These are operator
measurements, not end-to-end serving throughput.
"""

import argparse
import json
import statistics

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding

from vllm_ascend.ops.rotary_embedding import AscendLlama3RotaryEmbedding
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


class ReferenceLlama3RotaryEmbedding(Llama3RotaryEmbedding):
    """Bypass exact-name OOT registration when constructing the reference."""


def graph_time_us(fn, iterations):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured_output = fn()
    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    del graph, captured_output
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 1024])
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations <= 0 or any(tokens <= 0 for tokens in args.tokens):
        parser.error("tokens and iterations must be positive")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    init_device_properties_triton()
    with set_current_vllm_config(VllmConfig()):
        config = (128, 128, 32768, 500000, True, torch.bfloat16, 8, 1, 4, 8192)
        reference = ReferenceLlama3RotaryEmbedding(*config).npu()
        candidate = AscendLlama3RotaryEmbedding(*config).npu()
        torch.testing.assert_close(candidate.cos_sin_cache, reference.cos_sin_cache, rtol=0, atol=0)
        for tokens in args.tokens:
            query = torch.randn(tokens, 8192, dtype=torch.bfloat16, device="npu")
            key = torch.randn(tokens, 1024, dtype=torch.bfloat16, device="npu")
            positions = torch.randint(0, 32768, (tokens,), device="npu")

            def native(positions=positions, query=query, key=key):
                return reference.forward_native(positions, query.clone(), key.clone())

            def fused(positions=positions, query=query, key=key):
                return candidate(positions, query.clone(), key.clone())

            expected = reference.forward_static(
                positions, query.float(), key.float(), 128, 128, reference.cos_sin_cache.float(), True
            )
            actual = fused()
            for output, target in zip(actual, expected):
                torch.testing.assert_close(output, target.to(output.dtype), atol=1e-3, rtol=1e-3)
            baseline_us = graph_time_us(native, args.iterations)
            candidate_us = graph_time_us(fused, args.iterations)
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "native_graph_us": baseline_us,
                        "fused_graph_us": candidate_us,
                        "speedup": statistics.median(baseline_us) / statistics.median(candidate_us),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
