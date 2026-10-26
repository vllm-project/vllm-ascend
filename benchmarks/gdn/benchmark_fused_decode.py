# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare the complete existing GDN decode path with the fused A3 kernel.

Run from the repository root with the matching FLA NPU wheel installed:
    python benchmarks/gdn/benchmark_fused_decode.py
"""

import argparse
import json
import statistics

import torch
import torch_npu
from fla_npu.ops.ascendc import recurrent_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend.ops.triton.fused_gdn_decode import fused_gdn_decode
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def measure(fn, iterations, graph_unroll):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    allocated = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(graph_unroll):
            output = fn()
    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (iterations * graph_unroll))
    extra_memory = torch.npu.max_memory_allocated() - allocated
    return statistics.median(samples), extra_memory, output


def run_case(batch, args):
    torch.manual_seed(42)
    h, hv, dim = args.heads, args.value_heads, 128
    options = {"device": "npu", "dtype": torch.bfloat16}
    q = torch.randn(batch, h, dim, **options)
    k = torch.randn_like(q)
    v = torch.randn(batch, hv, dim, **options)
    a, b = torch.randn(batch, hv, **options), torch.randn(batch, hv, **options)
    log = torch.randn(hv, device="npu")
    bias = torch.randn(hv, device="npu")
    initial = torch.randn(batch + 1, hv, dim, dim, device="npu") * 0.1
    baseline_state, fused_state = initial.clone(), initial.clone()
    indices = torch.arange(batch, device="npu", dtype=torch.int32)
    lengths = torch.ones(batch + 1, device="npu", dtype=torch.int32)
    lengths[0] = 0
    boundaries = torch.arange(batch + 1, device="npu", dtype=torch.int32)

    def baseline():
        g, beta = fused_gdn_gating_patch(log, a, b, bias)
        return recurrent_gated_delta_rule(
            l2norm_fwd(q),
            l2norm_fwd(k),
            v,
            baseline_state,
            g=g.squeeze(0),
            beta=beta.squeeze(0),
            scale=dim**-0.5,
            actual_seq_lengths=lengths,
            ssm_state_indices=indices,
        )

    def fused():
        return fused_gdn_decode(q, k, v, a, b, log, bias, fused_state, indices, boundaries)

    expected, actual = baseline(), fused()
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=1e-2)
    torch.testing.assert_close(fused_state, baseline_state, atol=1e-4, rtol=1e-3)
    baseline_us, baseline_memory, _ = measure(baseline, args.iterations, args.graph_unroll)
    fused_us, fused_memory, _ = measure(fused, args.iterations, args.graph_unroll)
    print(
        json.dumps(
            {
                "batch": batch,
                "baseline_us": baseline_us,
                "fused_us": fused_us,
                "speedup": baseline_us / fused_us,
                "baseline_extra_bytes": baseline_memory,
                "fused_extra_bytes": fused_memory,
                "state_mib": initial.numel() * initial.element_size() / 1024**2,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--graph-unroll", type=int, default=16, help="Calls per graph to amortize Python graph-replay overhead"
    )
    args = parser.parse_args()
    torch.npu.set_device(0)
    init_device_properties_triton()
    print(
        json.dumps(
            {
                "device": torch.npu.get_device_name(0),
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "args": vars(args),
            }
        )
    )
    for batch in args.batches:
        run_case(batch, args)


if __name__ == "__main__":
    main()
