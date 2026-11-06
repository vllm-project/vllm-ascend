# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Compare ForeachAdd + native HcPost with their fusion on an idle A3 device."""

import argparse
import json
import statistics
from functools import partial

import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.hc_post import _hc_post_add_kernel, fused_hc_post_add
from vllm_ascend.utils import enable_custom_op


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def native(shared, routed, residual, post, comb):
    x = torch._foreach_add([shared], [routed])[0]
    return torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
    ).squeeze(0)


def tiled(shared, routed, residual, post, comb, *, block):
    """Sweep the existing kernel's tile without changing its arithmetic."""
    tokens, _, dim = residual.shape
    output = torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)
    _hc_post_add_kernel[(tokens, (dim + block - 1) // block)](
        shared,
        routed,
        residual,
        post,
        comb,
        output,
        shared.stride(0),
        routed.stride(0),
        residual.stride(0),
        residual.stride(1),
        post.stride(0),
        comb.stride(0),
        comb.stride(1),
        DIM=dim,
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def capture(fn, banks):
    for values in banks:
        fn(*values)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        outputs = [fn(*values) for values in banks]
    # Capture records work without guaranteeing initialized output buffers.
    # Replay before checking numerical equality of the two captured graphs.
    graph.replay()
    torch.npu.synchronize()
    return graph, outputs


def measure(graph, banks, repeats):
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (banks * repeats)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=positive_int, nargs="+", default=[10, 12])
    parser.add_argument("--dim", type=positive_int, default=4096)
    parser.add_argument("--banks", type=positive_int, default=40)
    parser.add_argument("--groups", type=positive_int, default=7)
    parser.add_argument("--replays", type=positive_int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        choices=[256, 512, 1024, 2048],
        help="also compare tile sizes against the current fused kernel",
    )
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    if not get_current_hardware_profile().supports(HardwareCapability.HC_POST_ADD_SEPARATE_ROUNDING):
        raise RuntimeError("this benchmark requires A3")
    enable_custom_op()
    torch.manual_seed(1024)
    for tokens in args.tokens:
        banks = [
            (
                torch.randn(tokens, args.dim, device="npu", dtype=torch.bfloat16),
                torch.randn(tokens, args.dim, device="npu", dtype=torch.bfloat16),
                torch.randn(tokens, 4, args.dim, device="npu", dtype=torch.bfloat16),
                torch.rand(tokens, 4, device="npu", dtype=torch.float32),
                torch.rand(tokens, 4, 4, device="npu", dtype=torch.float32),
            )
            for _ in range(args.banks)
        ]
        functions = {"baseline": native, "fused": fused_hc_post_add}
        for block in args.blocks or []:
            functions[f"block{block}"] = partial(tiled, block=block)
        graphs = {}
        reference = None
        for name, fn in functions.items():
            graph, outputs = capture(fn, banks)
            if reference is None:
                reference = outputs
            else:
                for a, b in zip(reference, outputs):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            graphs[name] = graph
        for _ in range(3):
            for graph in graphs.values():
                graph.replay()
        samples = {name: [] for name in graphs}
        for group in range(args.groups):
            # Alternate order to reduce clock/temperature ordering bias.
            order = list(graphs.items())
            for name, graph in order if group % 2 == 0 else reversed(order):
                samples[name].append(measure(graph, args.banks, args.replays))
        baseline = statistics.median(samples["baseline"])
        fused = statistics.median(samples["fused"])
        print(
            json.dumps(
                {
                    "tokens": tokens,
                    "dim": args.dim,
                    "banks": args.banks,
                    "baseline_us": baseline,
                    "fused_us": fused,
                    "kernel_improvement_pct": (1 - fused / baseline) * 100,
                    "samples_us": samples,
                    "tile_sweep": [
                        {
                            "block": block,
                            "median_us": statistics.median(samples[f"block{block}"]),
                            "vs_current_fused_pct": (1 - statistics.median(samples[f"block{block}"]) / fused) * 100,
                        }
                        for block in args.blocks or []
                    ],
                    "scope": "single-rank graph microbenchmark; not service TPOT",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
