# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project
"""Benchmark the original and AscendC GDN QKV rearrange with NPU Graph.

Each implementation is captured into its own ``torch.npu.NPUGraph``.  Timing
contains graph replay only and excludes graph capture.
"""

import argparse
import statistics
import time
from collections.abc import Callable
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)

from vllm_ascend.ops.rearrange_qkv import rearrange_mixed_qkv
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_custom_op,
    get_ascend_device_type,
)


def _capture(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.npu.NPUGraph, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fn()
    return graph, output


def _measure_replay(
    replay: Callable[[], None],
    iterations: int,
) -> tuple[float, float]:
    """Return NPU-event and wall-clock latency in microseconds per replay."""
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        replay()
    end.record()
    end.synchronize()
    wall_end = time.perf_counter()

    event_us = start.elapsed_time(end) * 1000 / iterations
    wall_us = (wall_end - wall_start) * 1_000_000 / iterations
    return event_us, wall_us


def _benchmark_pair(
    original_graph: torch.npu.NPUGraph,
    ascendc_graph: torch.npu.NPUGraph,
    warmup: int,
    iterations: int,
    rounds: int,
) -> tuple[float, float, float, float]:
    for _ in range(warmup):
        original_graph.replay()
        ascendc_graph.replay()
    torch.npu.synchronize()

    original_event_samples: list[float] = []
    original_wall_samples: list[float] = []
    ascendc_event_samples: list[float] = []
    ascendc_wall_samples: list[float] = []

    # Alternate the order to reduce temperature and run-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            original_result = _measure_replay(
                original_graph.replay,
                iterations,
            )
            ascendc_result = _measure_replay(
                ascendc_graph.replay,
                iterations,
            )
        else:
            ascendc_result = _measure_replay(
                ascendc_graph.replay,
                iterations,
            )
            original_result = _measure_replay(
                original_graph.replay,
                iterations,
            )

        original_event_samples.append(original_result[0])
        original_wall_samples.append(original_result[1])
        ascendc_event_samples.append(ascendc_result[0])
        ascendc_wall_samples.append(ascendc_result[1])

    return (
        statistics.median(original_event_samples),
        statistics.median(ascendc_event_samples),
        statistics.median(original_wall_samples),
        statistics.median(ascendc_wall_samples),
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[1, 4, 16, 64, 256, 1024, 4096],
    )
    parser.add_argument("--q-dim", type=int, default=1024)
    parser.add_argument("--v-dim", type=int, default=3072)
    parser.add_argument("--head-k-dim", type=int, default=128)
    parser.add_argument("--head-v-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    enable_custom_op()

    if get_ascend_device_type() not in {
        AscendDeviceType.A2,
        AscendDeviceType.A3,
    }:
        raise RuntimeError("npu_rearrange_qkv is only built for A2 and A3")
    if not hasattr(torch.ops._C_ascend, "npu_rearrange_qkv"):
        raise RuntimeError(
            "torch.ops._C_ascend.npu_rearrange_qkv is unavailable; rebuild and reinstall vllm-ascend first"
        )

    layer = SimpleNamespace(
        key_dim=args.q_dim,
        value_dim=args.v_dim,
        tp_size=1,
        head_k_dim=args.head_k_dim,
        head_v_dim=args.head_v_dim,
    )
    layer.rearrange_mixed_qkv = lambda x: QwenGatedDeltaNetAttention.rearrange_mixed_qkv(layer, x)

    print(
        "tokens,q_dim,v_dim,"
        "original_graph_event_us,ascendc_graph_event_us,event_speedup,"
        "original_graph_wall_us,ascendc_graph_wall_us,wall_speedup"
    )
    for tokens in args.tokens:
        mixed_qkv = torch.randn(
            tokens,
            2 * args.q_dim + args.v_dim,
            dtype=torch.bfloat16,
            device=f"npu:{args.device}",
        )

        original = lambda mixed_qkv=mixed_qkv: layer.rearrange_mixed_qkv(mixed_qkv)
        ascendc = lambda mixed_qkv=mixed_qkv: rearrange_mixed_qkv(layer, mixed_qkv)

        # Initialize kernels and allocator state before graph capture.
        for _ in range(args.warmup):
            original()
            ascendc()
        torch.npu.synchronize()

        original_graph, original_output = _capture(original)
        ascendc_graph, ascendc_output = _capture(ascendc)
        torch.npu.synchronize()

        original_graph.replay()
        ascendc_graph.replay()
        torch.npu.synchronize()
        for actual, expected in zip(ascendc_output, original_output):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        original_event_us, ascendc_event_us, original_wall_us, ascendc_wall_us = _benchmark_pair(
            original_graph,
            ascendc_graph,
            warmup=args.warmup,
            iterations=args.iters,
            rounds=args.rounds,
        )

        print(
            f"{tokens},{args.q_dim},{args.v_dim},"
            f"{original_event_us:.4f},{ascendc_event_us:.4f},"
            f"{original_event_us / ascendc_event_us:.3f},"
            f"{original_wall_us:.4f},{ascendc_wall_us:.4f},"
            f"{original_wall_us / ascendc_wall_us:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
