# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project
"""Compare the vLLM GDN QKV rearrange with the AscendC operator."""

import argparse
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

from vllm_ascend.ops.rearrange_qkv import rearrange_mixed_qkv
from vllm_ascend.utils import enable_custom_op


def measure(fn, warmup, iterations, use_graph):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    if use_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = fn()
        run = graph.replay
    else:
        run = fn

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run()
    end.record()
    end.synchronize()
    if use_graph:
        del output
    return start.elapsed_time(end) * 1000 / iterations


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 16, 64, 256, 1024, 4096])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--q-dim", type=int, default=1024, help="Per-rank Q/K width")
    parser.add_argument("--v-dim", type=int, default=3072, help="Per-rank V width")
    parser.add_argument("--head-k-dim", type=int, default=128)
    parser.add_argument("--head-v-dim", type=int, default=128)
    args = parser.parse_args()

    enable_custom_op()
    layer = SimpleNamespace(
        key_dim=args.q_dim,
        value_dim=args.v_dim,
        tp_size=1,
        head_k_dim=args.head_k_dim,
        head_v_dim=args.head_v_dim,
    )
    layer.rearrange_mixed_qkv = lambda x: QwenGatedDeltaNetAttention.rearrange_mixed_qkv(layer, x)

    print("tokens,q_dim,v_dim,original_us,ascendc_us,speedup")
    for tokens in args.tokens:
        mixed_qkv = torch.randn(tokens, 2 * args.q_dim + args.v_dim, device="npu", dtype=torch.bfloat16)
        original = lambda mixed_qkv=mixed_qkv: layer.rearrange_mixed_qkv(mixed_qkv)
        ascendc = lambda mixed_qkv=mixed_qkv: rearrange_mixed_qkv(layer, mixed_qkv)
        for actual, expected in zip(ascendc(), original()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        original_us = measure(original, args.warmup, args.iters, args.graph)
        ascendc_us = measure(ascendc, args.warmup, args.iters, args.graph)
        print(f"{tokens},{args.q_dim},{args.v_dim},{original_us:.4f},{ascendc_us:.4f},{original_us / ascendc_us:.3f}")


if __name__ == "__main__":
    main()
