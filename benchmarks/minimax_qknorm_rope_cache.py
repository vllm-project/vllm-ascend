# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""ACLGraph latency of MiniMax-M3 prepare + cache insertion on one NPU.

Example:
    python benchmarks/minimax_qknorm_rope_cache.py --reference-module /tmp/pr15772.py

The optional reference module is split_qkv_index_rmsnorm_rope.py from Ascend
PR #15772. It is measured with both original cache-insertion calls, including
its external RoPE row gather. The fused kernel reads RoPE rows directly.
Use --model-baseline --separate-index for separate W8A8 QKV / BF16 index
projections. All times cover normalization, RoPE and all three cache writes.
"""

import argparse
import importlib.util
import json
import statistics

import torch
import torch_npu

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.triton.linearnorm.minimax_qknorm_rope_cache import minimax_qknorm_rope_cache
from vllm_ascend.ops.triton.rope import rope_forward_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

WARMUP_ITERATIONS = 5
CAPTURE_ITERATIONS = 20
REPLAY_ITERATIONS = 10
TIMING_SAMPLES = 5


def graph_latency_us(fn):
    for _ in range(WARMUP_ITERATIONS):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(CAPTURE_ITERATIONS):
            fn()
    graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(TIMING_SAMPLES):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(REPLAY_ITERATIONS):
            graph.replay()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (CAPTURE_ITERATIONS * REPLAY_ITERATIONS))
    return statistics.median(samples)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128, 192, 256])
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--index-heads", type=int, default=1)
    parser.add_argument("--cache-blocks", type=int, default=5392)
    parser.add_argument("--reference-module")
    parser.add_argument("--kernel-module")
    parser.add_argument("--separate-index", action="store_true")
    parser.add_argument("--model-baseline", action="store_true")
    parser.add_argument("--profile-dir")
    args = parser.parse_args()
    torch.manual_seed(0)
    enable_custom_op()
    init_device_properties_triton()
    ref_prepare = None
    if args.reference_module:
        spec = importlib.util.spec_from_file_location("minimax_reference", args.reference_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ref_prepare = module.split_qkv_index_rmsnorm_rope_impl
    for tokens in args.tokens:
        benchmark_case(args, tokens, ref_prepare)


def benchmark_case(args, tokens, ref_prepare):
    head_dim, rotary_dim, eps, block_size = 128, 64, 1e-6, 128
    qh, kh, ih = args.q_heads, args.kv_heads, args.index_heads
    packed = torch.randn(tokens, (qh + 2 * kh + ih + 1) * head_dim, dtype=torch.bfloat16, device="npu")
    weights = [torch.randn(head_dim, dtype=packed.dtype, device="npu") * 0.1 for _ in range(4)]
    scaled_weights = [1 + w for w in weights]
    angles = torch.randn(8192, rotary_dim // 2, device="npu")
    cs = torch.cat((angles.cos(), angles.sin()), -1).to(packed.dtype)
    positions = torch.randint(8192, (tokens,), device="npu")
    blocks = max(args.cache_blocks, (tokens * 2 + block_size - 1) // block_size)
    key = torch.zeros(blocks, block_size, kh, head_dim, device="npu", dtype=packed.dtype)
    value = torch.zeros_like(key)
    index = torch.zeros(blocks, block_size, head_dim, device="npu", dtype=packed.dtype)
    slots = torch.randperm(blocks * block_size, device="npu")[:tokens]
    index_slots = torch.randperm(blocks * block_size, device="npu")[:tokens]
    main_width = (qh + 2 * kh) * head_dim
    main_input = packed[:, :main_width].contiguous()
    index_input = packed[:, main_width:].contiguous()
    fused_impl = minimax_qknorm_rope_cache
    if args.kernel_module:
        spec = importlib.util.spec_from_file_location("minimax_candidate", args.kernel_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fused_impl = module.minimax_qknorm_rope_cache

    def model_prepare():
        q, k, v = torch.ops.vllm.qkv_rmsnorm_rope(
            main_input,
            cs,
            positions,
            1 + weights[0],
            1 + weights[1],
            qh * head_dim,
            kh * head_dim,
            head_dim,
            eps,
        )
        iq, ik = index_input.split((ih * head_dim, head_dim), -1)
        iq = DeviceOperator.npu_gemma_rms_norm(iq.contiguous().view(-1, head_dim), weights[2], eps)
        ik = DeviceOperator.npu_gemma_rms_norm(ik.contiguous(), weights[3], eps)
        iq, ik = rope_forward_triton(
            iq.view(tokens, ih, head_dim),
            ik.view(tokens, 1, head_dim),
            cos_sin_cache=cs,
            positions=positions,
            rope_dim=rotary_dim,
        )
        return q, k, v, iq.reshape(tokens, -1), ik.reshape(tokens, -1)

    def separate_prepare():
        parts = list(packed.split([qh * head_dim, kh * head_dim, kh * head_dim, ih * head_dim, head_dim], -1))
        cos, sin = cs[positions].chunk(2, -1)
        cos = torch.cat((cos, cos), -1).view(tokens, 1, 1, rotary_dim)
        sin = torch.cat((sin, sin), -1).view(tokens, 1, 1, rotary_dim)
        for part, weight in zip((0, 1, 3, 4), scaled_weights):
            x = parts[part].reshape(-1, head_dim).contiguous()
            x = torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0].view(tokens, 1, -1, head_dim)
            rotated = torch_npu.npu_rotary_mul(x[..., :rotary_dim].contiguous(), cos, sin)
            parts[part] = torch.cat((rotated, x[..., rotary_dim:]), -1).reshape(tokens, -1)
        return parts

    def baseline():
        if args.model_baseline:
            q, k, v, iq, ik = model_prepare()
        elif ref_prepare is None:
            q, k, v, iq, ik = separate_prepare()
        else:
            q, k, v, iq, ik = ref_prepare(
                packed,
                cs,
                positions,
                *scaled_weights,
                qh * head_dim,
                kh * head_dim,
                ih * head_dim,
                head_dim,
                head_dim,
                eps,
            )
        DeviceOperator.reshape_and_cache(k.view(tokens, kh, head_dim), v.view(tokens, kh, head_dim), key, value, slots)
        torch.ops._C_ascend.npu_scatter_nd_update_sk(index.view(-1, head_dim), index_slots.view(-1, 1), ik)
        return q, iq

    def fused():
        return fused_impl(
            main_input if args.separate_index else packed,
            cs,
            positions,
            *weights,
            key,
            value,
            index,
            slots,
            index_slots,
            tokens,
            qh,
            kh,
            ih,
            eps,
            index_packed=index_input if args.separate_index else None,
        )

    baseline_us = graph_latency_us(baseline)
    fused_us = graph_latency_us(fused)
    print(
        json.dumps(
            {
                "tokens": tokens,
                "q_heads": qh,
                "kv_heads": kh,
                "index_heads": ih,
                "cache_blocks": blocks,
                "separate_index": args.separate_index,
                "baseline_us": baseline_us,
                "fused_us": fused_us,
                "speedup": baseline_us / fused_us,
                "baseline": "model" if args.model_baseline else ("pr15772" if ref_prepare else "separate_ops"),
            }
        ),
        flush=True,
    )
    if args.profile_dir:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"{args.profile_dir}/{tokens}"),
            record_shapes=True,
        ):
            with torch.profiler.record_function("baseline_prepare_insert"):
                baseline()
            with torch.profiler.record_function("fused_prepare_insert"):
                fused()
            torch.npu.synchronize()


if __name__ == "__main__":
    main()
