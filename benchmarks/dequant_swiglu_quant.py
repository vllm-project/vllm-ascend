# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused dequant_swiglu_quant custom op.

Run: python benchmarks/dequant_swiglu_quant.py

Production callers (DeepSeek W8A8 dynamic quant):
  * shared experts   : x=int32 [tokens, 2H], no group_index, swiglu_mode=1
                       (AscendSharedExperts._run_a8_int_mlp, every decoder
                       layer, prefill + decode)
  * MoE MC2 fallback : x=int32 [tokens, 2H], group_index=int64 cumsum,
                       swiglu_mode=0 (w8a8_dynamic.apply_gmm1_act_quant)

Times only torch.ops._C_ascend.npu_dequant_swiglu_quant (the op under
optimization) against an unfused reference composition. Times exclude
compilation, graph capture and host tensor allocation. Repetition counts
adapt to a warmup measurement so slow shapes stay bounded.
"""

import argparse
import json
import statistics

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def graph_latency_us(fn, *args):
    for _ in range(3):
        fn(*args)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    fn(*args)
    end.record()
    end.synchronize()
    estimate_ms = max(start.elapsed_time(end), 0.001)
    # Capture at most about 10 ms of work and measure about 50 ms per sample.
    batch = min(32, max(1, int(10 / estimate_ms)))
    repeats = min(20, max(1, int(50 / (batch * estimate_ms))))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local",
                         auto_dispatch_capture=True):
        for _ in range(batch):
            output = fn(*args)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (batch * repeats))
    # Keep the captured outputs alive until timing completes.
    del output
    return statistics.median(samples)


def make_inputs(tokens, x_last_dim, groups, device="npu"):
    """x is the int32 output of the W8A8 gate_up matmul."""
    torch.manual_seed(7)
    x = torch.randint(-1000, 1000, (tokens, x_last_dim), dtype=torch.int32,
                      device=device)
    if groups is None:
        weight_scale = (torch.randn(x_last_dim, dtype=torch.float32,
                                    device=device) * 0.001)
    else:
        # The group path requires weight_scale [group_num, 2H].
        weight_scale = (torch.randn(groups, x_last_dim, dtype=torch.float32,
                                    device=device) * 0.001)
    activation_scale = (torch.rand(tokens, 1, dtype=torch.float32,
                                   device=device) * 4 + 0.5)
    group_index = None
    if groups is not None:
        # Rounded per-group row counts summing to `tokens', mirroring
        # cumsum_group_list(group_list, group_list_type, 1) in w8a8_dynamic
        # (the op expects per-group sizes, not a cumsum).
        base = tokens // groups
        sizes = [base] * groups
        for i in range(tokens - base * groups):
            sizes[i] += 1
        group_index = torch.tensor(sizes, dtype=torch.int64,
                                   device=device)
    return x, weight_scale, activation_scale, group_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-last-dim", type=int, nargs="+",
                        default=[4096, 2048, 1024, 512],
                        help="x last dim (2H). 4096 = DeepSeek shared expert "
                             "at TP1; halves as TP grows.")
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[1, 4, 16, 64, 128, 512, 1024, 2048, 4096])
    parser.add_argument("--swiglu-mode", type=int, default=1,
                        help="1 = shared-experts path (SwiGluGate), "
                             "0 = MC2 fallback (plain SwiGLU)")
    parser.add_argument("--clamp-limit", type=float, default=0.0,
                        help="DeepSeek V3 uses 0.0 (no clamp); "
                             "V3.2-style dst quant uses 7.0")
    parser.add_argument("--groups", type=int, default=None,
                        help="enable the MoE group_index path with this many "
                             "groups (requires tokens >= groups)")
    args = parser.parse_args()

    for x_last_dim in args.x_last_dim:
        half = x_last_dim // 2
        for tokens in args.tokens:
            if args.groups is not None and tokens < args.groups:
                continue
            x, weight_scale, activation_scale, group_index = make_inputs(
                tokens, x_last_dim, args.groups)
            # Pre-expand per-row weight scales for the group path outside the
            # captured region (repeat_interleave with a device count syncs).
            if group_index is not None:
                ws_rows = weight_scale.repeat_interleave(group_index, dim=0)
            else:
                ws_rows = None

            def fused(x, weight_scale, activation_scale, group_index):
                return torch.ops._C_ascend.npu_dequant_swiglu_quant(
                    x=x,
                    weight_scale=weight_scale,
                    activation_scale=activation_scale,
                    bias=None,
                    quant_scale=None,
                    quant_offset=None,
                    group_index=group_index,
                    activate_left=True,
                    quant_mode=1,
                    swiglu_mode=args.swiglu_mode,
                    clamp_limit=args.clamp_limit,
                    glu_alpha=1.0,
                    glu_bias=0.0)

            def reference(x, weight_scale, activation_scale, group_index):
                # Unfused composition producing the same outputs: eager
                # dequant + SwiGLU + the CANN built-in dynamic quant.
                ws = ws_rows if ws_rows is not None \
                    else weight_scale.reshape(1, -1)
                f = x.float() * ws * activation_scale.reshape(-1, 1)
                gate = f[..., :half]
                up = f[..., half:]
                if args.swiglu_mode == 1:
                    if args.clamp_limit > 0.0:
                        gate = gate.clamp(max=args.clamp_limit)
                        up = up.clamp(min=-args.clamp_limit,
                                      max=args.clamp_limit)
                    swiglu = F.silu(gate) * up
                else:
                    swiglu = F.silu(gate) * up
                # npu_dynamic_quant (and the fused kernel's cast chain:
                # fp32 -> int32 RINT -> half ROUND -> int8 TRUNC) quantize
                # from a bf16-rounded value.
                swiglu = swiglu.to(torch.bfloat16)
                return torch_npu.npu_dynamic_quant(swiglu)

            us = graph_latency_us(fused, x, weight_scale, activation_scale,
                                  group_index)
            ref_us = graph_latency_us(reference, x, weight_scale,
                                      activation_scale, group_index)
            print(json.dumps({
                "op": "dequant_swiglu_quant",
                "path": "group" if args.groups is not None else "nogroup",
                "swiglu_mode": args.swiglu_mode,
                "clamp_limit": args.clamp_limit,
                "groups": args.groups,
                "x_last_dim": x_last_dim,
                "tokens": tokens,
                "us": round(us, 3),
                "ref_us": round(ref_us, 3),
            }))


if __name__ == "__main__":
    main()
