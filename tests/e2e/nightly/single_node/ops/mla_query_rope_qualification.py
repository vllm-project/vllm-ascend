# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU gate for the C8 query component: zero descales, views, graph, and timing."""

import argparse
import json
import os
from pathlib import Path

import torch

_CUSTOM_OPP_PATH = os.environ.get("ASCEND_CUSTOM_OPP_PATH")

from flash_mla_c8_qualification import C8Case, bench_graph, error  # noqa: E402

from vllm_ascend.ops.triton.mla_query_rope import scale_mla_query_rope  # noqa: E402

# Framework import registers its default package. Preserve the test's explicit
# native overlay precedence before the first C8 ACLNN call.
if _CUSTOM_OPP_PATH is not None:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _CUSTOM_OPP_PATH


def tensors(tokens, heads, strided=False, trailing_scale=False):
    generator = torch.Generator().manual_seed(tokens + heads)
    raw = torch.randn((tokens, heads, 64), generator=generator).bfloat16()
    sq_cpu = torch.rand((tokens, heads), generator=generator) * 0.035 + 0.005
    if strided:
        backing = torch.full((tokens * 2, heads, 128), -9, dtype=torch.bfloat16, device="npu")
        rope = backing[::2, :, 1::2]
        scale_backing = torch.full((tokens * 2, heads * 2), -9.0, device="npu")
        sq = scale_backing[::2, ::2]
    else:
        rope = torch.empty_like(raw, device="npu")
        sq = torch.empty_like(sq_cpu, device="npu")
    rope.copy_(raw)
    sq.copy_(sq_cpu)
    return raw, rope, sq.unsqueeze(-1) if trailing_scale else sq, sq_cpu


def reference(raw, scales, kv_scale):
    safe = torch.where(scales == 0, 1.0, scales)
    return (raw.float() / safe[..., None] / kv_scale).bfloat16(), safe


def verify(raw, out, sq, scales, sk):
    expected, safe = reference(raw, scales, sk)
    actual = out.cpu()
    torch.testing.assert_close(sq.cpu().reshape(safe.shape), safe, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    # The fused FP32 divide may round a tie differently before its BF16 store.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.0079, atol=1e-5)
    return error(actual, expected)


def operator_gate(tokens, heads, strided, trailing_scale):
    raw, rope, sq, scales = tensors(tokens, heads, strided, trailing_scale)
    scales.reshape(-1)[::7] = 0
    sq.copy_(scales.reshape(sq.shape))
    sk = torch.tensor([0.015625], device="npu")
    out = scale_mla_query_rope(rope, sq, sk)
    err = verify(raw, out, sq, scales, 0.015625)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = scale_mla_query_rope(rope, sq, sk)
    # Change runtime scales and input values without changing captured addresses.
    for all_zero, kv in ((True, 0.03125), (False, 0.02)):
        current = torch.zeros_like(scales) if all_zero else scales
        sq.copy_(current.reshape(sq.shape))
        sk.fill_(kv)
        rope.copy_(-raw)
        graph.replay()
        torch.npu.synchronize()
        verify(-raw, out, sq, current, kv)
    return {
        "gate": "zero_stride_graph",
        "tokens": tokens,
        "heads": heads,
        "strided": strided,
        "trailing_scale": trailing_scale,
        "error": err,
    }


def attention_gate(heads):
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    case = C8Case([127, 129], heads)
    raw = torch.randn(case.qr_cpu.shape, generator=torch.Generator().manual_seed(91)).bfloat16()
    case.sq_cpu[::2] = 0
    latent = case.q_cpu.float()
    latent[::2] = 0
    case.q_cpu = latent.to(torch.float8_e4m3fn)
    case.q.copy_(case.q_cpu)
    case.sq.copy_(case.sq_cpu)
    case.qr = scale_mla_query_rope(raw.to("npu"), case.sq, case.sk)
    case.qr_cpu, case.sq_cpu = case.qr.cpu(), case.sq.cpu()
    rows = []
    for causal in (False, True):
        result = case.check(causal=causal)
        rows.append({"gate": "c8_zero_latent_nonzero_bf16_component", **result})
    return rows


def benchmark(tokens, heads):
    _, rope, sq, _ = tensors(tokens, heads, strided=True)
    sk = torch.tensor([0.015625], device="npu")
    baseline = bench_graph(lambda: (rope / sq.unsqueeze(-1) / sk).to(torch.bfloat16))
    fused = bench_graph(lambda: scale_mla_query_rope(rope, sq, sk))
    return {
        "gate": "graph_latency",
        "tokens": tokens,
        "heads": heads,
        "baseline_two_div_cast": baseline,
        "fused": fused,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.set_num_threads(8)
    rows = []

    def record(row):
        rows.append(row)
        args.json.write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)

    shapes = ((32, 12), (64, 96), (768, 12), (768, 96), (1, 12))
    for tokens, heads in shapes:
        for strided, trailing in ((False, False), (True, True)):
            record(operator_gate(tokens, heads, strided, trailing))
    for heads in (12, 96):
        for row in attention_gate(heads):
            record(row)
    if args.benchmark:
        for tokens, heads in shapes[:-1]:
            record(benchmark(tokens, heads))


if __name__ == "__main__":
    main()
