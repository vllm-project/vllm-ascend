# SPDX-License-Identifier: Apache-2.0
"""Synthetic A3 SFA microbenchmark; does not measure model TPOT."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op


def make_inputs(c8: bool, rows: int):
    torch.manual_seed(0)
    kv_len, block_size, heads, dim, rope_dim, topk = 4096, 128, 64, 512, 64, 2048
    query = torch.randn(rows, heads, dim, dtype=torch.bfloat16).npu()
    query_rope = torch.randn(rows, heads, rope_dim, dtype=torch.bfloat16).npu()
    key = torch.randn(kv_len // block_size, block_size, 1, dim, dtype=torch.bfloat16).npu()
    key_rope = torch.randn(*key.shape[:-1], rope_dim, dtype=torch.bfloat16).npu()
    inputs = dict(
        query=query,
        key=key,
        value=key,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_indices=torch.full((rows, 1, topk), -1, dtype=torch.int32).npu(),
        block_table=torch.arange(kv_len // block_size, dtype=torch.int32).repeat(rows, 1).npu(),
        actual_seq_lengths_query=torch.arange(1, rows + 1, dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.full((rows,), kv_len, dtype=torch.int32).npu(),
        scale_value=(dim + rope_dim) ** -0.5,
        sparse_block_size=1,
        sparse_mode=0,
        attention_mode=2,
        layout_query="TND",
        layout_kv="PA_BSND",
        return_softmax_lse=True,
    )
    op = torch.ops._C_ascend.npu_sparse_flash_attention
    if c8:
        quantized = (key.float() * 32).round().clamp(-128, 127).to(torch.int8)
        scales = torch.full((*key.shape[:-1], dim // 128), 1 / 32, dtype=torch.float32, device=key.device)
        packed = torch.cat((quantized, key_rope.contiguous().view(torch.int8), scales.view(torch.int8)), dim=-1)
        inputs.update(
            query=torch.cat((query, query_rope), dim=-1),
            key=packed,
            value=packed,
            key_quant_mode=2,
            value_quant_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=rope_dim,
        )
        del inputs["query_rope"], inputs["key_rope"]
        op = torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention
    indices = torch.stack([torch.randperm(kv_len, dtype=torch.int32)[:topk] for _ in range(rows)]).unsqueeze(1).npu()
    return op, inputs, indices


def measure(op, inputs, warmup, iterations):
    for _ in range(warmup):
        op(**inputs)
    torch.npu.synchronize()
    timings = []
    for _ in range(iterations):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        op(**inputs)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000)
    return statistics.median(timings)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c8", action="store_true")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--label", required=True, help="Source/build revision identifying this run")
    parser.add_argument("--output", type=Path, required=True, help="Save timings, input hashes and CPU output tensors")
    parser.add_argument(
        "--reference", type=Path, help="Compare with results from a separately installed baseline build"
    )
    args = parser.parse_args()
    if args.rows <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("rows and iterations must be positive; warmup must be nonnegative")
    if not get_current_hardware_profile().supports(HardwareCapability.SFA_C8_DCP_REPLICATED_INDEXER):
        parser.error("requires the A2/A3 custom SFA operators; use A3 to measure prefix trimming")
    torch_npu.npu.config.allow_internal_format = True
    enable_custom_op()
    op, inputs, original = make_inputs(args.c8, args.rows)
    reference = torch.load(args.reference, map_location="cpu", weights_only=True) if args.reference else None
    results = {"label": args.label, "c8": args.c8, "rows": args.rows, "cases": []}
    if reference is not None and (reference["c8"] != args.c8 or reference["rows"] != args.rows):
        parser.error("reference must have the same c8 and rows settings")
    for count in (0, 1, 128, 512, 513, 2048):
        inputs["sparse_indices"].copy_(original)
        inputs["sparse_indices"][..., count:] = -1
        digest = hashlib.sha256()
        for name, value in sorted(inputs.items()):
            digest.update(name.encode())
            if isinstance(value, torch.Tensor):
                tensor = value.cpu().contiguous()
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.view(torch.uint8).numpy().tobytes())
            else:
                digest.update(str(value).encode())
        outputs = [x.cpu() for x in op(**inputs)]
        old = None
        if reference is not None:
            old = next(c for c in reference["cases"] if c["valid_count"] == count)
            if old["input_sha256"] != digest.hexdigest():
                raise ValueError("Inputs differ from baseline; timings and outputs cannot be compared")
            for i, (actual, expected) in enumerate(zip(outputs, old["outputs"])):
                tol = 1e-2 if i == 0 else 1e-5
                torch.testing.assert_close(actual.float(), expected.float(), atol=tol, rtol=tol)
            actual_lse = outputs[1].float() + outputs[2].float().log()
            expected_lse = old["outputs"][1].float() + old["outputs"][2].float().log()
            torch.testing.assert_close(actual_lse, expected_lse, atol=1e-5, rtol=1e-5)
        latency = measure(op, inputs, args.warmup, args.iterations)
        case = dict(valid_count=count, input_sha256=digest.hexdigest(), median_us=latency)
        if old is not None:
            case.update(baseline_us=old["median_us"], speedup=old["median_us"] / latency)
        print(json.dumps(dict(label=args.label, c8=args.c8, rows=args.rows, **case)))
        results["cases"].append(dict(**case, outputs=outputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.output)


if __name__ == "__main__":
    main()
