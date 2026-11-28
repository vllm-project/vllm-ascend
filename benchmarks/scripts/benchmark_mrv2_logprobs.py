# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare the MRV2 logprob kernels to a baseline file from git show."""

import argparse
import importlib.util
import json
import time
from functools import partial
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.sample import logprob as candidate

SHAPES = (
    (1, 32000, 1),
    (1, 65535, 6),
    (1, 65536, 6),
    (1, 151936, 6),
    (4, 151936, 21),
    (8, 128256, 21),
    (32, 151937, 6),
    (64, 151936, 65),
)


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("baseline_logprob", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import the baseline from {path}")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    return baseline


def measure(call, iterations):
    start = torch.npu.Event(enable_timing=True)
    stop = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    host_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        call()
    stop.record()
    stop.synchronize()
    return {
        "device_us": start.elapsed_time(stop) * 1000 / iterations,
        "host_and_device_us": (time.perf_counter() - host_start) * 1_000_000 / iterations,
    }


def validate(output, reference, kind, logits, token_ids):
    if kind == "topk":
        torch.testing.assert_close(output.logprob_token_ids, reference.logprob_token_ids, atol=0, rtol=0)
        torch.testing.assert_close(output.selected_token_ranks, reference.selected_token_ranks, atol=0, rtol=0)
        token_ids = reference.logprob_token_ids
        sampled_logits = logits.gather(1, token_ids[:, :1])
        ranks = (logits >= sampled_logits).sum(dim=1)
        torch.testing.assert_close(output.selected_token_ranks, ranks, atol=0, rtol=0)
        actual, baseline = output.logprobs, reference.logprobs
    else:
        actual, baseline = output, reference
    actual_cpu, baseline_cpu = actual.cpu(), baseline.cpu()
    golden = torch.log_softmax(logits.cpu().float(), dim=1).gather(1, token_ids.cpu())
    torch.testing.assert_close(actual_cpu, baseline_cpu, atol=1e-5, rtol=0)
    torch.testing.assert_close(actual_cpu, golden, atol=1e-5, rtol=0)
    return {
        "max_abs_vs_baseline": (actual_cpu - baseline_cpu).abs().max().item(),
        "max_abs_vs_reference": (actual_cpu - golden).abs().max().item(),
    }


def profile_pair(calls, output_dir):
    profiler = torch_npu.profiler
    for name, call in calls.items():
        trace_dir = output_dir / name
        trace_dir.mkdir(parents=True, exist_ok=True)
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            on_trace_ready=profiler.tensorboard_trace_handler(str(trace_dir)),
            experimental_config=profiler._ExperimentalConfig(
                profiler_level=profiler.ProfilerLevel.Level1,
                aic_metrics=profiler.AiCMetrics.PipeUtilization,
            ),
        ):
            for _ in range(12):
                call()
            torch.npu.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 3:
        parser.error("use positive iterations and at least three repeats")

    torch.manual_seed(20261126)
    init_device_properties_triton()
    baseline = load_baseline(args.baseline)
    print(json.dumps({"device": torch.npu.get_device_name(), "torch": torch.__version__}), flush=True)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for batch_size, vocab_size, topk in SHAPES:
            logits = torch.randn((batch_size, vocab_size), device="npu", dtype=dtype)
            ids = torch.randint(vocab_size, (batch_size, topk), device="npu")
            sampled_ids = ids[:, 0].contiguous()
            for kind in ("token", "topk"):
                if kind == "token":
                    calls = {
                        "baseline": partial(baseline.compute_token_logprobs, logits, ids),
                        "candidate": partial(candidate.compute_token_logprobs, logits, ids),
                    }
                else:
                    calls = {
                        "baseline": partial(baseline.compute_topk_logprobs, logits, topk - 1, sampled_ids),
                        "candidate": partial(candidate.compute_topk_logprobs, logits, topk - 1, sampled_ids),
                    }
                accuracy = validate(calls["candidate"](), calls["baseline"](), kind, logits, ids)
                for call in calls.values():
                    for _ in range(10):
                        call()
                samples = {"baseline": [], "candidate": []}
                for repeat in range(args.repeats):
                    order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
                    for name in order:
                        samples[name].append(measure(calls[name], args.iterations))
                print(
                    json.dumps(
                        {
                            "kind": kind,
                            "dtype": str(dtype),
                            "batch_size": batch_size,
                            "vocab_size": vocab_size,
                            "num_logprobs": topk,
                            "accuracy": accuracy,
                            "samples": samples,
                        }
                    ),
                    flush=True,
                )
                if args.profile_dir and dtype == torch.float32 and (batch_size, vocab_size) == (4, 151936):
                    profile_pair(calls, args.profile_dir / kind)


if __name__ == "__main__":
    main()
