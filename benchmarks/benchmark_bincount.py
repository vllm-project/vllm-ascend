# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness-checked single-case entry point for external msprof capture."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.worker.v2.sample.penalties import bincount


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional frozen penalties.py comparator")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--state-rows", type=int, help="Backing rows; defaults to requests plus two inactive sentinels")
    parser.add_argument("--prompt", type=int, default=4096)
    parser.add_argument("--output", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--pattern", choices=["random", "repeat", "same-word"], default="random")
    parser.add_argument("--token", type=int, help="Repeated token ID; defaults to min(7, vocab - 1)")
    parser.add_argument("--task", choices=["profile-all", "profile-op"], default="profile-all")
    parser.add_argument("--warmup-calls", type=int, default=5)
    parser.add_argument("--measured-calls", type=int, default=20)
    args = parser.parse_args()
    if min(args.requests, args.vocab, args.measured_calls) < 1 or min(args.prompt, args.output, args.warmup_calls) < 0:
        parser.error("requests, vocab and measured calls must be positive; lengths and warmup must be nonnegative")
    rows = args.state_rows if args.state_rows is not None else args.requests + 2
    repeated_token = min(7, args.vocab - 1) if args.token is None else args.token
    if rows < args.requests or not 0 <= repeated_token < args.vocab:
        parser.error("state rows must cover requests and the repeated token must be inside the vocabulary")
    entry = bincount
    if args.source:
        spec = importlib.util.spec_from_file_location("_bincount_comparator", args.source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        entry = module.bincount
    source = Path(inspect.getsourcefile(entry)).resolve()
    print(
        json.dumps(
            {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "case": vars(args)},
            default=str,
        ),
        flush=True,
    )
    torch.npu.set_device(0)
    prefill = args.prompt + args.output
    generator = torch.Generator().manual_seed(42)
    tokens = torch.randint(args.vocab, (rows, max(prefill, 1)), generator=generator, dtype=torch.int32)
    if args.pattern == "repeat":
        tokens.fill_(repeated_token)
    elif args.pattern == "same-word":
        tokens[:] = torch.arange(tokens.shape[1]).remainder(min(args.vocab, 32))
    words = (args.vocab + 31) // 32
    expected_mask = torch.full((rows, words), -1, dtype=torch.int32)
    expected_counts = torch.full((rows, args.vocab), -1, dtype=torch.int32)
    for row in range(args.requests):
        expected_mask[row] = 0
        expected_counts[row] = 0
        if args.pattern == "repeat":
            if args.prompt:
                expected_mask[row, repeated_token // 32] = 1 << (repeated_token % 32)
            expected_counts[row, repeated_token] = args.output
        else:
            prompt_ids = tokens[row, : args.prompt].unique().long()
            packed = torch.zeros(words, dtype=torch.int64)
            packed.scatter_add_(0, prompt_ids // 32, 1 << (prompt_ids % 32))
            expected_mask[row] = packed.to(torch.int32)
            expected_counts[row] = torch.bincount(tokens[row, args.prompt : prefill].long(), minlength=args.vocab)
    mapping = torch.arange(args.requests - 1, -1, -1, dtype=torch.int32, device="npu")
    device_tokens = tokens.npu()
    prompt_lengths = torch.full((rows,), args.prompt, dtype=torch.int32, device="npu")
    prefill_lengths = torch.full((rows,), prefill, dtype=torch.int32, device="npu")
    masks = torch.full_like(expected_mask, -1, device="npu")
    counts = torch.full_like(expected_counts, -1, device="npu")
    call_args = mapping, device_tokens, prompt_lengths, prefill_lengths, masks, counts, prefill
    if args.task == "profile-op":
        entry(*call_args)
        torch.npu.synchronize()
    else:
        for _ in range(args.warmup_calls):
            entry(*call_args)
            torch.npu.synchronize()
        print(f"MEASURED_CALLS={args.measured_calls}", flush=True)
        for _ in range(args.measured_calls):
            entry(*call_args)
            torch.npu.synchronize()
    torch.testing.assert_close(masks.cpu(), expected_mask, rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
    print("PROFILE_CORRECTNESS_PASS", flush=True)


if __name__ == "__main__":
    main()
