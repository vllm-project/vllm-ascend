# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure MRV2 bitmask NPU kernel time; optionally load a baseline source file."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch_npu
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.v2.apply_grammar_bitmask import _apply_grammar_bitmask_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    kernel = _apply_grammar_bitmask_kernel
    if args.kernel_source:
        spec = importlib.util.spec_from_file_location("bitmask_baseline", args.kernel_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        kernel = module._apply_grammar_bitmask_kernel
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    torch.manual_seed(14160)
    for rows in [1, 16, 256]:
        for vocab in [32000, 152064, 256001]:
            words = triton.cdiv(vocab, 32)
            for pattern in ["allowed", "blocked", "random", "mostly_allowed"]:
                masks = torch.randint(-(2**31), 2**31 - 1, (rows, words), dtype=torch.int32)
                if pattern == "allowed":
                    masks.fill_(-1)
                elif pattern == "blocked":
                    masks.zero_()
                elif pattern == "mostly_allowed":
                    masks = ~(torch.ones_like(masks) << torch.randint(0, 32, masks.shape, dtype=torch.int32))
                masks = masks.npu()
                logits = torch.randn(rows, vocab, device="npu")
                mapping = torch.arange(rows, dtype=torch.int32, device="npu")
                offsets = torch.arange(rows + 1, dtype=torch.int32, device="npu")

                def run(
                    rows=rows, vocab=vocab, logits=logits, mapping=mapping, offsets=offsets, masks=masks, words=words
                ):
                    kernel[(rows, triton.cdiv(vocab, 8192))](
                        logits, vocab, mapping, offsets, masks, words, vocab, MASK_STRIDE=1, BLOCK_SIZE=8192
                    )
                    torch.npu.synchronize()

                for _ in range(3):
                    run()
                activities = [torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
                with torch_npu.profiler.profile(activities=activities) as profiler:
                    for _ in range(3):
                        run()
                path = args.output_dir / f"{rows}-{vocab}-{pattern}.json"
                profiler.export_chrome_trace(str(path))
                events = json.loads(path.read_text())
                if isinstance(events, dict):
                    events = events["traceEvents"]
                durations = [
                    event["dur"]
                    for event in events
                    if event.get("ph") == "X"
                    and event.get("args", {}).get("Task Type") == "KERNEL_AIVEC"
                    and "grammar_bitmask" in event.get("name", "")
                ]
                if len(durations) != 3:
                    raise RuntimeError(f"Expected three kernel events in {path}, got {len(durations)}")
                print(
                    json.dumps({"rows": rows, "vocab": vocab, "pattern": pattern, "kernel_us": durations}), flush=True
                )


if __name__ == "__main__":
    main()
