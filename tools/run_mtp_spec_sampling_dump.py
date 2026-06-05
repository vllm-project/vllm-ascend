#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one minimal MTP generation and dump one spec-sampling PoC case."
    )
    parser.add_argument(
        "--model",
        default="/mnt/hcs/models/DeepSeek-V4-Flash-w8a8-mtp",
        help="Local MTP model path inside the runtime environment.",
    )
    parser.add_argument(
        "--dump-dir",
        default="/tmp/spec_sampling_poc",
        help="Directory used by VLLM_ASCEND_SPEC_SAMPLING_POC_DIR.",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, my name is",
        help="Single prompt used to trigger one minimal MTP generation.",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ["VLLM_ASCEND_SPEC_SAMPLING_POC_DIR"] = args.dump_dir
    os.environ["VLLM_ASCEND_SPEC_SAMPLING_POC_MAX_CASES"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        distributed_executor_backend="mp",
        enable_expert_parallel=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
            "disable_padded_drafter_batch": False,
        },
        max_model_len=args.max_model_len,
        disable_log_stats=True,
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[20],
        ),
    )

    outputs = llm.generate(
        [args.prompt],
        SamplingParams(
            temperature=0,
            max_tokens=args.max_tokens,
            ignore_eos=False,
        ),
    )
    print(outputs[0].outputs[0].text)

    case_paths = sorted(dump_dir.glob("case_*/case.pt"))
    if not case_paths:
        raise RuntimeError(f"No PoC case dumped under {dump_dir}")

    print(f"Dumped PoC case: {case_paths[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

