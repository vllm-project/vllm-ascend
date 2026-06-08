#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one minimal MTP generation and dump one spec-sampling PoC case.")
    parser.add_argument(
        "--model",
        default="/home/117_share/weight/DeepSeek-V4-Flash-w8a8-mtp",
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
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--distributed-executor-backend", default="mp")
    parser.add_argument("--enable-expert-parallel", action="store_true", default=True)
    parser.add_argument("--disable-expert-parallel", action="store_true")
    parser.add_argument("--cudagraph-mode", default="FULL_DECODE_ONLY")
    parser.add_argument(
        "--cudagraph-capture-sizes",
        default="20",
        help="Comma-separated capture sizes, e.g. 20 or 12,20",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ["VLLM_ASCEND_SPEC_SAMPLING_POC_DIR"] = args.dump_dir
    os.environ["VLLM_ASCEND_SPEC_SAMPLING_POC_MAX_CASES"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "marker_runner_start.json").write_text(
        '{"stage":"runner_start"}',
        encoding="utf-8",
    )

    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    capture_sizes = [int(item) for item in args.cudagraph_capture_sizes.split(",") if item.strip()]
    enable_expert_parallel = args.enable_expert_parallel and not args.disable_expert_parallel

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_expert_parallel=enable_expert_parallel,
        enforce_eager=args.enforce_eager,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
            "disable_padded_drafter_batch": False,
        },
        max_model_len=args.max_model_len,
        disable_log_stats=True,
        compilation_config=CompilationConfig(
            cudagraph_mode=args.cudagraph_mode,
            cudagraph_capture_sizes=capture_sizes,
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
