#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0.
#
"""Run a small Qwen3.5 serving benchmark matrix for GDN chunk metadata work."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch_npu  # noqa: F401

from tests.e2e.conftest import RemoteOpenAIServer
from vllm.benchmarks.serve import add_cli_args as add_bench_cli_args
from vllm.benchmarks.serve import main as run_bench_main
from vllm.utils.network_utils import get_open_port
from vllm.utils.argparse_utils import FlexibleArgumentParser

DEFAULT_BS = (1, 4, 8, 16)
DEFAULT_WORKLOADS = ("short", "prefill")
DEFAULT_MODELS_GLOB = "Qwen3.5*"
DEFAULT_SERVER_TIMEOUT_S = 2400
DEFAULT_MAX_NUM_BATCHED_TOKENS = 32768

COMPILATION_CONFIG = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [1, 4, 8, 16],
}

SPECULATIVE_CONFIG = {
    "method": "qwen3_5_mtp",
    "num_speculative_tokens": 3,
    "enforce_eager": True,
}


@dataclass(frozen=True)
class Workload:
    name: str
    input_len: int
    output_len: int
    min_prompts: int
    prompt_multiplier: int


WORKLOADS: dict[str, Workload] = {
    "short": Workload(
        name="short",
        input_len=64,
        output_len=64,
        min_prompts=16,
        prompt_multiplier=4,
    ),
    "prefill": Workload(
        name="prefill",
        input_len=2048,
        output_len=64,
        min_prompts=16,
        prompt_multiplier=4,
    ),
}


@dataclass
class BenchmarkRow:
    mode: str
    model: str
    bs: int
    workload: str
    ttft_ms: float
    tpot_ms: float
    tps: float
    acceptance_rate: float
    e2e_s: float

    def markdown_row(self) -> str:
        return (
            f"| {self.model} | {self.bs} | {self.workload} | "
            f"{self.ttft_ms:.2f} | {self.tpot_ms:.2f} | {self.tps:.2f} | "
            f"{self.acceptance_rate:.2f} | {self.e2e_s:.2f} |"
        )


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item]


def _parse_csv_strs(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_model_paths(args: argparse.Namespace) -> list[Path]:
    if args.model_path:
        return [Path(args.model_path).resolve()]

    weights_root = Path(args.weights_root)
    return sorted(
        path for path in weights_root.glob(args.model_glob) if path.is_dir()
    )


def _num_prompts_for(bs: int, workload: Workload) -> int:
    return max(workload.min_prompts, bs * workload.prompt_multiplier)


def _server_args(
    model_path: Path,
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
) -> list[str]:
    served_model_name = model_path.name
    return [
        "--host",
        host,
        "--port",
        str(port),
        "--tokenizer",
        str(model_path),
        "--trust-remote-code",
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--distributed-executor-backend",
        args.distributed_executor_backend,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--compilation-config",
        json.dumps(COMPILATION_CONFIG),
        "--speculative-config",
        json.dumps(SPECULATIVE_CONFIG),
        "--async-scheduling",
    ]


def _bench_cmd(
    *,
    model_path: Path,
    server: RemoteOpenAIServer,
    workload: Workload,
    bs: int,
    num_prompts: int,
) -> list[str]:
    return [
        "--backend",
        "openai-chat",
        "--model",
        model_path.name,
        "--served-model-name",
        model_path.name,
        "--tokenizer",
        str(model_path),
        "--trust-remote-code",
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--endpoint",
        "/v1/chat/completions",
        "--dataset-name",
        "random",
        "--num-prompts",
        str(num_prompts),
        "--input-len",
        str(workload.input_len),
        "--output-len",
        str(workload.output_len),
        "--max-concurrency",
        str(bs),
        "--temperature",
        "0",
        "--ignore-eos",
        "--disable-tqdm",
        "--percentile-metrics",
        "ttft,tpot,e2el",
        "--metric-percentiles",
        "50,90,99",
    ]


def _run_benchmark(cmd: list[str]) -> dict[str, Any]:
    print(f"RUN vllm.benchmarks.serve {' '.join(cmd)}", flush=True)
    parser = FlexibleArgumentParser(description="vLLM benchmark serve driver")
    add_bench_cli_args(parser)
    bench_args = parser.parse_args(cmd)
    return run_bench_main(bench_args)


def _result_to_row(mode: str, model_path: Path, workload: Workload, bs: int, result: dict[str, Any]) -> BenchmarkRow:
    return BenchmarkRow(
        mode=mode,
        model=str(model_path),
        bs=bs,
        workload=workload.name,
        ttft_ms=float(result["median_ttft_ms"]),
        tpot_ms=float(result["median_tpot_ms"]),
        tps=float(result["output_throughput"]),
        acceptance_rate=float(result.get("spec_decode_acceptance_rate", 0.0)),
        e2e_s=float(result["duration"]),
    )


def _print_rows(rows: list[BenchmarkRow]) -> None:
    print("JSON_RESULTS_BEGIN", flush=True)
    print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True), flush=True)
    print("JSON_RESULTS_END", flush=True)
    print("MARKDOWN_ROWS_BEGIN", flush=True)
    for row in rows:
        print(row.markdown_row(), flush=True)
    print("MARKDOWN_ROWS_END", flush=True)


def run_matrix(args: argparse.Namespace) -> list[BenchmarkRow]:
    models = _resolve_model_paths(args)
    if not models:
        raise RuntimeError(
            f"No Qwen3.5 model directories found under {args.weights_root} matching {args.model_glob!r}"
        )

    workloads = [WORKLOADS[name] for name in _parse_csv_strs(args.workloads)]
    batch_sizes = _parse_csv_ints(args.batch_sizes)
    rows: list[BenchmarkRow] = []

    for model_path in models:
        server_host = "127.0.0.1"
        server_port = get_open_port()
        server_args = _server_args(
            model_path,
            args,
            host=server_host,
            port=server_port,
        )
        print(f"Launching benchmark server for {model_path}", flush=True)
        with RemoteOpenAIServer(
            str(model_path),
            server_args,
            server_host=server_host,
            server_port=server_port,
            auto_port=False,
            max_wait_seconds=args.server_timeout_s,
        ) as server:
            for workload in workloads:
                for bs in batch_sizes:
                    num_prompts = _num_prompts_for(bs, workload)
                    bench_cmd = _bench_cmd(
                        model_path=model_path,
                        server=server,
                        workload=workload,
                        bs=bs,
                        num_prompts=num_prompts,
                    )
                    result = _run_benchmark(bench_cmd)
                    rows.append(
                        _result_to_row(
                            mode=args.mode,
                            model_path=model_path,
                            workload=workload,
                            bs=bs,
                            result=result,
                        )
                    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3.5 async scheduling + MTP performance matrix."
    )
    parser.add_argument("--mode", default="baseline", help="Record label, e.g. baseline or optimized.")
    parser.add_argument("--model-path", default=None, help="Single model path to benchmark.")
    parser.add_argument("--weights-root", default="/home/weights")
    parser.add_argument("--model-glob", default=DEFAULT_MODELS_GLOB)
    parser.add_argument("--batch-sizes", default="1,4,8,16")
    parser.add_argument("--workloads", default="short,prefill")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--distributed-executor-backend", default="mp")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=DEFAULT_MAX_NUM_BATCHED_TOKENS)
    parser.add_argument("--server-timeout-s", type=int, default=DEFAULT_SERVER_TIMEOUT_S)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    rows = run_matrix(args)
    _print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
