#!/usr/bin/env python3
"""Multi-path Sleep Mode E2E matrix for Ascend.

Usage (TP=2 example):
  export VLLM_USE_MODELSCOPE=True
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_ASCEND_ENABLE_NZ=0
  python scripts/e2e_sleep_matrix.py --tp 2 --rounds 3 --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")


@dataclass
class PathResult:
    name: str
    ok: bool
    sleep_ms: float
    wake_ms: float
    first_token_ms: float
    free_before_gib: float
    free_after_sleep_gib: float
    free_after_wake_gib: float
    output: str
    error: str = ""


def _mem_gib():
    import torch

    free, total = torch.npu.mem_get_info()
    return free / 1024**3, total / 1024**3


def run_one(name: str, model: str, tp: int, rounds: int, additional_config: dict) -> PathResult:
    import torch
    from vllm import LLM, SamplingParams

    free0, _ = _mem_gib()
    try:
        llm = LLM(
            model=model,
            tensor_parallel_size=tp,
            max_model_len=512,
            enforce_eager=False,
            enable_sleep_mode=True,
            additional_config=additional_config,
            gpu_memory_utilization=0.7,
        )
        params = SamplingParams(temperature=0.0, max_tokens=16)
        # warmup
        _ = llm.generate(["ping"], params)

        sleep_ms = 0.0
        wake_ms = 0.0
        for _ in range(rounds):
            s0 = time.perf_counter()
            llm.sleep(level=1)
            sleep_ms += (time.perf_counter() - s0) * 1000
            free_sleep, _ = _mem_gib()
            w0 = time.perf_counter()
            llm.wake_up()
            wake_ms += (time.perf_counter() - w0) * 1000

        free_wake, _ = _mem_gib()
        g0 = time.perf_counter()
        outs = llm.generate(["How are you?"], params)
        first_token_ms = (time.perf_counter() - g0) * 1000
        text = outs[0].outputs[0].text
        del llm
        torch.npu.empty_cache()
        return PathResult(
            name=name,
            ok=True,
            sleep_ms=sleep_ms / rounds,
            wake_ms=wake_ms / rounds,
            first_token_ms=first_token_ms,
            free_before_gib=free0,
            free_after_sleep_gib=free_sleep,
            free_after_wake_gib=free_wake,
            output=text[:120],
        )
    except Exception as exc:  # noqa: BLE001
        return PathResult(
            name=name,
            ok=False,
            sleep_ms=0.0,
            wake_ms=0.0,
            first_token_ms=0.0,
            free_before_gib=free0,
            free_after_sleep_gib=0.0,
            free_after_wake_gib=0.0,
            output="",
            error=str(exc),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", default="/data/sleep_e2e_results.json")
    args = parser.parse_args()

    base = {"enable_sleep_mode_extra_cleanup": True, "sleep_allow_fallback": True, "weight_nz_mode": 0}
    matrix = [
        (
            "combined_lightweight_inplace",
            {**base, "sleep_hccl_policy": "lightweight", "sleep_acl_graph_policy": "inplace"},
        ),
        ("hccl_destroy_acl_recapture", {**base, "sleep_hccl_policy": "destroy", "sleep_acl_graph_policy": "recapture"}),
        ("workspace_only", {**base, "sleep_hccl_policy": "skip", "sleep_acl_graph_policy": "workspace_only"}),
        ("fallback_enabled_combo", {**base, "sleep_hccl_policy": "lightweight", "sleep_acl_graph_policy": "inplace"}),
    ]

    results = []
    for name, cfg in matrix:
        print(f"=== running {name} ===", flush=True)
        r = run_one(name, args.model, args.tp, args.rounds, cfg)
        print(asdict(r), flush=True)
        results.append(asdict(r))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"tp": args.tp, "rounds": args.rounds, "model": args.model, "results": results}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
