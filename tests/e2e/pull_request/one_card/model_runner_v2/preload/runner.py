# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""One fresh engine per invocation; emit reproducible preload measurements."""

import argparse
import json
import time
from pathlib import Path


def _worker_memory(worker):
    import torch

    free, total = torch.npu.mem_get_info(worker.device)
    return {
        "model_memory_usage": worker.model_runner.model_memory_usage,
        "requested_memory": worker.requested_memory,
        "allocated": torch.npu.memory_allocated(worker.device),
        "reserved": torch.npu.memory_reserved(worker.device),
        "free": free,
        "total": total,
        "kv_cache_memory": worker.available_kv_cache_memory_bytes,
    }


class PreloadValidationWorker:
    """Test-only worker extension; metrics travel as ordinary RPC data."""

    def preload_memory_stats(self):
        return _worker_memory(self)

    def preload_rope_stats(self):
        from vllm_ascend.ops import rotary_embedding as rope

        layers = [module for module in self.model_runner.get_model().modules() if hasattr(module, "cos_sin_cache")]
        return {
            "types": sorted({type(module).__name__ for module in layers}),
            "global_cache_device": str(rope._cos_sin_cache.device) if rope._cos_sin_cache is not None else None,
            "global_matches_model_buffer": any(rope._cos_sin_cache is module.cos_sin_cache for module in layers),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("disk", "zero_copy", "copy"), default="disk")
    parser.add_argument("--socket-dir")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--nz", type=int, choices=(0, 2), default=0)
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--hf-overrides", type=json.loads, default={})
    args = parser.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    import vllm_ascend

    started = time.perf_counter()
    kwargs = {}
    if args.mode != "disk":
        kwargs.update(
            load_format="ipc_cache",
            model_loader_extra_config={
                "mode": args.mode,
                "socket_dir": args.socket_dir,
                "fallback": args.fallback,
                "state_timeout_s": 30,
            },
        )
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        seed=0,
        enforce_eager=not args.graph,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.55,
        enable_prefix_caching=False,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]},
        additional_config={"weight_nz_mode": args.nz},
        hf_overrides=args.hf_overrides,
        worker_extension_cls=("tests.e2e.pull_request.one_card.model_runner_v2.preload.runner.PreloadValidationWorker"),
        **kwargs,
    )
    ready_s = time.perf_counter() - started
    prompts = [
        "The capital of France is",
        "Explain why the sky appears blue in one sentence.",
        "Write a Python function that adds two integers.",
        "请用一句话介绍杭州。",
    ]
    sampling = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    # Separate warmup from the measured steady-state batch.
    llm.generate(prompts, sampling, use_tqdm=False)
    # Dynamic batching is not numerically batch-invariant on this backend:
    # even two disk loads can produce different greedy tokens near a tie.
    # Hold the batch shape fixed for the strict correctness comparison, and
    # measure the realistic mixed batch independently below.
    correctness_outputs = [llm.generate([prompt], sampling, use_tqdm=False)[0] for prompt in prompts]
    started = time.perf_counter()
    outputs = llm.generate(prompts * 5, sampling, use_tqdm=False)
    generation_s = time.perf_counter() - started
    result = {
        "mode": args.mode,
        "model": args.model,
        "dtype": args.dtype,
        "graph": args.graph,
        "nz": args.nz,
        "hf_overrides": args.hf_overrides,
        "vllm_source": vllm.__file__,
        "vllm_version": vllm.__version__,
        "ascend_source": vllm_ascend.__file__,
        "ready_s": ready_s,
        "generation_s": generation_s,
        "tokens_per_second": sum(len(out.outputs[0].token_ids) for out in outputs) / generation_s,
        "token_ids": [list(out.outputs[0].token_ids) for out in correctness_outputs],
        "performance_batch_token_ids": [list(out.outputs[0].token_ids) for out in outputs],
        "worker_memory": llm.collective_rpc("preload_memory_stats"),
        "rope_state": llm.collective_rpc("preload_rope_stats"),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("token_ids")}), flush=True)
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
