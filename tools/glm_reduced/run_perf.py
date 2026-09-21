# SPDX-License-Identifier: Apache-2.0
"""vLLM boundary runner: collect repeatable performance samples.

Requires a working vllm + vllm_ascend NPU runtime; intentionally thin. The
workload shape (prompt lengths, output tokens, warmup/measured iterations)
comes from the reduction profile, so runs are comparable by construction.
Aggregation and comparison live in ``tools/glm_reduced/perf.py`` (CPU-tested).

The regression gate compares two runs of the SAME checkpoint — typically one
reduced checkpoint across two runtime revisions/modes:

    python -m tools.glm_reduced.run_perf REDUCED --profile glm-moe-dsa \
        --hardware <tag> --mode baseline-rev-A --output perf-baseline.json
    python -m tools.glm_reduced.run_perf REDUCED --profile glm-moe-dsa \
        --hardware <tag> --mode candidate-rev-B --output perf-candidate.json
    python -m tools.glm_reduced compare-perf perf-baseline.json perf-candidate.json \
        --max-latency-regression-pct <per-profile/hardware>

Prefix caching is disabled by default so repeated iterations measure real
prefill+decode work rather than cache reuse; enabling it is recorded in the
engine metadata and changes the workload fingerprint implicitly via settings.
Every measured iteration must produce exactly the declared token count, so
early EOS or cache-shortened work fails validation instead of passing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

from .perf import PERF_FORMAT
from .profiles import get_profile
from .reducer import manifest_checkpoint_id
from .run_logits_dump import _parse_extra_args, _runtime_meta


def _workload_id(profile_name: str, prompt_lengths: tuple[int, ...], output_tokens: int) -> str:
    payload = json.dumps(
        {"profile": profile_name, "prompt_lengths": list(prompt_lengths), "output_tokens": output_tokens},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    """Measure complete offline generations and publish only full-length runs.

    This separate offline helper records whole-generation wall time; the nightly
    serving gate uses vllm bench serve for streaming TTFT and TPOT measurements.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="checkpoint directory")
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--hardware", required=True, help="free-form hardware tag; runs are only comparable within the same tag"
    )
    parser.add_argument("--mode", required=True, help="run identity (e.g. rev-A-eager, rev-B-graph)")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint-id", default=None, help="checkpoint identity; default: reduction manifest digest of the model dir"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="unique identity of THIS run (default: generated UUID); baseline and candidate must differ",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="off by default so iterations measure equal work; recorded when enabled",
    )
    parser.add_argument("--extra-engine-arg", action="append", default=[])
    args = parser.parse_args(argv)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print(
            "run_perf requires a vllm + vllm_ascend runtime (NPU). This host does not provide it; "
            "aggregation/comparison are testable via `python -m tools.glm_reduced compare-perf`.",
            file=sys.stderr,
        )
        return 2

    profile = get_profile(args.profile)
    workload = profile.performance_workload
    checkpoint_id = args.checkpoint_id or manifest_checkpoint_id(args.model)
    run_id = args.run_id or uuid.uuid4().hex
    engine_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "trust_remote_code": True,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    engine_kwargs.update(_parse_extra_args(args.extra_engine_arg))
    llm = LLM(**engine_kwargs)

    prompts = [
        {"prompt_token_ids": [31 + (j * 733 + i) % 2000 for i in range(length)]}
        for j, length in enumerate(workload.prompt_lengths)
    ]
    sampling = SamplingParams(temperature=0, max_tokens=workload.output_tokens, ignore_eos=True, detokenize=False)
    expected_tokens = len(prompts) * workload.output_tokens

    for _ in range(workload.warmup_iterations):
        llm.generate(prompts, sampling, use_tqdm=False)

    latencies: list[float] = []
    total_tokens: list[int] = []
    problems: list[str] = []
    for iteration in range(workload.measured_iterations):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        latencies.append(time.perf_counter() - start)
        produced = sum(len(o.outputs[0].token_ids) for o in outputs)
        if produced != expected_tokens:
            problems.append(f"iteration {iteration}: produced {produced} tokens, expected {expected_tokens}")
        total_tokens.append(produced)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print("perf record incomplete; refusing to write it", file=sys.stderr)
        return 1

    payload = {
        "format": PERF_FORMAT,
        "meta": {
            "model": args.model,
            "checkpoint_id": checkpoint_id,
            "hardware": args.hardware,
            "mode": args.mode,
            "run_id": run_id,
            "profile": args.profile,
            "workload_id": _workload_id(args.profile, workload.prompt_lengths, workload.output_tokens),
            "warmup_iterations": workload.warmup_iterations,
            "measured_iterations": workload.measured_iterations,
            "expected_iteration_tokens": expected_tokens,
            "seed": args.seed,
            "runtime": _runtime_meta(),
            "engine": {k: v for k, v in engine_kwargs.items() if k != "model"},
        },
        "latencies_s": latencies,
        "total_tokens": total_tokens,
    }
    output_path = Path(args.output)
    tmp_path = output_path.with_name(output_path.name + f".partial-{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
