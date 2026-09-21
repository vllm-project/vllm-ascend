# SPDX-License-Identifier: Apache-2.0
"""vLLM boundary runner: dump fixed-workload top-k logprobs for a checkpoint.

This legacy top-k logprob probe requires a working vllm + vllm_ascend NPU
runtime. For full-model prefix versus reduced-model hidden states and raw
logits, use the separate run_prefix_probe diagnostic (see PREFIX_PROBE.md).
This script is intentionally thin:
all comparison logic lives in ``tools/glm_reduced/compare.py`` and is
CPU-tested; this script only runs a fixed prompt set with greedy decoding and
serializes per-position top-k logprobs.

A valid comparison pairs two runs of the SAME checkpoint across a differing
axis (run mode, runtime revision, dtype), e.g.:

    python -m tools.glm_reduced.run_logits_dump REDUCED --profile glm-moe-dsa \
        --mode eager --enforce-eager --output baseline.jsonl
    python -m tools.glm_reduced.run_logits_dump REDUCED --profile glm-moe-dsa \
        --mode graph --output candidate.jsonl
    python -m tools.glm_reduced compare-logits baseline.jsonl candidate.jsonl \
        --atol <per-profile> --rtol <per-profile>

The checkpoint identity defaults to the reduction manifest digest of the
checkpoint directory; pass --checkpoint-id for checkpoints not built by this
tool. Dumps are validated for completeness (exact output lengths, full
position coverage, finite values) and written atomically, so a partial dump
cannot masquerade as a successful run.

Note: logit parity does NOT prove semantic task accuracy; it is a
numerical-parity gate that complements the accuracy suites in tests/e2e/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from .compare import DUMP_FORMAT
from .profiles import get_profile
from .reducer import manifest_checkpoint_id

# Engine knobs recorded in the dump metadata; --extra-engine-arg may not
# override them, because the recorded metadata would then lie.
_RESERVED_ENGINE_ARGS = {
    "model",
    "dtype",
    "max_model_len",
    "enforce_eager",
    "seed",
    "max_logprobs",
    "trust_remote_code",
    "enable_prefix_caching",
}


def _parse_extra_args(items: list[str]) -> dict:
    engine_kwargs = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--extra-engine-arg expects key=value, got {item!r}")
        if key in _RESERVED_ENGINE_ARGS:
            raise SystemExit(
                f"--extra-engine-arg may not override reserved engine setting {key!r}; "
                "it is part of the recorded run identity"
            )
        try:
            engine_kwargs[key] = json.loads(value)  # typed: ints, bools, dicts
        except json.JSONDecodeError:
            raise SystemExit(
                f"--extra-engine-arg value for {key!r} is not valid JSON: {value!r} "
                '(use JSON types: 2, true, false, 0.9, {...}; quote strings as "text")'
            ) from None
    return engine_kwargs


def _runtime_meta() -> dict:
    import vllm

    runtime = {"vllm": getattr(vllm, "__version__", "unknown")}
    try:
        import vllm_ascend

        runtime["vllm_ascend"] = getattr(vllm_ascend, "__version__", "unknown")
    except ImportError:
        runtime["vllm_ascend"] = None
    try:
        import torch

        runtime["torch"] = getattr(torch, "__version__", "unknown")
    except ImportError:
        runtime["torch"] = None
    return runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="checkpoint directory")
    parser.add_argument("--profile", required=True, help="reduction profile (defines the fixed workload)")
    parser.add_argument("--mode", required=True, help="run identity recorded in the dump (e.g. eager, graph)")
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument(
        "--checkpoint-id", default=None, help="checkpoint identity; default: reduction manifest digest of the model dir"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="unique identity of THIS run (default: generated UUID); baseline and candidate must differ",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--extra-engine-arg",
        action="append",
        default=[],
        help="extra key=value forwarded to vllm.LLM (JSON-typed values; repeatable)",
    )
    args = parser.parse_args(argv)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print(
            "run_logits_dump requires a vllm + vllm_ascend runtime (NPU). This host does not provide it; "
            "the comparison logic is still fully testable via `python -m tools.glm_reduced compare-logits`.",
            file=sys.stderr,
        )
        return 2

    profile = get_profile(args.profile)
    workload = profile.precision_workload
    checkpoint_id = args.checkpoint_id or manifest_checkpoint_id(args.model)
    run_id = args.run_id or uuid.uuid4().hex
    engine_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "trust_remote_code": True,
        "enable_prefix_caching": False,  # fixed workload must not degrade into cache reuse
        "max_logprobs": workload.logprob_top_k,  # vLLM's default can be below the requested top-k
    }
    engine_kwargs.update(_parse_extra_args(args.extra_engine_arg))
    llm = LLM(**engine_kwargs)

    prompts = []
    for prompt_index, length in enumerate(workload.prompt_lengths):
        # Deterministic synthetic token ids; fixed for baseline and candidate.
        prompts.append({"prompt_token_ids": [17 + (prompt_index * 131 + i) % 1000 for i in range(length)]})
    sampling = SamplingParams(
        temperature=0,
        max_tokens=workload.output_tokens,
        ignore_eos=True,
        logprobs=workload.logprob_top_k,
        detokenize=False,
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    if len(outputs) != len(prompts):
        print(f"engine returned {len(outputs)} outputs for {len(prompts)} prompts", file=sys.stderr)
        return 1

    meta = {
        "format": DUMP_FORMAT,
        "model": args.model,
        "checkpoint_id": checkpoint_id,
        "run_id": run_id,
        "profile": args.profile,
        "dtype": args.dtype,
        "mode": args.mode,
        "seed": args.seed,
        "prompt_count": len(prompts),
        "output_tokens": workload.output_tokens,
        "runtime": _runtime_meta(),
        "engine": {k: v for k, v in engine_kwargs.items() if k != "model"},
    }
    records = []
    problems = []
    for prompt_index, (prompt, output) in enumerate(zip(prompts, outputs)):
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)
        logprobs = completion.logprobs or []
        if len(token_ids) != workload.output_tokens or len(logprobs) != workload.output_tokens:
            problems.append(
                f"prompt {prompt_index}: expected {workload.output_tokens} tokens/logprob positions, "
                f"got {len(token_ids)}/{len(logprobs)}"
            )
            continue
        top_logprobs = []
        for position, logprob_dict in enumerate(logprobs):
            entries = [{"token_id": token_id, "logprob": lp.logprob} for token_id, lp in sorted(logprob_dict.items())]
            sampled = token_ids[position]
            values = [e["logprob"] for e in entries]
            if not entries or any(v != v or v in (float("inf"), float("-inf")) for v in values):
                problems.append(f"prompt {prompt_index} position {position}: missing/non-finite logprobs")
                break
            if sampled not in {e["token_id"] for e in entries}:
                problems.append(
                    f"prompt {prompt_index} position {position}: sampled token {sampled} missing from top-k"
                )
                break
            top_logprobs.append(entries)
        else:
            records.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_token_ids": prompt["prompt_token_ids"],
                    "token_ids": token_ids,
                    "top_logprobs": top_logprobs,
                }
            )
    if problems or len(records) != len(prompts):
        for problem in problems:
            print(problem, file=sys.stderr)
        print("dump incomplete; refusing to write a partial baseline/candidate", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    tmp_path = output_path.with_name(output_path.name + f".partial-{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")
    os.replace(tmp_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
