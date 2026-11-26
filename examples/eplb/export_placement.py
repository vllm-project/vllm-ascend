# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Calibrate MRv2 dynamically and export an installed deployment collectively.

Run with VLLM_USE_V2_MODEL_RUNNER=1. Each non-empty line in --prompts is one
representative request. Export waits for any in-flight rearrangement to commit;
it does not guarantee that the workload has converged to an optimal placement.
"""

import argparse
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workload", required=True, help="Dataset, request lengths, concurrency and calibration settings"
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--num-redundant-experts", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--step-interval", type=int, default=3000)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--load-collection-phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--placement-scope", choices=("auto", "prefill", "decode", "mixed"), default="mixed")
    parser.add_argument("--communicator", choices=("hixl", "torch_gloo"), default=None)
    args = parser.parse_args()
    prompts = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts or args.rounds < 1:
        parser.error("Provide at least one prompt and a positive number of rounds")
    eplb_config = {
        "num_redundant_experts": args.num_redundant_experts,
        "window_size": args.window_size,
        "step_interval": args.step_interval,
    }
    if args.communicator is not None:
        eplb_config["communicator"] = args.communicator
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=eplb_config,
        additional_config={
            "eplb_config": {
                "load_collection_phase": args.load_collection_phase,
                "placement_scope": args.placement_scope,
            }
        },
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    for _ in range(args.rounds):
        llm.generate(prompts, SamplingParams(temperature=0, max_tokens=args.max_tokens))
    results = llm.collective_rpc(
        "export_eplb_placement",
        args=(str(args.output.resolve()), args.workload),
    )
    print(results)


if __name__ == "__main__":
    main()
