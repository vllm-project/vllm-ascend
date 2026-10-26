# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compare YaRN ND disk/preload execution on one otherwise idle NPU."""

import argparse
import os
from pathlib import Path

from .validate import cached_bytes, check_tokens, daemon, run_engine


def check_rope_state(data):
    for state in data["rope_state"]:
        assert state["types"] == ["AscendYaRNRotaryEmbedding"], state
        assert state["global_cache_device"].startswith("npu"), state
        assert state["global_matches_model_buffer"], state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    # Keep the checkpoint unchanged and apply the same YaRN parameters to
    # producer and consumers. The small table suffices for these short prompts.
    overrides = {
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 4096,
            "rope_theta": 1000000.0,
        }
    }
    common = {"dtype": args.dtype, "hf_overrides": overrides}
    baselines = {}
    for graph in (False, True):
        label = "graph" if graph else "eager"
        data = run_engine(args.output_dir, args.model, f"disk-{label}", graph=graph, **common)
        check_rope_state(data)
        baselines[graph] = data
    with daemon(args.output_dir, args.model, "zero", args.dtype, hf_overrides=overrides) as socket_dir:
        for graph in (False, True):
            label = "graph" if graph else "eager"
            data = run_engine(
                args.output_dir,
                args.model,
                f"zero-{label}",
                mode="zero_copy",
                graph=graph,
                socket_dir=socket_dir,
                **common,
            )
            check_rope_state(data)
            check_tokens(data, baselines[graph])
    for graph in (False, True):
        label = "graph" if graph else "eager"
        with daemon(args.output_dir, args.model, f"copy-{label}", args.dtype, hf_overrides=overrides) as socket_dir:
            data = run_engine(
                args.output_dir, args.model, f"copy-{label}", mode="copy", graph=graph, socket_dir=socket_dir, **common
            )
            check_rope_state(data)
            check_tokens(data, baselines[graph])
            assert cached_bytes(socket_dir) == 0
    print("ALL YARN CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
