# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Start the paired real-weight BF16 server on loopback for API validation."""

import argparse
import json
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--ep", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--worker-extension-cls", default="bf16_scope.ScopeWorker")
    options = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bf16_scope import CORE_TARGETS

    root = options.adapters.resolve()
    sys.argv = [
        "api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(options.port),
        "--model",
        options.model,
        "--served-model-name",
        "m31-base",
        "--dtype",
        "bfloat16",
        "--enable-lora",
        "--max-loras",
        "2",
        "--max-cpu-loras",
        "3",
        "--max-lora-rank",
        "8",
        "--lora-dtype",
        "bfloat16",
        "--lora-target-modules",
        *CORE_TARGETS,
        "--lora-modules",
        f"a={root}/all-a",
        f"b={root}/all-b",
        "--enforce-eager",
        "--language-model-only",
        "--skip-tokenizer-init",
        "--max-num-seqs",
        "6",
        "--max-model-len",
        "256",
        "--max-num-batched-tokens",
        "512",
        "--block-size",
        "128",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--tensor-parallel-size",
        str(options.tp),
        "--data-parallel-size",
        str(options.dp),
        "--gpu-memory-utilization",
        "0.6",
        "--seed",
        "20261124",
        "--generation-config",
        "vllm",
        "--worker-extension-cls",
        options.worker_extension_cls,
        "--additional-config",
        json.dumps({"enable_fused_mc2": 0}),
    ]
    if options.ep:
        sys.argv.append("--enable-expert-parallel")
    if options.kv_cache_memory_bytes is not None:
        sys.argv.extend(["--kv-cache-memory-bytes", str(options.kv_cache_memory_bytes)])
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
