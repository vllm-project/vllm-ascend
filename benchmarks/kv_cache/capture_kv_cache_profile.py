# SPDX-License-Identifier: Apache-2.0

"""Capture real KV cache group sizes while initializing a model on NPU.

The command runs vLLM in its single-process mode and temporarily wraps the
model runner's cache initialization method. No production source file is
modified and the wrapper exists only for the lifetime of this process.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any


def _describe_spec(spec: Any) -> dict[str, Any]:
    kind = type(spec).__name__
    result: dict[str, Any] = {
        "kind": kind,
        "page_size_bytes": int(spec.page_size_bytes),
        "block_size_tokens": int(spec.block_size),
        "contains_mamba": "Mamba" in kind,
    }
    if hasattr(spec, "num_speculative_blocks"):
        result["num_speculative_blocks"] = int(spec.num_speculative_blocks)
    if hasattr(spec, "shapes"):
        result["state_shapes"] = [list(shape) for shape in spec.shapes]
    if hasattr(spec, "dtypes"):
        result["state_dtypes"] = [str(dtype) for dtype in spec.dtypes]
    if hasattr(spec, "kv_cache_specs"):
        inner_specs = [_describe_spec(inner) for inner in spec.kv_cache_specs.values()]
        result["inner_specs"] = inner_specs
        result["contains_mamba"] = any(inner["contains_mamba"] for inner in inner_specs)
        result["unpadded_page_size_bytes"] = sum(int(inner["unpadded_page_size_bytes"]) for inner in inner_specs)
    elif hasattr(spec, "shapes") and hasattr(spec, "dtypes"):
        from vllm.utils.torch_utils import get_dtype_size

        result["unpadded_page_size_bytes"] = sum(
            math.prod(shape) * get_dtype_size(dtype) for shape, dtype in zip(spec.shapes, spec.dtypes)
        )
    elif hasattr(spec, "unpadded_page_size_bytes"):
        result["unpadded_page_size_bytes"] = int(spec.unpadded_page_size_bytes)
    elif hasattr(spec, "real_page_size_bytes"):
        result["unpadded_page_size_bytes"] = int(spec.real_page_size_bytes)
    else:
        result["unpadded_page_size_bytes"] = int(spec.page_size_bytes)
    return result


def _capture_config(kv_cache_config: Any, kernel_block_sizes: list[int]) -> dict[str, Any]:
    groups = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        group_data = _describe_spec(group.kv_cache_spec)
        kernel_block_size = int(kernel_block_sizes[group_id]) if group_id < len(kernel_block_sizes) else None
        allocation_block_size = int(group_data["block_size_tokens"])
        small_page_size = int(group_data["unpadded_page_size_bytes"])
        uniform_slots = 1
        if not group_data["contains_mamba"] and kernel_block_size:
            if allocation_block_size % kernel_block_size != 0:
                raise ValueError(
                    f"group {group_id} block size {allocation_block_size} is not divisible by kernel block size "
                    f"{kernel_block_size}"
                )
            uniform_slots = allocation_block_size // kernel_block_size
            allocation_block_size = kernel_block_size
            if small_page_size % uniform_slots != 0:
                raise ValueError(f"group {group_id} page size cannot be split into {uniform_slots} kernel pages")
            small_page_size //= uniform_slots
        group_data.update(
            {
                "name": f"group_{group_id}_{group_data['kind']}",
                "group_id": group_id,
                "num_layers": len(group.layer_names),
                "layer_names": list(group.layer_names),
                "kernel_block_size_tokens": kernel_block_size,
                "experimental_block_size_tokens": allocation_block_size,
                "experimental_small_page_size_bytes": small_page_size,
                "uniform_slots_per_page": uniform_slots,
                "current_padding_bytes": int(group_data["page_size_bytes"])
                - int(group_data["unpadded_page_size_bytes"]),
            }
        )
        groups.append(group_data)

    tensors = [
        {
            "size_bytes": int(tensor.size),
            "shared_by": list(tensor.shared_by),
        }
        for tensor in kv_cache_config.kv_cache_tensors
    ]
    return {
        "num_scheduler_blocks": int(kv_cache_config.num_blocks),
        "total_raw_tensor_bytes": sum(tensor["size_bytes"] for tensor in tensors),
        "groups": groups,
        "raw_tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--load-format", default="dummy")
    parser.add_argument(
        "--kv-cache-budget-bytes",
        type=int,
        default=8 * 1024**3,
        help=(
            "Synthetic available-memory budget used only to build the real "
            "KVCacheConfig. The profiler exits before allocating this memory."
        ),
    )
    parser.add_argument(
        "--skip-tokenizer-init",
        action="store_true",
        help="Skip tokenizer loading when only cache metadata is needed.",
    )
    args = parser.parse_args()

    # The wrapper must run in the engine process, so disable v1 multiprocessing
    # before importing vLLM or vLLM-Ascend.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM

    from vllm_ascend.utils import adapt_patch

    # CPU unit tests install patches from conftest, while this standalone NPU
    # profiler is executed directly. Apply the same platform patches before
    # importing the model runner to avoid partially initialized backend modules.
    adapt_patch()
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.worker import NPUWorker

    original_initialize = NPUModelRunner.initialize_kv_cache_tensors
    original_determine_memory = NPUWorker.determine_available_memory
    captured: dict[str, Any] = {}

    class CaptureComplete(RuntimeError):
        """Stop engine startup after the real KVCacheConfig is available."""

    def wrapped_initialize(self, kv_cache_config, kernel_block_sizes):
        captured.update(_capture_config(kv_cache_config, kernel_block_sizes))
        raise CaptureComplete

    def fixed_available_memory(self) -> int:
        return args.kv_cache_budget_bytes

    NPUModelRunner.initialize_kv_cache_tensors = wrapped_initialize
    NPUWorker.determine_available_memory = fixed_available_memory
    try:
        with contextlib.suppress(CaptureComplete):
            LLM(
                model=args.model,
                max_model_len=args.max_model_len,
                max_num_seqs=4,
                enforce_eager=True,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=1,
                trust_remote_code=True,
                load_format=args.load_format,
                skip_tokenizer_init=args.skip_tokenizer_init,
            )
        if not captured:
            raise RuntimeError("KV cache initialization was not captured")
        captured.update(
            {
                "model": args.model,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "kv_cache_budget_bytes": args.kv_cache_budget_bytes,
                "load_format": args.load_format,
                "skip_tokenizer_init": args.skip_tokenizer_init,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(captured, file, indent=2)
    finally:
        NPUModelRunner.initialize_kv_cache_tensors = original_initialize
        NPUWorker.determine_available_memory = original_determine_memory

    print(f"KV cache profile written to {args.output}")


if __name__ == "__main__":
    main()
