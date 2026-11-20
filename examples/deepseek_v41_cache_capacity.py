# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure V4.1 planner capacity without allocating weights or KV storage."""

import argparse
import json
from types import SimpleNamespace

import torch
from vllm.config import CacheConfig
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
from vllm.v1.core import kv_cache_planning
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.models.deepseek_v41.cache_config import (
    STATE_RING_ROWS,
    make_index_cache_spec,
    make_mla_cache_spec,
    make_swa_cache_spec,
)
from vllm_ascend.models.deepseek_v41.model import build_layer_plan
from vllm_ascend.patch.platform import patch_kv_cache_utils  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--layout", choices=["LBNHC", "BLHNC"], required=True)
    parser.add_argument("--budget-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--swa-block-size", type=int, default=32)
    parser.add_argument("--upstream-grouping", action="store_true", help="Measure proposed upstream packed grouping")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-batched-tokens", type=int, default=4096)
    parser.add_argument("--concurrent-batches", type=int, default=2, help="Async V2 with PP=1 reserves two batches")
    args = parser.parse_args()
    config = DeepseekV41Config.from_pretrained(args.model).get_text_config()
    cache = CacheConfig(block_size=args.block_size)
    cache.kv_cache_layout = args.layout
    runtime = SimpleNamespace(
        cache_config=cache,
        model_config=SimpleNamespace(
            hf_text_config=config,
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
            supports_kv_cache_grouping=False,
            max_model_len=args.max_model_len,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        attention_config=SimpleNamespace(hisparse_config=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        kv_transfer_config=None,
        max_in_flight_tokens=args.max_batched_tokens * args.concurrent_batches,
    )
    specs = {}
    for role in build_layer_plan(config).layers:
        prefix = f"model.layers.{role.layer_idx}.self_attn"
        specs[f"{prefix}.swa_cache"] = make_swa_cache_spec(
            block_size=min(args.swa_block_size, args.block_size),
            window_size=config.sliding_window,
            head_size=config.head_dim,
            dtype=torch.bfloat16,
            cache_dtype="auto",
        )
        if role.is_kv_source:
            specs[f"{prefix}.long_kv_cache"] = make_mla_cache_spec(
                block_size=args.block_size,
                head_size=config.head_dim,
                compress_ratio=role.compress_ratio,
            )
            specs[f"{prefix}.indexer.k_cache"] = make_index_cache_spec(
                block_size=args.block_size,
                head_size=config.index_head_dim,
                compress_ratio=role.compress_ratio,
            )
            if role.compress_ratio == 2:
                specs[f"{prefix}.compressor.state_cache"] = CircularBufferSpec(
                    block_size=STATE_RING_ROWS,
                    num_kv_heads=1,
                    head_size=2 * config.head_dim,
                    dtype=torch.float32,
                    head_size_v=0,
                )
    if args.upstream_grouping:
        groups = patch_kv_cache_utils._orig_get_packed_kv_cache_groups(runtime, specs)
        assert groups is not None
    else:
        groups = kv_cache_planning.get_kv_cache_groups(runtime, specs)
    planned = kv_cache_planning.get_kv_cache_config_from_groups(runtime, groups, args.budget_bytes)
    page_bytes = kv_cache_planning._pool_bytes_per_block(runtime, groups)
    request_bytes = kv_cache_planning._max_memory_usage_bytes_from_groups(
        kv_cache_planning.DefaultKVCacheConfigBuilder(), runtime, groups
    )
    print(
        json.dumps(
            {
                "layout": args.layout,
                "budget_bytes": args.budget_bytes,
                "pool_bytes_per_block": page_bytes,
                "num_blocks": planned.num_blocks,
                "groups": len(groups),
                "group_layers": [len(group.layer_names) for group in groups],
                "group_manager_block_sizes": [group.kv_cache_spec.block_size for group in groups],
                "max_model_len": args.max_model_len,
                "max_batched_tokens": args.max_batched_tokens,
                "concurrent_batches": args.concurrent_batches,
                "max_in_flight_tokens": runtime.max_in_flight_tokens,
                "blocks_per_max_length_request": request_bytes // page_bytes,
                "max_length_request_capacity": planned.num_blocks / (request_bytes // page_bytes),
                "group_page_bytes": [group.kv_cache_spec.page_size_bytes for group in groups],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
