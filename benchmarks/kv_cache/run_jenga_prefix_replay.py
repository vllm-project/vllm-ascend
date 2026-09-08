# SPDX-License-Identifier: Apache-2.0
"""Replay the Jenga prefix/eviction policy without model or device data.

This is a scheduler-policy microbenchmark, not an NPU serving benchmark.  It
checks all five allocation tiers, layer-type prefix reconciliation, eviction,
and in-place large-page ownership rebinds. ``bytes_copied=0`` means the replay
models metadata invalidation and ownership change, not payload migration. No
model output, prompt text, KV payload, or machine log is read or written unless
``--json-output`` is given.

Example::

    python -m benchmarks.kv_cache.run_jenga_prefix_replay --iterations 1000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any

ROOT = Path(__file__).parents[2]


def _load_module(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_reference_modules():
    """Load pure policy files without importing the vLLM plugin package."""

    package_name = "_jenga_prefix_replay"
    core_name = f"{package_name}.core"
    package = sys.modules.setdefault(package_name, ModuleType(package_name))
    package.__path__ = [str(ROOT / "vllm_ascend")]
    core_package = sys.modules.setdefault(core_name, ModuleType(core_name))
    core_package.__path__ = [str(ROOT / "vllm_ascend" / "core")]

    typed = _load_module(
        f"{core_name}.typed_kv_cache",
        ROOT / "vllm_ascend" / "core" / "typed_kv_cache.py",
    )
    cache = _load_module(
        f"{core_name}.jenga_prefix_cache",
        ROOT / "vllm_ascend" / "core" / "jenga_prefix_cache.py",
    )
    policy = _load_module(
        f"{core_name}.jenga_prefix_policy",
        ROOT / "vllm_ascend" / "core" / "jenga_prefix_policy.py",
    )
    return typed, cache, policy


TYPED, CACHE, POLICY = _load_reference_modules()


def _plan(num_superpages: int = 4):
    return TYPED.TypedKVCachePlan.exact_lcm(
        (
            TYPED.TypedPageSpec(0, page_size_bytes=12, block_size_tokens=4),
            TYPED.TypedPageSpec(1, page_size_bytes=8, block_size_tokens=8),
        ),
        total_memory_bytes=24 * num_superpages,
    )


def _cache_and_release(
    pool,
    handle,
    block_hash: str,
    *,
    prefix_length: int,
    last_access: float,
) -> None:
    pool.mark_cacheable(handle, block_hash, prefix_length, last_access)
    pool.release([handle])


def _exercise_five_tiers() -> dict[str, Any]:
    observed: list[int] = []
    small_evictions = 0
    large_evictions = 0
    ownership_rebinds = 0
    stale_hash_misses = 0

    # Tiers 2 then 1: claim an empty L0 page, then stay request-local.
    pool = CACHE.JengaPrefixCache(_plan())
    observed.append(int(pool.allocate("affine", {0: 1}).decisions[0].tier))
    observed.append(int(pool.allocate("affine", {0: 1}).decisions[0].tier))
    pool.check_invariants()

    # Tier 3: no empty L0 page remains; reclaim one whose children are all
    # EVICTABLE and rebind it to another layer type.
    pool = CACHE.JengaPrefixCache(_plan(num_superpages=3))
    old_pages = pool.allocate("old-attention", {0: 2}).for_group(0)
    for index, handle in enumerate(old_pages):
        _cache_and_release(
            pool,
            handle,
            f"old-prefix-{index}",
            prefix_length=(index + 1) * 4,
            last_access=float(index + 1),
        )
    pool.allocate("state-owner", {1: 1})
    decision = pool.allocate("state-next", {1: 1}).decisions[0]
    observed.append(int(decision.tier))
    stale_hash_misses += sum(pool.get_cached_page(0, f"old-prefix-{index}") is None for index in range(2))
    small_evictions += pool.stats.small_page_evictions
    large_evictions += pool.stats.large_page_evictions
    ownership_rebinds += pool.stats.ownership_rebinds
    pool.check_invariants()

    # Tier 4: every L0 is owned and none is fully evictable, so use an empty
    # child of the requested type that is associated with another request.
    pool = CACHE.JengaPrefixCache(_plan(num_superpages=3))
    pool.allocate("attention-owner", {0: 1})
    pool.allocate("state-owner", {1: 1})
    observed.append(int(pool.allocate("attention-neighbour", {0: 1}).decisions[0].tier))
    pool.check_invariants()

    # Tier 5: a mixed USED/EVICTABLE L0 cannot be reclaimed wholesale.  Evict
    # one target-type child instead.
    pool = CACHE.JengaPrefixCache(_plan(num_superpages=3))
    victim, _used = pool.allocate("attention", {0: 2}).for_group(0)
    _cache_and_release(
        pool,
        victim,
        "small-victim",
        prefix_length=32,
        last_access=1.0,
    )
    pool.allocate("state", {1: 3})
    observed.append(int(pool.allocate("attention-next", {0: 1}).decisions[0].tier))
    small_evictions += pool.stats.small_page_evictions
    large_evictions += pool.stats.large_page_evictions
    ownership_rebinds += pool.stats.ownership_rebinds
    pool.check_invariants()

    expected = {1, 2, 3, 4, 5}
    if set(observed) != expected:
        raise AssertionError(f"five-tier replay observed {observed}, expected {sorted(expected)}")
    return {
        "observed_tiers": observed,
        "tier_counts": {str(tier): observed.count(tier) for tier in sorted(expected)},
        "small_page_evictions": small_evictions,
        "large_page_evictions": large_evictions,
        "ownership_rebinds": ownership_rebinds,
        "stale_hash_misses": stale_hash_misses,
        "bytes_copied": 0,
        "invariant_errors": 0,
    }


def _exercise_layerwise_prefix() -> dict[str, Any]:
    groups = {
        "attention": POLICY.PrefixGroupState(
            POLICY.FullAttentionPrefixPolicy(block_size_tokens=128),
            set(range(12)),
        ),
        "gdn_checkpoint": POLICY.PrefixGroupState(
            POLICY.CheckpointStatePrefixPolicy(checkpoint_stride_tokens=512),
            {1, 2},
        ),
    }
    prompt_tokens = 1538
    match = POLICY.find_longest_common_prefix(groups, prompt_tokens - 1)
    return {
        "prompt_tokens": prompt_tokens,
        "common_prefix_tokens": match.prefix_tokens,
        # This is policy eligibility only. No model kernel is executed here,
        # so it must not be reported as measured prefill work avoided.
        "policy_eligible_prefix_tokens": match.prefix_tokens,
        "touched_blocks": {str(group_id): list(blocks) for group_id, blocks in match.touched_blocks.items()},
    }


def _exercise_eviction_prefix_feedback() -> dict[str, Any]:
    """Feed physical eviction results back into per-layer prefix legality."""

    pool = CACHE.JengaPrefixCache(_plan(num_superpages=3))
    logical_pages: dict[str, dict[int, Any]] = {
        "attention": {},
        "gdn_checkpoint": {},
    }

    attention = pool.allocate("attention-owner", {0: 2}).for_group(0)
    for logical_index, handle in enumerate(attention):
        block_hash = f"attention-{logical_index}"
        _cache_and_release(
            pool,
            handle,
            block_hash,
            prefix_length=(logical_index + 1) * 4,
            last_access=1.0,
        )
        logical_pages["attention"][logical_index] = pool.get_cached_page(0, block_hash)

    state = pool.allocate("state-owner", {1: 1}).for_group(1)[0]
    _cache_and_release(
        pool,
        state,
        "state-0",
        prefix_length=8,
        last_access=2.0,
    )
    logical_pages["gdn_checkpoint"][0] = pool.get_cached_page(1, "state-0")

    cached_sets = {
        "attention": set(logical_pages["attention"]),
        "gdn_checkpoint": set(logical_pages["gdn_checkpoint"]),
    }

    def match_prefix():
        return POLICY.find_longest_common_prefix(
            {
                "attention": POLICY.PrefixGroupState(
                    POLICY.FullAttentionPrefixPolicy(4),
                    cached_sets["attention"],
                ),
                "gdn_checkpoint": POLICY.PrefixGroupState(
                    POLICY.CheckpointStatePrefixPolicy(8),
                    cached_sets["gdn_checkpoint"],
                ),
            },
            8,
        )

    before = match_prefix()
    group_ids = {"attention": 0, "gdn_checkpoint": 1}
    hit_handles = []
    aligned_timestamp = 50.0
    for group_name, logical_indices in before.touched_blocks.items():
        group_id = group_ids[group_name]
        for logical_index in logical_indices:
            cached = logical_pages[group_name][logical_index]
            assert cached is not None
            handle = pool.acquire_cached_block(
                "hit-request",
                group_id,
                cached.block_id,
                aligned_timestamp,
                expected_generation=cached.generation,
                expected_hash=cached.block_hash,
            )
            assert handle is not None
            hit_handles.append(handle)

    hit_timestamps = {
        pool.get_cached_page_by_block(handle.group_id, handle.block_id).last_access for handle in hit_handles
    }
    if hit_timestamps != {aligned_timestamp}:
        raise AssertionError("layer-wise hits did not receive one aligned timestamp")
    pool.release(hit_handles, last_access=aligned_timestamp)

    replacement = pool.allocate("replacement", {1: 1})
    evicted = replacement.decisions[0].evicted_pages
    block_to_logical = {
        (group_ids[group_name], cached.block_id): (group_name, logical_index)
        for group_name, entries in logical_pages.items()
        for logical_index, cached in entries.items()
        if cached is not None
    }
    removed = []
    for page in evicted:
        logical = block_to_logical.get((page.group_id, page.block_id))
        if logical is None:
            continue
        group_name, logical_index = logical
        cached_sets[group_name].discard(logical_index)
        removed.append(f"{group_name}:{logical_index}")

    after = match_prefix()
    pool.check_invariants()
    return {
        "prefix_before_eviction": before.prefix_tokens,
        "prefix_after_eviction": after.prefix_tokens,
        "evicted_logical_pages": sorted(removed),
        "aligned_last_access": aligned_timestamp,
        "stale_hash_misses": sum(pool.get_cached_page(page.group_id, page.block_hash) is None for page in evicted),
    }


def run_replay(iterations: int = 0) -> dict[str, Any]:
    if iterations < 0:
        raise ValueError("iterations must be non-negative")

    allocation = _exercise_five_tiers()
    layerwise = _exercise_layerwise_prefix()
    eviction_feedback = _exercise_eviction_prefix_feedback()
    microbenchmark: dict[str, int | float | str] = {
        "kind": "host_scheduler_policy_only",
        "iterations": iterations,
        "verified_tier_decisions": iterations * 5,
    }
    if iterations:
        started = perf_counter()
        for _ in range(iterations):
            _exercise_five_tiers()
        elapsed = perf_counter() - started
        microbenchmark.update(
            elapsed_seconds=elapsed,
            verified_tier_decisions_per_second=(iterations * 5) / elapsed,
        )

    return {
        "scope": "policy_replay_not_model_or_npu_throughput",
        "layerwise_prefix": layerwise,
        "allocation": allocation,
        "eviction_prefix_feedback": eviction_feedback,
        "microbenchmark": microbenchmark,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    result = run_replay(args.iterations)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
