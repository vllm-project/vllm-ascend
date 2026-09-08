# SPDX-License-Identifier: Apache-2.0

from benchmarks.kv_cache.run_jenga_prefix_replay import run_replay


def test_prefix_policy_replay_covers_dependencies_and_allocation_tiers() -> None:
    result = run_replay(iterations=2)

    assert result["scope"] == "policy_replay_not_model_or_npu_throughput"
    assert result["layerwise_prefix"] == {
        "prompt_tokens": 1538,
        "common_prefix_tokens": 1536,
        "policy_eligible_prefix_tokens": 1536,
        "touched_blocks": {
            "attention": list(range(12)),
            "gdn_checkpoint": [2],
        },
    }
    assert result["eviction_prefix_feedback"] == {
        "prefix_before_eviction": 8,
        "prefix_after_eviction": 0,
        "evicted_logical_pages": ["attention:0", "attention:1"],
        "aligned_last_access": 50.0,
        "stale_hash_misses": 2,
    }
    allocation = result["allocation"]
    assert allocation["tier_counts"] == {str(tier): 1 for tier in range(1, 6)}
    assert allocation["small_page_evictions"] == 3
    assert allocation["large_page_evictions"] == 1
    assert allocation["ownership_rebinds"] == 1
    assert allocation["stale_hash_misses"] == 2
    assert allocation["bytes_copied"] == 0
    assert allocation["invariant_errors"] == 0

    benchmark = result["microbenchmark"]
    assert benchmark["kind"] == "host_scheduler_policy_only"
    assert benchmark["iterations"] == 2
    assert benchmark["verified_tier_decisions"] == 10
    assert benchmark["elapsed_seconds"] > 0
    assert benchmark["verified_tier_decisions_per_second"] > 0
