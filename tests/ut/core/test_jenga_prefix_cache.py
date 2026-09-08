# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[3]
PACKAGE_NAME = "_jenga_prefix_cache_test"
CORE_PACKAGE_NAME = f"{PACKAGE_NAME}.core"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import the two pure-Python files through a private package namespace without
# executing vllm_ascend/__init__.py or replacing real package modules that may
# already have been imported by another test.
if PACKAGE_NAME not in sys.modules:
    package = ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT / "vllm_ascend")]
    sys.modules[PACKAGE_NAME] = package
if CORE_PACKAGE_NAME not in sys.modules:
    core_package = ModuleType(CORE_PACKAGE_NAME)
    core_package.__path__ = [str(ROOT / "vllm_ascend" / "core")]
    sys.modules[CORE_PACKAGE_NAME] = core_package

typed = _load_module(
    f"{CORE_PACKAGE_NAME}.typed_kv_cache",
    ROOT / "vllm_ascend" / "core" / "typed_kv_cache.py",
)
jenga = _load_module(
    f"{CORE_PACKAGE_NAME}.jenga_prefix_cache",
    ROOT / "vllm_ascend" / "core" / "jenga_prefix_cache.py",
)

AllocationTier = jenga.AllocationTier
JengaOutOfPagesError = jenga.JengaOutOfPagesError
JengaPrefixCache = jenga.JengaPrefixCache
PageState = jenga.PageState
TypedKVCachePlan = typed.TypedKVCachePlan
TypedPageSpec = typed.TypedPageSpec


def _plan(num_superpages: int = 4):
    return TypedKVCachePlan.exact_lcm(
        (
            TypedPageSpec(group_id=0, page_size_bytes=12, block_size_tokens=4),
            TypedPageSpec(group_id=1, page_size_bytes=8, block_size_tokens=2),
        ),
        total_memory_bytes=24 * num_superpages,
    )


def _cache_and_release(
    pool: JengaPrefixCache,
    handle,
    block_hash: str,
    *,
    prefix_length: int,
    last_access: float,
) -> None:
    pool.mark_cacheable(handle, block_hash, prefix_length, last_access)
    pool.release([handle])


def test_tiers_one_and_two_preserve_request_affinity() -> None:
    pool = JengaPrefixCache(_plan())

    first = pool.allocate("request-a", {0: 1})
    second = pool.allocate("request-a", {0: 1})

    assert first.decisions[0].tier is AllocationTier.EMPTY_LARGE_PAGE
    assert second.decisions[0].tier is AllocationTier.REQUEST_AFFINE_EMPTY
    assert first.decisions[0].handle.superpage_id == second.decisions[0].handle.superpage_id
    assert pool.stats.tier_counts[:2] == (1, 1)
    pool.check_invariants()


def test_tier_three_rebinds_fully_evictable_large_page_and_drops_stale_hashes() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=3))
    old_pages = pool.allocate("old-attention", {0: 2}).for_group(0)
    for index, handle in enumerate(old_pages):
        _cache_and_release(
            pool,
            handle,
            f"attention-prefix-{index}",
            prefix_length=(index + 1) * 16,
            last_access=float(index + 1),
        )
    # Consume the only other large page.  It deliberately retains empty small
    # pages: tier three must run before tier four.
    pool.allocate("mamba-owner", {1: 1})

    result = pool.allocate("new-mamba", {1: 1})
    decision = result.decisions[0]

    assert decision.tier is AllocationTier.EVICTABLE_LARGE_PAGE
    assert decision.previous_owner_group_id == 0
    assert decision.handle.group_id == 1
    assert len(decision.evicted_pages) == 2
    assert pool.get_cached_page(0, "attention-prefix-0") is None
    assert pool.get_cached_page(0, "attention-prefix-1") is None
    assert pool.stats.large_page_evictions == 1
    assert pool.stats.ownership_rebinds == 1
    assert pool.stats.small_page_evictions == 2
    assert pool.stats.bytes_copied == result.bytes_copied == 0
    pool.check_invariants()


def test_tier_four_uses_other_requests_empty_page_only_after_levels_zero_to_three() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=3))
    existing = pool.allocate("attention-owner", {0: 1}).decisions[0].handle
    pool.allocate("mamba-owner", {1: 1})

    result = pool.allocate("attention-neighbour", {0: 1})

    assert result.decisions[0].tier is AllocationTier.ANY_EMPTY
    assert result.decisions[0].handle.superpage_id == existing.superpage_id
    # Tier four deliberately consumes a page associated with another request;
    # it does not change the scalar placement association of the large page.
    assert pool.large_page(existing.superpage_id).affinity_request_ids == {"attention-owner"}
    pool.check_invariants()


def test_interleaved_requests_keep_scalar_large_page_affinity_until_cleared() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=3))
    first = pool.allocate("request-1", {1: 1}).for_group(1)[0]
    pool.allocate("other-group", {0: 1})
    second = pool.allocate("request-2", {1: 1}).for_group(1)[0]

    third = pool.allocate("request-1", {1: 1})

    assert second.superpage_id == first.superpage_id
    assert third.decisions[0].tier is AllocationTier.REQUEST_AFFINE_EMPTY
    assert third.decisions[0].handle.superpage_id == first.superpage_id
    assert pool.large_page(first.superpage_id).affinity_request_ids == {"request-1"}

    pool.release([first, third.decisions[0].handle])
    # Releasing leases does not implicitly end the request's placement
    # association.  The request lifecycle clears it explicitly.
    assert pool.large_page(first.superpage_id).affinity_request_ids == {"request-1"}
    pool.clear_request_affinity("request-1")
    assert not pool.large_page(first.superpage_id).affinity_request_ids


def test_tier_five_evicts_small_page_but_never_mixed_used_large_page() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=3))
    cached, protected = pool.allocate("attention", {0: 2}).for_group(0)
    _cache_and_release(
        pool,
        cached,
        "small-victim",
        prefix_length=32,
        last_access=1.0,
    )
    pool.allocate("mamba", {1: 3})

    result = pool.allocate("attention-2", {0: 1})

    assert result.decisions[0].tier is AllocationTier.EVICTABLE_SMALL_PAGE
    assert result.decisions[0].handle.block_id == cached.block_id
    assert pool.get_cached_page(0, "small-victim") is None
    assert pool.large_page(protected.superpage_id).owner_group_id == 0
    assert pool.large_page(protected.superpage_id).affinity_request_ids == {"attention"}
    assert pool.stats.large_page_evictions == 0
    assert pool.stats.small_page_evictions == 1
    pool.check_invariants()


@pytest.mark.parametrize(
    ("prefix_lengths", "expected_superpage"),
    [
        ((32, 64), 2),  # equal time: evict the longer cached prefix first
        ((64, 64), 1),  # complete tie: evict the lower group-local block ID
    ],
)
def test_small_page_lru_tie_break_is_exact(prefix_lengths, expected_superpage) -> None:
    pool = JengaPrefixCache(_plan(num_superpages=3))
    handles = pool.allocate("owner", {0: 4}).for_group(0)
    # Keep slots one and three USED, making both candidate large pages mixed and
    # therefore ineligible for a level-zero eviction.
    for handle, prefix_length in zip((handles[0], handles[2]), prefix_lengths):
        _cache_and_release(
            pool,
            handle,
            f"candidate-{handle.superpage_id}",
            prefix_length=prefix_length,
            last_access=10.0,
        )

    decision = pool.allocate("next", {0: 1}).decisions[0]

    assert decision.tier is AllocationTier.EVICTABLE_SMALL_PAGE
    assert decision.handle.superpage_id == expected_superpage


def test_level_zero_lru_uses_maximum_child_timestamp() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=4))
    handles = pool.allocate("attention", {0: 4}).for_group(0)
    timestamps = (1.0, 10.0, 5.0, 6.0)
    for handle, timestamp in zip(handles, timestamps):
        _cache_and_release(
            pool,
            handle,
            f"prefix-{handle.block_id}",
            prefix_length=handle.block_id * 16,
            last_access=timestamp,
        )
    pool.allocate("mamba-owner", {1: 1})

    decision = pool.allocate("mamba-next", {1: 1}).decisions[0]

    # Superpage one has aggregate time max(1, 10) == 10; superpage two has
    # max(5, 6) == 6 and is therefore the older whole-page victim.
    assert decision.tier is AllocationTier.EVICTABLE_LARGE_PAGE
    assert decision.handle.superpage_id == 2


def test_used_child_prevents_cross_type_large_page_eviction() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=2))
    handles = pool.allocate("attention", {0: 2}).for_group(0)
    for index, handle in enumerate(handles):
        _cache_and_release(
            pool,
            handle,
            f"shared-{index}",
            prefix_length=16,
            last_access=1.0,
        )
    hit = pool.acquire_cached("reader", 0, "shared-0", last_access=2.0)
    assert hit is not None
    before = pool.snapshot()

    with pytest.raises(JengaOutOfPagesError):
        pool.allocate("mamba", {1: 1})

    assert pool.snapshot() == before
    assert pool.large_page(1).owner_group_id == 0


def test_same_hash_is_isolated_by_group_namespace() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=3))
    attention = pool.allocate("attention", {0: 1}).for_group(0)[0]
    mamba = pool.allocate("mamba", {1: 1}).for_group(1)[0]
    _cache_and_release(pool, attention, "same-token-hash", prefix_length=64, last_access=1.0)
    _cache_and_release(pool, mamba, "same-token-hash", prefix_length=128, last_access=2.0)

    attention_hit = pool.get_cached_page(0, "same-token-hash")
    mamba_hit = pool.get_cached_page(1, "same-token-hash")

    assert attention_hit is not None and attention_hit.group_id == 0
    assert mamba_hit is not None and mamba_hit.group_id == 1
    assert attention_hit.prefix_length == 64
    assert mamba_hit.prefix_length == 128
    assert attention_hit.superpage_id != mamba_hit.superpage_id


def test_dry_run_and_failed_multi_group_allocation_are_atomic() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=2))
    pool.allocate("existing", {0: 1})
    before = pool.snapshot()

    with pytest.raises(JengaOutOfPagesError):
        pool.dry_run("too-large", {0: 1, 1: 1})
    assert pool.snapshot() == before
    assert not pool.can_allocate("too-large", {0: 1, 1: 1})
    assert pool.snapshot() == before

    with pytest.raises(JengaOutOfPagesError):
        pool.allocate("too-large", {0: 1, 1: 1})
    assert pool.snapshot() == before


def test_successful_dry_run_matches_commit_without_mutating_first() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=3))
    before = pool.snapshot()

    dry_run = pool.dry_run("request", {0: 2, 1: 1})

    assert not dry_run.committed
    assert pool.snapshot() == before
    committed = pool.allocate("request", {0: 2, 1: 1})
    assert committed.committed
    assert [decision.tier for decision in dry_run.decisions] == [decision.tier for decision in committed.decisions]
    assert [decision.handle.block_id for decision in dry_run.decisions] == [
        decision.handle.block_id for decision in committed.decisions
    ]


def test_admission_protection_keeps_later_group_prefix_hit_out_of_level_zero_eviction() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=2))
    handles = pool.allocate("writer", {0: 2}).for_group(0)
    for index, handle in enumerate(handles):
        _cache_and_release(
            pool,
            handle,
            f"prefix-{index}",
            prefix_length=(index + 1) * 16,
            last_access=1.0,
        )
    protected = pool.get_cached_page(0, "prefix-0")
    assert protected is not None
    before = pool.snapshot()

    # Without hit protection, the whole cached large page is a legal tier-three
    # victim.  Protecting one hit makes it mixed USED/EVICTABLE and therefore
    # prevents that cross-type eviction.
    assert pool.can_allocate("mamba", {1: 1})
    assert not pool.can_allocate("mamba", {1: 1}, protected=[protected])
    with pytest.raises(JengaOutOfPagesError):
        pool.allocate("mamba", {1: 1}, protected=[protected])
    assert pool.snapshot() == before


def test_duplicate_cached_leases_require_matching_number_of_releases() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=2))
    original = pool.allocate("writer", {0: 1}).for_group(0)[0]
    _cache_and_release(pool, original, "prefix", prefix_length=16, last_access=1.0)
    first = pool.acquire_cached("reader", 0, "prefix", last_access=2.0)
    second = pool.acquire_cached("reader", 0, "prefix", last_access=3.0)
    assert first == second

    pool.release([first])
    assert pool.get_cached_page(0, "prefix").state is PageState.USED
    pool.release([second])
    assert pool.get_cached_page(0, "prefix").state is PageState.EVICTABLE


def test_stale_request_handle_cannot_mutate_page_leased_by_another_request() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=2))
    writer = pool.allocate("writer", {0: 1}).for_group(0)[0]
    pool.mark_cacheable(writer, "prefix", prefix_length=16, last_access=1.0)
    reader = pool.acquire_cached("reader", 0, "prefix", last_access=2.0)
    assert reader is not None
    pool.release([writer])
    before = pool.snapshot()

    with pytest.raises(ValueError, match="does not hold"):
        pool.mark_cacheable(writer, "replacement", prefix_length=32, last_access=3.0)
    assert pool.snapshot() == before
    with pytest.raises(ValueError, match="does not hold"):
        pool.update_last_access([writer], last_access=3.0)
    assert pool.snapshot() == before

    pool.release([reader])
    assert pool.get_cached_page(0, "prefix").state is PageState.EVICTABLE


def test_failed_release_with_duplicate_handle_is_atomic() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=2))
    first, second = pool.allocate("writer", {0: 2}).for_group(0)
    before = pool.snapshot()

    # The first entry is valid, but the duplicate would exceed the one lease
    # held by this request.  Neither page may be changed when validation fails.
    with pytest.raises(ValueError, match="does not hold"):
        pool.release([first, first])

    assert pool.snapshot() == before
    assert pool.get_request_handle("writer", 0, first.block_id) == first
    assert pool.get_request_handle("writer", 0, second.block_id) == second


def test_exact_block_acquire_disambiguates_duplicate_hash_and_checks_generation() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=2))
    first, second = pool.allocate("writer", {0: 2}).for_group(0)
    _cache_and_release(pool, first, "duplicate", prefix_length=16, last_access=1.0)
    _cache_and_release(pool, second, "duplicate", prefix_length=16, last_access=1.0)

    exact_lookup = pool.get_cached_page_by_block(0, second.block_id)
    assert exact_lookup is not None
    assert exact_lookup.block_id == second.block_id
    assert exact_lookup.generation == second.generation

    exact = pool.acquire_cached_block(
        "reader",
        0,
        second.block_id,
        2.0,
        expected_generation=second.generation,
    )

    assert exact is not None and exact.block_id == second.block_id
    assert pool.get_request_handle("reader", 0, second.block_id) == exact
    assert pool.get_request_handle("reader", 0, first.block_id) is None
    with pytest.raises(ValueError, match="generation is stale"):
        pool.acquire_cached_block(
            "reader-2",
            0,
            first.block_id,
            3.0,
            expected_generation=first.generation + 100,
        )


def test_exact_block_acquire_checks_expected_hash_before_mutating_state() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=2))
    original = pool.allocate("writer", {0: 1}).for_group(0)[0]
    _cache_and_release(pool, original, "prefix", prefix_length=16, last_access=1.0)
    reference = pool.get_cached_page_by_block(0, original.block_id)
    assert reference is not None
    before = pool.snapshot()

    with pytest.raises(ValueError, match="hash is stale"):
        pool.acquire_cached_block(
            "reader",
            0,
            original.block_id,
            2.0,
            expected_generation=reference.generation,
            expected_hash="different-prefix",
        )

    assert pool.snapshot() == before
    exact = pool.acquire_cached_block(
        "reader",
        0,
        original.block_id,
        2.0,
        expected_generation=reference.generation,
        expected_hash=reference.block_hash,
    )
    assert exact is not None


def test_reset_is_atomic_with_used_pages_and_clears_cache_and_stats_when_idle() -> None:
    pool = JengaPrefixCache(_plan(num_superpages=2))
    handle = pool.allocate("writer", {0: 1}).for_group(0)[0]
    before = pool.snapshot()

    assert not pool.reset()
    assert pool.snapshot() == before

    _cache_and_release(pool, handle, "prefix", prefix_length=16, last_access=1.0)
    assert pool.reset()
    assert pool.get_cached_page(0, "prefix") is None
    assert pool.large_page(1).owner_group_id is None
    assert pool.stats.allocations == 0
    assert pool.stats.tier_counts == (0, 0, 0, 0, 0)
    assert pool.stats.bytes_copied == 0
    pool.check_invariants()
