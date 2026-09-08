# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from benchmarks.kv_cache.build_scenario_from_profile import _page_count
from benchmarks.kv_cache.capture_kv_cache_profile import _capture_config
from benchmarks.kv_cache.jenga_allocator import (
    HeterogeneousPageAllocator,
    OutOfPagesError,
    PageType,
    RequestDemand,
    UniformPageAllocator,
    exact_lcm_page_size,
    replay_workload,
)


def test_exact_lcm_page_size() -> None:
    assert exact_lcm_page_size([PageType("attention", 12), PageType("mamba", 8)]) == 24


def test_request_prefers_its_partial_large_page() -> None:
    allocator = HeterogeneousPageAllocator(
        total_memory_bytes=48,
        page_types=[PageType("attention", 12), PageType("mamba", 8)],
        large_page_size_bytes=24,
    )

    allocator.allocate_request("r0", {"mamba": 2})
    pages = allocator.request_pages("r0")

    assert [(page.large_page_id, page.slot) for page in pages] == [(0, 0), (0, 1)]
    assert allocator.snapshot().active_large_pages == 1


def test_requests_pack_into_a_shared_partial_large_page() -> None:
    allocator = HeterogeneousPageAllocator(
        total_memory_bytes=48,
        page_types=[PageType("attention", 12), PageType("mamba", 8)],
        large_page_size_bytes=24,
    )

    allocator.allocate_request("r0", {"mamba": 1})
    allocator.allocate_request("r1", {"mamba": 1})

    assert allocator.snapshot().active_large_pages == 1
    assert allocator.request_pages("r0")[0].large_page_id == 0
    assert allocator.request_pages("r1")[0].large_page_id == 0


def test_empty_large_page_can_change_type_after_request_finishes() -> None:
    allocator = HeterogeneousPageAllocator(
        total_memory_bytes=24,
        page_types=[PageType("attention", 12), PageType("mamba", 8)],
        large_page_size_bytes=24,
    )
    allocator.allocate_request("r0", {"mamba": 3})
    allocator.free_request("r0")
    allocator.allocate_request("r1", {"attention": 2})

    snapshot = allocator.snapshot()
    assert snapshot.allocated_pages_by_type == {"attention": 2}
    assert snapshot.useful_bytes == snapshot.active_reserved_bytes == 24


def test_failed_request_allocation_is_atomic() -> None:
    allocator = HeterogeneousPageAllocator(
        total_memory_bytes=24,
        page_types=[PageType("attention", 12), PageType("mamba", 8)],
        large_page_size_bytes=24,
    )

    try:
        allocator.allocate_request("too-large", {"attention": 3})
    except OutOfPagesError:
        pass
    else:
        raise AssertionError("allocation should fail")

    assert allocator.snapshot().allocated_small_pages == 0
    allocator.allocate_request("fits", {"mamba": 3})
    assert allocator.snapshot().allocated_small_pages == 3


def test_bounded_superpage_reports_layout_tail_and_free_slots() -> None:
    allocator = HeterogeneousPageAllocator(
        total_memory_bytes=40,
        page_types=[PageType("attention", 12), PageType("mamba", 8)],
        large_page_size_bytes=20,
    )
    allocator.allocate_request("r0", {"attention": 1})

    snapshot = allocator.snapshot()
    assert snapshot.layout_tail_bytes == 8
    assert snapshot.free_slot_bytes == 0
    assert snapshot.active_fragmentation_bytes == 8


def test_typed_pages_admit_more_requests_than_uniform_max_page() -> None:
    page_types = [PageType("attention", 16), PageType("mamba", 4)]
    requests = [RequestDemand(f"r{i}", start_step=0, duration_steps=2, page_counts={"mamba": 2}) for i in range(4)]

    uniform_result = replay_workload(UniformPageAllocator(64, page_types), requests)
    typed_result = replay_workload(HeterogeneousPageAllocator(64, page_types, 16), requests)

    assert uniform_result.accepted_requests == 2
    assert typed_result.accepted_requests == 4
    assert typed_result.rejected_requests == 0


def test_uniform_page_can_hold_multiple_kernel_objects_but_keeps_tail_padding() -> None:
    page_types = [PageType("attention", 4), PageType("mamba", 10)]
    allocator = UniformPageAllocator(
        40,
        page_types,
        physical_page_size_bytes=20,
        objects_per_page={"attention": 5, "mamba": 1},
    )
    allocator.allocate_request("r0", {"attention": 6})

    snapshot = allocator.snapshot()
    assert snapshot.active_large_pages == 2
    assert snapshot.useful_bytes == 24
    assert snapshot.active_reserved_bytes == 40


def test_profile_demand_uses_growing_attention_and_bounded_mamba_state() -> None:
    attention = {
        "block_size_tokens": 128,
        "contains_mamba": False,
    }
    mamba = {
        "block_size_tokens": 128,
        "contains_mamba": True,
        "num_speculative_blocks": 1,
    }

    assert _page_count(attention, 1024) == 8
    assert _page_count(mamba, 64) == 2
    assert _page_count(mamba, 1024) == 3


def test_profile_converts_enlarged_attention_page_to_kernel_pages() -> None:
    class AttentionSpec:
        page_size_bytes = 80
        unpadded_page_size_bytes = 60
        block_size = 20

    config = SimpleNamespace(
        num_blocks=7,
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=AttentionSpec(),
                layer_names=["layer.0.attn"],
            )
        ],
        kv_cache_tensors=[SimpleNamespace(size=560, shared_by=["layer.0.attn"])],
    )

    profile = _capture_config(config, kernel_block_sizes=[5])
    group = profile["groups"][0]
    assert group["uniform_slots_per_page"] == 4
    assert group["experimental_block_size_tokens"] == 5
    assert group["experimental_small_page_size_bytes"] == 15
    assert group["current_padding_bytes"] == 20
