# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Regression coverage for independent-pool admission and reservations.

Written during the baseline test freeze; execution must wait for handoff.
"""

import functools
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import RequestStatus

from vllm_ascend.patch.platform import patch_kv_cache_utils as planner
from vllm_ascend.patch.platform.patch_per_group_admission import (
    _can_allocate,
    _inflight_reservations,
    _patch_allocate_slots,
)
from vllm_ascend.patch.platform.patch_per_group_pools import (
    GroupBlockReservation,
    PerGroupBlockPools,
    get_full_attention_group_ids,
    is_window_capped_hybrid,
)


def _rewrite_method(cls, name: str, replacements: tuple[tuple[str, str], ...], *, install: bool = True):
    """Test-only source rewrite mirroring the retired legacy replay patch.

    Reads (and publishes) ``_ascend_source`` so it composes with the
    admission rewrite in either install order, exactly like the legacy
    module did before it was removed upstream.
    """
    original = getattr(cls, name)
    source = getattr(original, "_ascend_source", None)
    if source is None:
        source = textwrap.dedent(inspect.getsource(original))
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError(f"Cannot backport SWA replay: unexpected {cls.__name__}.{name} body")
        source = source.replace(before, after)
    namespace: dict = {}
    exec(compile(source, inspect.getfile(original), "exec"), original.__globals__, namespace)
    patched = namespace[name]
    functools.update_wrapper(patched, original)
    patched._ascend_source = source
    if install:
        setattr(cls, name, patched)
    return patched


BLOCK_SIZE = 16
FULL_SPEC = FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.float32)
SLIDING_SPEC = SlidingWindowSpec(
    block_size=BLOCK_SIZE,
    num_kv_heads=1,
    head_size=8,
    dtype=torch.float32,
    sliding_window=32,
)


def _config(counts=(8, 16), *, independent=True):
    config = KVCacheConfig(
        num_blocks=counts[1],
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["sliding"], SLIDING_SPEC, False),
            KVCacheGroupSpec(["full"], FULL_SPEC, False),
        ],
    )
    if independent:
        config.group_num_blocks = counts
    return config


def _manager(counts=(8, 16), *, enable_caching=False, independent=True):
    return KVCacheManager(
        kv_cache_config=_config(counts, independent=independent),
        max_model_len=128,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        max_in_flight_tokens=32,
        enable_caching=enable_caching,
    )


def _request(request_id="new", num_tokens=32, status=RequestStatus.WAITING):
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        num_prompt_tokens=num_tokens,
        num_computed_tokens=0,
        num_in_flight_tokens=0,
        status=status,
        block_hashes=[],
    )


def _state(manager):
    return tuple(
        (
            pool.get_num_free_blocks(),
            tuple(block.ref_cnt for block in pool.blocks),
        )
        for pool in manager.coordinator.block_pools
    )


@pytest.mark.parametrize("group_id", [0, 1], ids=["sliding-shortfall", "full-shortfall"])
def test_group_shortfall_returns_none_before_any_allocation(group_id):
    manager = _manager()
    pool = manager.coordinator.block_pools[group_id]
    held = pool.get_new_blocks(pool.get_num_free_blocks() - 1)
    before = _state(manager)
    request = _request()

    assert manager.allocate_slots(request, 32) is None
    assert _state(manager) == before
    assert all(not mgr.req_to_blocks.get(request.request_id) for mgr in manager.coordinator.single_type_managers)

    pool.free_blocks(held)
    result = manager.allocate_slots(request, 32)
    assert result is not None
    assert [len(blocks) for blocks in result.blocks] == [2, 2]


def test_exact_fit_is_not_rejected_by_summed_demand():
    manager = _manager((3, 3))

    result = manager.allocate_slots(_request(), 32)

    assert result is not None
    assert [pool.get_num_free_blocks() for pool in manager.coordinator.block_pools] == [0, 0]


def test_full_sequence_gate_checks_full_demand_not_only_first_chunk():
    manager = _manager()
    full = manager.coordinator.block_pools[1]
    full.get_new_blocks(full.get_num_free_blocks() - 2)
    before = _state(manager)

    assert manager.allocate_slots(_request(num_tokens=128), 16, full_sequence_must_fit=True) is None
    assert _state(manager) == before


@pytest.mark.parametrize("status", [RequestStatus.WAITING, RequestStatus.PREEMPTED])
def test_full_watermark_keeps_existing_waiting_admission_semantics(status):
    manager = _manager((4, 4))
    manager.watermark_blocks = 2
    request = _request(status=status)
    before = _state(manager)

    assert manager.allocate_slots(request, 32, has_scheduled_reqs=True) is None
    assert _state(manager) == before
    assert manager.allocate_slots(request, 32, has_scheduled_reqs=False) is not None


def test_sliding_inflight_reservation_is_preserved():
    manager = _manager((4, 16))
    before = _state(manager)

    assert manager.allocate_slots(_request(), 32, reserved_blocks=GroupBlockReservation((2, 1))) is None
    assert _state(manager) == before


def test_inflight_reservations_pass_the_pin_prefill_end():
    manager = _manager()
    first, second = _request("a", 32), _request("b", 48)
    # A resumed sequence: its prefill ends before the replayed token tail.
    second.num_prompt_tokens = 40
    scheduler = SimpleNamespace(
        kv_cache_manager=manager,
        max_model_len=128,
        _inflight_prefills=[first, second],
    )
    seen = []
    real = manager.coordinator.get_num_blocks_to_allocate_by_group

    def spy(*args, **kwargs):
        seen.append(kwargs.get("prefill_end"))
        return real(*args, **kwargs)

    manager.coordinator.get_num_blocks_to_allocate_by_group = spy

    assert _inflight_reservations(scheduler) == GroupBlockReservation((5, 5))
    # max(num_prompt_tokens, num_tokens - 1), the pin's own boundary.
    assert seen == [32, 47]


def test_sparse_retention_keeps_the_pin_prefill_end_in_group_demand():
    manager = _manager()
    demand_manager = manager.coordinator.single_type_managers[1]
    seen = []
    real = demand_manager.get_num_blocks_to_allocate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("prefill_end"))
        return real(*args, **kwargs)

    demand_manager.get_num_blocks_to_allocate = spy
    # 0 = keep only the latest replay boundary: the chunk reaching prefill_end
    # is the one reserving a checkpoint, so the pin's value must be forwarded.
    manager.coordinator.retention_interval = 0

    assert manager.allocate_slots(_request(), 32) is not None
    assert seen == [32]


def test_dense_retention_zeroes_the_pin_prefill_end_in_group_demand():
    manager = _manager()
    demand_manager = manager.coordinator.single_type_managers[1]
    seen = []
    real = demand_manager.get_num_blocks_to_allocate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("prefill_end"))
        return real(*args, **kwargs)

    demand_manager.get_num_blocks_to_allocate = spy
    # None = dense retention: every chunk publishes a state, so the pin's
    # coordinator zeroes the boundary and the manager must see that zero.
    manager.coordinator.retention_interval = None

    assert manager.allocate_slots(_request(), 32) is not None
    assert seen == [0]


def test_prefix_hit_is_not_touched_when_another_group_cannot_fit():
    manager = _manager(enable_caching=True)
    sliding, full = manager.coordinator.block_pools
    # Keep the hit blocks held (ref_cnt == 1): reusing each under caching
    # needs a copy-on-write block, so the full group demands one block while
    # none is free. The rejection must happen before any hit is touched.
    hits = [pool.get_new_blocks(1) for pool in (sliding, full)]
    full.get_new_blocks(full.get_num_free_blocks())
    computed = manager.create_kv_cache_blocks(tuple(hits))
    before = _state(manager)

    assert manager.allocate_slots(_request(), 16, num_new_computed_tokens=16, new_computed_blocks=computed) is None
    assert _state(manager) == before
    assert all(block.ref_cnt == 1 for blocks in hits for block in blocks)


def test_external_kv_does_not_partially_allocate_before_rejection():
    manager = _manager()
    full = manager.coordinator.block_pools[1]
    full.get_new_blocks(full.get_num_free_blocks() - 1)
    before = _state(manager)

    assert (
        manager.allocate_slots(_request(num_tokens=64), 16, num_external_computed_tokens=32, delay_cache_blocks=True)
        is None
    )
    assert _state(manager) == before


def test_coordinator_prediction_is_pure_even_if_current_pool_is_empty():
    manager = _manager()
    coordinator = manager.coordinator
    full = coordinator.block_pools[1]
    full.get_new_blocks(full.get_num_free_blocks())
    kwargs = dict(
        request_id="future",
        num_tokens=32,
        new_computed_blocks=manager.empty_kv_cache_blocks.blocks,
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_local_computed_tokens=0,
        num_tokens_main_model=32,
        apply_admission_cap=True,
    )

    assert coordinator.get_num_blocks_to_allocate_by_group(**kwargs) == (2, 2)
    assert coordinator.get_num_blocks_to_allocate(**kwargs) == 2


def test_equal_group_counts_still_have_independent_id_spaces():
    manager = _manager((4, 4))

    assert isinstance(manager.block_pool, PerGroupBlockPools)
    assert manager.coordinator.block_pools[0] is not manager.coordinator.block_pools[1]


def test_shared_pool_keeps_summed_accounting():
    manager = _manager(enable_caching=True, independent=False)
    coordinator = manager.coordinator
    kwargs = dict(
        request_id="shared",
        num_tokens=32,
        new_computed_blocks=manager.empty_kv_cache_blocks.blocks,
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_local_computed_tokens=0,
        num_tokens_main_model=32,
    )

    assert coordinator.get_num_blocks_to_allocate(**kwargs) == 4
    assert coordinator.block_pools[0] is coordinator.block_pools[1]
    assert _can_allocate(manager.block_pool, 4, reserved_blocks=0, watermark_blocks=0)


def test_full_group_selection_unwraps_uniform_specs():
    wrapped = UniformTypeKVCacheSpecs.from_specs({"full": FULL_SPEC})
    groups = _config().kv_cache_groups
    groups[1] = KVCacheGroupSpec(["full"], wrapped, False)

    assert get_full_attention_group_ids(groups) == (1,)


def test_sliding_only_model_does_not_enable_full_primary_layout(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DISABLE_WINDOW_CAPPED_POOLS", raising=False)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(is_layer_compact=True, is_block_compact=True)
        )
    )
    groups = [KVCacheGroupSpec([name], SLIDING_SPEC, False) for name in ("a", "b")]

    assert not is_window_capped_hybrid(config, groups)


def test_inflight_reservations_sum_actual_remaining_group_demands():
    manager = _manager()
    first, second = _request("a", 32), _request("b", 48)
    scheduler = SimpleNamespace(
        kv_cache_manager=manager,
        max_model_len=128,
        _inflight_prefills=[first, second],
    )

    reserved = _inflight_reservations(scheduler)

    assert reserved == GroupBlockReservation((5, 5))


def test_partial_prefix_hit_reserves_cow_before_touching_shared_blocks():
    manager = _manager(enable_caching=True)
    hits = [pool.get_new_blocks(1) for pool in manager.coordinator.block_pools]
    full = manager.coordinator.block_pools[1]
    full.get_new_blocks(full.get_num_free_blocks())
    before = _state(manager)

    assert (
        manager.allocate_slots(
            _request(num_tokens=16),
            8,
            num_new_computed_tokens=8,
            new_computed_blocks=manager.create_kv_cache_blocks(tuple(hits)),
        )
        is None
    )
    assert _state(manager) == before
    assert all(block.ref_cnt == 1 for blocks in hits for block in blocks)


def test_lookahead_slots_are_part_of_group_local_demand():
    manager = _manager((8, 3))
    before = _state(manager)

    assert manager.allocate_slots(_request(), 16, num_lookahead_tokens=32) is None
    assert _state(manager) == before


def test_step_gate_observes_capacity_released_outside_sliding_window():
    manager = _manager((4, 16))
    request = _request(num_tokens=64)
    assert manager.allocate_slots(request, 48) is not None
    assert manager.coordinator.block_pools[0].get_num_free_blocks() == 0
    request.num_computed_tokens = 48

    assert manager.allocate_slots(request, 16) is not None
    assert manager.coordinator.single_type_managers[0].req_to_blocks[request.request_id][0].is_null


@pytest.mark.parametrize("replay_first", [False, True], ids=["admission-first", "replay-first"])
def test_legacy_replay_and_group_admission_compose_in_either_order(replay_first):
    class ComposedManager:
        allocate_slots = inspect.unwrap(KVCacheManager.allocate_slots)

    # The paired vLLM pin already carries the replay-aware guard, so the
    # legacy-style rewrite is modeled as an inert extra condition: it must
    # match the pinned body exactly once and compose with the admission
    # rewrite in either install order.
    replay_replacement = (
        (
            "and num_new_computed_tokens == 0\n    ):",
            "and num_new_computed_tokens == 0\n"
            "        and num_new_computed_tokens == 0  # legacy replay composed\n"
            "    ):",
        ),
    )
    if replay_first:
        _rewrite_method(ComposedManager, "allocate_slots", replay_replacement)
    _patch_allocate_slots(ComposedManager)
    if not replay_first:
        _rewrite_method(ComposedManager, "allocate_slots", replay_replacement)

    manager = _manager(enable_caching=True)
    full = manager.coordinator.block_pools[1]
    held = full.get_new_blocks(full.get_num_free_blocks() - 1)
    before = _state(manager)
    assert ComposedManager.allocate_slots(manager, _request(), 32) is None
    assert _state(manager) == before

    full.free_blocks(held)
    hits = [pool.get_new_blocks(1) for pool in manager.coordinator.block_pools]
    result = ComposedManager.allocate_slots(
        manager,
        _request("replayed", 16),
        0,
        num_new_computed_tokens=16,
        new_computed_blocks=manager.create_kv_cache_blocks(tuple(hits)),
        delay_cache_blocks=True,
    )
    assert result is not None
    assert all(block.ref_cnt == 2 for blocks in hits for block in blocks)


def test_planner_uses_full_capacity_even_when_sliding_pool_has_more_blocks(monkeypatch):
    config = _config()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=32),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
    )
    monkeypatch.delenv("VLLM_ASCEND_WINDOW_CAPPED_IN_FLIGHT_FACTOR", raising=False)
    monkeypatch.setattr(planner, "_orig_get_kv_cache_config_from_groups", lambda *args: config)
    page_bytes = FULL_SPEC.page_size_bytes

    result = planner._get_window_capped_hybrid_kv_cache_config(
        vllm_config, config.kv_cache_groups, available_memory=20 * page_bytes
    )

    assert result.group_num_blocks == (12, 8)
    assert result.num_blocks == 8


@pytest.mark.parametrize("independent", [False, True], ids=["shared-scalar", "independent-vector"])
def test_inherited_scheduler_reservations_reach_real_allocation_gate(independent):
    class InheritedScheduler(Scheduler):
        pass

    manager = _manager(enable_caching=True, independent=independent)
    scheduler = InheritedScheduler.__new__(InheritedScheduler)
    scheduler.kv_cache_manager = manager
    scheduler.max_model_len = 128
    scheduler.num_spec_tokens = 0
    scheduler._inflight_prefills = [_request("a", 32), _request("b", 48)]

    reserved = scheduler._inflight_prefill_reserved_blocks()

    assert reserved == (GroupBlockReservation((5, 5)) if independent else 10)
    before = _state(manager)
    assert manager.allocate_slots(_request("must-wait", 48), 48, reserved_blocks=reserved) is None
    assert _state(manager) == before


def test_full_sequence_gate_checks_sliding_capacity_before_first_chunk():
    manager = _manager()
    sliding = manager.coordinator.block_pools[0]
    sliding.get_new_blocks(sliding.get_num_free_blocks() - 1)
    before = _state(manager)

    assert manager.allocate_slots(_request(num_tokens=128), 16, full_sequence_must_fit=True) is None
    assert _state(manager) == before


def test_secondary_full_shortfall_rejects_before_any_real_manager_allocates():
    config = _config()
    config.kv_cache_groups.append(KVCacheGroupSpec(["full-b"], FULL_SPEC, False))
    config.group_num_blocks = (8, 16, 4)
    manager = KVCacheManager(
        config,
        max_model_len=128,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        max_in_flight_tokens=32,
        enable_caching=False,
    )
    full_b = manager.coordinator.block_pools[2]
    full_b.get_new_blocks(full_b.get_num_free_blocks() - 1)
    before = _state(manager)

    assert manager.allocate_slots(_request(), 32) is None
    assert _state(manager) == before


def test_changed_upstream_body_fails_before_installing_patch():
    class UnsupportedManager:
        def allocate_slots(self):
            return "unchanged"

    original = UnsupportedManager.allocate_slots
    with pytest.raises(RuntimeError, match="allocate_slots"):
        _patch_allocate_slots(UnsupportedManager)
    assert UnsupportedManager.allocate_slots is original
