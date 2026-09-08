# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)

from vllm_ascend.core.jenga_prefix_coordinator import (
    JengaPrefixKVCacheCoordinator,
)
from vllm_ascend.core.typed_kv_cache import TypedKVCachePlan, TypedPageSpec


@dataclass(slots=True)
class _Request:
    request_id: str
    block_hashes: list[BlockHash]
    num_prompt_tokens: int
    shared_prefix_boundary: int = 0


def _make_coordinator(
    state_checkpoint_interval_tokens: int | None = None,
    *,
    max_model_len: int = 32,
    num_superpages: int = 6,
) -> JengaPrefixKVCacheCoordinator:
    attention = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.bfloat16,
    )
    state = MambaSpec(
        block_size=8,
        shapes=((2, 4),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    plan = TypedKVCachePlan.exact_lcm(
        (
            TypedPageSpec(0, attention.page_size_bytes, attention.block_size),
            TypedPageSpec(1, state.page_size_bytes, state.block_size),
        ),
        total_memory_bytes=num_superpages * attention.page_size_bytes,
    )
    config = KVCacheConfig(
        num_blocks=min(plan.num_blocks(0), plan.num_blocks(1)),
        kv_cache_tensors=[KVCacheTensor(plan.total_managed_bytes, ["attention", "state"])],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], attention),
            KVCacheGroupSpec(["state"], state),
        ],
    )
    checkpoint_kwargs = (
        {}
        if state_checkpoint_interval_tokens is None
        else {
            "state_checkpoint_interval_tokens": state_checkpoint_interval_tokens,
        }
    )
    return JengaPrefixKVCacheCoordinator(
        config,
        plan,
        max_model_len=max_model_len,
        max_in_flight_tokens=max_model_len,
        scheduler_block_size=8,
        hash_block_size=4,
        **checkpoint_kwargs,
    )


def _hashes(count: int) -> list[BlockHash]:
    return [BlockHash(index.to_bytes(32, "big")) for index in range(1, count + 1)]


def _admit_without_hits(
    coordinator: JengaPrefixKVCacheCoordinator,
    request_id: str,
    num_tokens: int,
) -> int:
    return coordinator.get_num_blocks_to_allocate(
        request_id,
        num_tokens=num_tokens,
        new_computed_blocks=([], []),
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_local_computed_tokens=0,
        num_tokens_main_model=num_tokens,
    )


def test_coordinator_reuses_common_prefix_and_finishes_atomic_stage() -> None:
    coordinator = _make_coordinator(state_checkpoint_interval_tokens=8)
    first = _Request("first", _hashes(8), num_prompt_tokens=9)

    assert (
        coordinator.get_num_blocks_to_allocate(
            first.request_id,
            num_tokens=8,
            new_computed_blocks=([], []),
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=8,
        )
        == 0
    )
    first_blocks = coordinator.allocate_new_blocks(first.request_id, 8, 8)
    assert [len(blocks) for blocks in first_blocks] == [2, 1]
    coordinator.cache_blocks(first, 8)
    coordinator.free(first.request_id)
    # Prefixes published during one scheduler step are intentionally not
    # consumable by Mamba until the following step.
    coordinator.new_step_starts()

    hit_blocks, hit_tokens, _ = coordinator.find_longest_cache_hit(
        first.block_hashes,
        max_cache_hit_length=8,
    )
    assert hit_tokens == 8
    assert [sum(not block.is_null for block in blocks) for blocks in hit_blocks] == [
        2,
        1,
    ]

    second = _Request("second", first.block_hashes, num_prompt_tokens=13)
    assert (
        coordinator.get_num_blocks_to_allocate(
            second.request_id,
            num_tokens=12,
            new_computed_blocks=hit_blocks,
            num_encoder_tokens=0,
            total_computed_tokens=8,
            num_local_computed_tokens=8,
            num_tokens_main_model=12,
        )
        == 0
    )
    coordinator.allocate_new_computed_blocks(
        second.request_id,
        hit_blocks,
        num_local_computed_tokens=8,
        num_external_computed_tokens=0,
    )
    second_new_blocks = coordinator.allocate_new_blocks(
        second.request_id,
        num_tokens=12,
        num_tokens_main_model=12,
    )
    assert [len(blocks) for blocks in second_new_blocks] == [1, 1]
    assert coordinator.typed_pool.policy.stats.cache_hits == 3
    assert coordinator.typed_pool.policy.stats.bytes_copied == 0

    coordinator.free(second.request_id)
    assert coordinator.typed_pool._prepared is None
    coordinator.typed_pool.policy.check_invariants()


def test_state_cache_does_not_publish_arbitrary_prompt_tail() -> None:
    coordinator = _make_coordinator(state_checkpoint_interval_tokens=512)
    request = _Request("short", _hashes(4), num_prompt_tokens=9)
    assert (
        coordinator.get_num_blocks_to_allocate(
            request.request_id,
            num_tokens=8,
            new_computed_blocks=([], []),
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=8,
        )
        == 0
    )
    coordinator.allocate_new_blocks(request.request_id, 8, 8)
    coordinator.cache_blocks(request, 8)
    coordinator.free(request.request_id)

    hit_blocks, hit_tokens, _ = coordinator.find_longest_cache_hit(
        request.block_hashes,
        max_cache_hit_length=8,
    )
    assert hit_tokens == 0
    assert all(not blocks for blocks in hit_blocks)


def test_default_mamba_checkpoint_is_materialized_by_scheduler_sized_chunks() -> None:
    coordinator = _make_coordinator(
        max_model_len=528,
        num_superpages=200,
    )
    request = _Request("long", _hashes(130), num_prompt_tokens=521)

    assert coordinator.effective_state_checkpoint_interval_tokens == 512
    # The Jenga scheduler hook clips a long prefill at 512.  That boundary is
    # therefore a real forward-pass endpoint, not metadata attached to a null
    # Mamba slot after one 520-token pass.
    assert _admit_without_hits(coordinator, request.request_id, 512) == 0
    first_chunk = coordinator.allocate_new_blocks(request.request_id, 512, 512)
    assert [len(blocks) for blocks in first_chunk] == [128, 64]
    coordinator.cache_blocks(request, 512)

    assert (
        coordinator.get_num_blocks_to_allocate(
            request.request_id,
            num_tokens=520,
            new_computed_blocks=([], []),
            num_encoder_tokens=0,
            total_computed_tokens=512,
            num_local_computed_tokens=512,
            num_tokens_main_model=520,
        )
        == 0
    )
    second_chunk = coordinator.allocate_new_blocks(request.request_id, 520, 520)
    assert [len(blocks) for blocks in second_chunk] == [2, 1]
    coordinator.cache_blocks(request, 520)
    state_blocks = coordinator.single_type_managers[1].req_to_blocks[request.request_id]
    published_state_checkpoints = [
        (block_index, block.block_hash_num_tokens)
        for block_index, block in enumerate(state_blocks)
        if not block.is_null and block.block_hash is not None
    ]
    assert published_state_checkpoints == [(63, 512)]

    coordinator.free(request.request_id)
    coordinator.new_step_starts()
    hit_blocks, hit_tokens, _ = coordinator.find_longest_cache_hit(
        request.block_hashes,
        max_cache_hit_length=520,
    )

    assert hit_tokens == 512
    assert sum(not block.is_null for block in hit_blocks[0]) == 128
    assert sum(not block.is_null for block in hit_blocks[1]) == 1


def test_same_step_mamba_sentinel_rejects_before_policy_scan_and_resets(
    monkeypatch,
) -> None:
    coordinator = _make_coordinator(state_checkpoint_interval_tokens=8)
    producer = _Request("producer", _hashes(2), num_prompt_tokens=9)
    assert _admit_without_hits(coordinator, producer.request_id, 8) == 0
    coordinator.allocate_new_blocks(producer.request_id, 8, 8)
    coordinator.cache_blocks(producer, 8)
    coordinator.free(producer.request_id)

    hit_blocks, hit_tokens, _ = coordinator.find_longest_cache_hit(
        producer.block_hashes,
        max_cache_hit_length=8,
    )
    assert hit_tokens == 8
    state_manager = coordinator.single_type_managers[1]
    assert state_manager.cached_blocks_this_step

    original_can_allocate = coordinator.typed_pool.can_allocate

    def unexpected_policy_scan(*args, **kwargs):
        pytest.fail("same-step Mamba sentinel must reject before a policy scan")

    monkeypatch.setattr(
        coordinator.typed_pool,
        "can_allocate",
        unexpected_policy_scan,
    )
    assert (
        coordinator.get_num_blocks_to_allocate(
            "consumer",
            num_tokens=8,
            new_computed_blocks=hit_blocks,
            num_encoder_tokens=0,
            total_computed_tokens=8,
            num_local_computed_tokens=8,
            num_tokens_main_model=8,
        )
        == 1
    )
    assert "consumer" not in coordinator._pending_admissions

    coordinator.new_step_starts()
    assert not state_manager.cached_blocks_this_step
    monkeypatch.setattr(
        coordinator.typed_pool,
        "can_allocate",
        original_can_allocate,
    )
    assert (
        coordinator.get_num_blocks_to_allocate(
            "consumer",
            num_tokens=8,
            new_computed_blocks=hit_blocks,
            num_encoder_tokens=0,
            total_computed_tokens=8,
            num_local_computed_tokens=8,
            num_tokens_main_model=8,
        )
        == 0
    )
    coordinator.free("consumer")


def test_boolean_admission_sentinel_requires_zero_scheduler_watermark() -> None:
    coordinator = _make_coordinator()

    admission_sentinel = _admit_without_hits(coordinator, "watermark", 8)
    assert admission_sentinel == 0
    assert coordinator.block_pool.get_num_free_blocks() == 0

    # KVCacheManager adds its watermark to the coordinator's scalar result.
    # Because the aggregate Jenga pool deliberately reports zero scalar free
    # blocks, even one watermark block rejects an otherwise valid request. The
    # construction layer must therefore fail fast on a non-zero watermark;
    # the coordinator itself is not passed that setting, so this test records
    # the incompatibility at the boundary it can observe.
    nonzero_watermark_blocks = 1
    assert admission_sentinel + nonzero_watermark_blocks > coordinator.block_pool.get_num_free_blocks()
    coordinator.free("watermark")


@pytest.mark.parametrize("terminal_operation", ["free", "pop_blocks_for_free"])
def test_terminal_operation_cancels_unconsumed_prepared_admission(
    terminal_operation: str,
) -> None:
    coordinator = _make_coordinator()
    request_id = f"prepared-{terminal_operation}"
    assert _admit_without_hits(coordinator, request_id, 8) == 0
    admission = coordinator._pending_admissions[request_id]

    with coordinator.typed_pool.request_context(
        request_id,
        timestamp=admission.timestamp,
    ):
        coordinator._prepare(request_id, admission)

    assert admission.prepared
    assert coordinator.typed_pool._prepared is not None
    result = getattr(coordinator, terminal_operation)(request_id)

    if terminal_operation == "pop_blocks_for_free":
        assert result == []
    assert request_id not in coordinator._pending_admissions
    assert coordinator.typed_pool._prepared is None
    large_pages = coordinator.typed_pool.policy.snapshot().large_pages[1:]
    assert all(page.owner_group_id is None for page in large_pages)
    assert all(request_id not in page.affinity_request_ids for page in large_pages)


def test_first_manager_allocation_exception_rolls_back_and_retries(
    monkeypatch,
) -> None:
    coordinator = _make_coordinator()
    request_id = "fail-before-consume"
    assert _admit_without_hits(coordinator, request_id, 8) == 0
    admission = coordinator._pending_admissions[request_id]
    attention_manager = coordinator.single_type_managers[0]
    original_allocate = attention_manager.allocate_new_blocks

    def injected_failure(*args, **kwargs):
        raise RuntimeError("injected first-manager failure")

    monkeypatch.setattr(
        attention_manager,
        "allocate_new_blocks",
        injected_failure,
    )
    with pytest.raises(RuntimeError, match="injected first-manager failure"):
        coordinator.allocate_new_blocks(request_id, 8, 8)

    assert not admission.prepared
    assert coordinator.typed_pool._prepared is None
    assert request_id not in coordinator._pending_admissions
    assert all(not manager.req_to_blocks.get(request_id) for manager in coordinator.single_type_managers)
    assert all(page.owner_group_id is None for page in coordinator.typed_pool.policy.snapshot().large_pages[1:])

    monkeypatch.setattr(attention_manager, "allocate_new_blocks", original_allocate)
    assert _admit_without_hits(coordinator, request_id, 8) == 0
    assert [len(blocks) for blocks in coordinator.allocate_new_blocks(request_id, 8, 8)] == [2, 1]
    coordinator.free(request_id)
    coordinator.typed_pool.policy.check_invariants()


def test_later_manager_exception_rolls_back_partial_stage_consumption(
    monkeypatch,
) -> None:
    coordinator = _make_coordinator()
    request_id = "fail-after-consume"
    assert _admit_without_hits(coordinator, request_id, 8) == 0
    admission = coordinator._pending_admissions[request_id]

    state_manager = coordinator.single_type_managers[1]
    original_allocate = state_manager.allocate_new_blocks

    def injected_failure(*args, **kwargs):
        raise RuntimeError("injected second-manager failure")

    monkeypatch.setattr(
        state_manager,
        "allocate_new_blocks",
        injected_failure,
    )
    with pytest.raises(RuntimeError, match="injected second-manager failure"):
        coordinator.allocate_new_blocks(request_id, 8, 8)

    assert coordinator.typed_pool._prepared is None
    assert not admission.prepared
    assert request_id not in coordinator._pending_admissions
    assert all(not manager.req_to_blocks.get(request_id) for manager in coordinator.single_type_managers)
    assert all(page.owner_group_id is None for page in coordinator.typed_pool.policy.snapshot().large_pages[1:])
    coordinator.typed_pool.policy.check_invariants()

    # The same request ID can be admitted again once the injected failure is
    # removed; rollback does not poison a healthy coordinator.
    monkeypatch.setattr(state_manager, "allocate_new_blocks", original_allocate)
    assert _admit_without_hits(coordinator, request_id, 8) == 0
    assert [len(blocks) for blocks in coordinator.allocate_new_blocks(request_id, 8, 8)] == [2, 1]
    coordinator.free(request_id)


@pytest.mark.parametrize("failure_site", ["manager", "policy"])
def test_second_admission_exception_discards_earlier_unprepared_decision(
    monkeypatch,
    failure_site: str,
) -> None:
    coordinator = _make_coordinator()
    request_id = f"second-admission-{failure_site}"
    assert _admit_without_hits(coordinator, request_id, 8) == 0
    old_admission = coordinator._pending_admissions[request_id]

    if failure_site == "manager":
        target = coordinator.single_type_managers[1]
        attribute = "get_num_blocks_to_allocate"
    else:
        target = coordinator.typed_pool
        attribute = "can_allocate"
    original = getattr(target, attribute)

    def injected_failure(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(f"injected {failure_site} admission failure")

    monkeypatch.setattr(target, attribute, injected_failure)
    with pytest.raises(RuntimeError, match=f"injected {failure_site}"):
        _admit_without_hits(coordinator, request_id, 8)

    assert request_id not in coordinator._pending_admissions
    assert not old_admission.prepared
    assert coordinator.typed_pool._prepared is None
    assert all(not manager.req_to_blocks.get(request_id) for manager in coordinator.single_type_managers)

    monkeypatch.setattr(target, attribute, original)
    assert _admit_without_hits(coordinator, request_id, 8) == 0
    coordinator.allocate_new_blocks(request_id, 8, 8)
    coordinator.free(request_id)


def test_computed_hit_exception_rolls_back_earlier_manager_touch(
    monkeypatch,
) -> None:
    coordinator = _make_coordinator(state_checkpoint_interval_tokens=8)
    producer = _Request("touch-producer", _hashes(2), num_prompt_tokens=9)
    assert _admit_without_hits(coordinator, producer.request_id, 8) == 0
    coordinator.allocate_new_blocks(producer.request_id, 8, 8)
    coordinator.cache_blocks(producer, 8)
    coordinator.free(producer.request_id)
    coordinator.new_step_starts()
    hit_blocks, hit_tokens, _ = coordinator.find_longest_cache_hit(
        producer.block_hashes,
        max_cache_hit_length=8,
    )
    assert hit_tokens == 8
    ref_counts_before = [block.ref_cnt for group in hit_blocks for block in group if not block.is_null]
    assert ref_counts_before == [0, 0, 0]

    request_id = "touch-consumer"
    assert (
        coordinator.get_num_blocks_to_allocate(
            request_id,
            num_tokens=8,
            new_computed_blocks=hit_blocks,
            num_encoder_tokens=0,
            total_computed_tokens=8,
            num_local_computed_tokens=8,
            num_tokens_main_model=8,
        )
        == 0
    )
    admission = coordinator._pending_admissions[request_id]
    state_manager = coordinator.single_type_managers[1]
    original_add_local = state_manager.add_local_computed_blocks

    def injected_failure(*args, **kwargs):
        raise RuntimeError("injected cache-hit failure")

    monkeypatch.setattr(
        state_manager,
        "add_local_computed_blocks",
        injected_failure,
    )
    with pytest.raises(RuntimeError, match="injected cache-hit failure"):
        coordinator.allocate_new_computed_blocks(
            request_id,
            hit_blocks,
            num_local_computed_tokens=8,
            num_external_computed_tokens=0,
        )

    assert coordinator.typed_pool._prepared is None
    assert not admission.prepared
    assert request_id not in coordinator._pending_admissions
    assert all(not manager.req_to_blocks.get(request_id) for manager in coordinator.single_type_managers)
    assert all(request_id not in manager.num_cached_block for manager in coordinator.single_type_managers)
    assert [block.ref_cnt for group in hit_blocks for block in group if not block.is_null] == ref_counts_before
    assert all(
        coordinator.typed_pool.policy.get_request_handle(
            request_id,
            group_id,
            block.block_id,
        )
        is None
        for group_id, blocks in enumerate(hit_blocks)
        for block in blocks
        if not block.is_null
    )
    coordinator.typed_pool.policy.check_invariants()

    monkeypatch.setattr(
        state_manager,
        "add_local_computed_blocks",
        original_add_local,
    )
    assert (
        coordinator.get_num_blocks_to_allocate(
            request_id,
            num_tokens=8,
            new_computed_blocks=hit_blocks,
            num_encoder_tokens=0,
            total_computed_tokens=8,
            num_local_computed_tokens=8,
            num_tokens_main_model=8,
        )
        == 0
    )
    coordinator.allocate_new_computed_blocks(
        request_id,
        hit_blocks,
        num_local_computed_tokens=8,
        num_external_computed_tokens=0,
    )
    assert coordinator.allocate_new_blocks(request_id, 8, 8) == ([], [])
    coordinator.free(request_id)
    assert [block.ref_cnt for group in hit_blocks for block in group if not block.is_null] == ref_counts_before
    coordinator.typed_pool.policy.check_invariants()
