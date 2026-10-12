# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinatorNoPrefixCache
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request
from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec
from vllm_ascend.core.prefix_cache import (
    kv_cache_group_participates_in_prefix_caching,
    kv_cache_spec_participates_in_prefix_caching,
)
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_resolve_kv_cache_block_sizes,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash_seed():
    init_none_hash(sha256)


def _full(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )


def _ring(block_size: int) -> AscendCircularBufferSpec:
    return AscendCircularBufferSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )


def _group(name: str, spec) -> KVCacheGroupSpec:
    return KVCacheGroupSpec([name], spec)


def _config(groups: list[KVCacheGroupSpec], num_blocks: int = 64) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


def _vllm_config(
    *,
    block_size: int,
    enable_prefix_caching: bool,
    prefix_match_unit: int | None = None,
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
            prefix_match_unit=prefix_match_unit,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        kv_transfer_config=None,
    )


def _request(request_id: str, tokens: list[int], hash_block_size: int) -> Request:
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, eos_token_id=10_000)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )


def test_spec_and_group_participation_is_explicit() -> None:
    full = _full(4)
    ring = _ring(3)
    assert kv_cache_spec_participates_in_prefix_caching(full)
    assert not kv_cache_spec_participates_in_prefix_caching(ring)
    assert kv_cache_group_participates_in_prefix_caching(_group("full", full))
    assert not kv_cache_group_participates_in_prefix_caching(_group("ring", ring))


def test_uniform_group_rejects_mixed_participation() -> None:
    mixed = UniformTypeKVCacheSpecs(
        block_size=4,
        kv_cache_specs={"full": _full(4), "ring": _ring(4)},
    )
    with pytest.raises(ValueError, match="cannot mix"):
        kv_cache_spec_participates_in_prefix_caching(mixed)


def test_hash_size_ignores_nonparticipating_ring_but_scheduler_keeps_it() -> None:
    groups = [_group("full", _full(4)), _group("ring", _ring(3))]
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        _config(groups),
        _vllm_config(block_size=4, enable_prefix_caching=True),
    )
    assert scheduler_block_size == 12
    assert hash_block_size == 4


def test_qwen38_geometry_uses_768_token_hash_not_8_token_ring() -> None:
    groups = [
        _group("qsa", _full(768)),
        _group("raw_ring", _ring(8)),
        _group("state", _full(768)),
    ]
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        _config(groups),
        _vllm_config(block_size=768, enable_prefix_caching=True),
    )
    assert scheduler_block_size == 768
    assert hash_block_size == 768


def test_pure_participating_groups_keep_existing_hash_rule() -> None:
    groups = [_group("full4", _full(4)), _group("full6", _full(6))]
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        _config(groups),
        _vllm_config(block_size=4, enable_prefix_caching=True),
    )
    assert scheduler_block_size == 12
    assert hash_block_size == 2


def test_multiple_participating_sizes_still_constrain_hash() -> None:
    groups = [
        _group("full4", _full(4)),
        _group("full6", _full(6)),
        _group("ring", _ring(5)),
    ]
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        _config(groups),
        _vllm_config(block_size=4, enable_prefix_caching=True),
    )
    assert scheduler_block_size == 60
    assert hash_block_size == 2


def test_all_nonparticipating_groups_have_no_fine_hashing() -> None:
    groups = [_group("ring3", _ring(3)), _group("ring5", _ring(5))]
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        _config(groups),
        _vllm_config(block_size=4, enable_prefix_caching=True),
    )
    assert scheduler_block_size == 15
    assert hash_block_size == scheduler_block_size


def test_prefix_cache_disabled_keeps_scheduler_granularity() -> None:
    groups = [_group("full", _full(4)), _group("ring", _ring(3))]
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        _config(groups),
        _vllm_config(block_size=4, enable_prefix_caching=False),
    )
    assert scheduler_block_size == 12
    assert hash_block_size == scheduler_block_size


def _mixed_coordinator() -> AscendHybridKVCacheCoordinator:
    return AscendHybridKVCacheCoordinator(
        kv_cache_config=_config(
            [_group("full", _full(4)), _group("ring", _ring(3))],
            num_blocks=32,
        ),
        max_model_len=24,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=4,
        scheduler_block_size=12,
        max_num_batched_tokens=24,
    )


def _publish_and_release(
    coordinator: AscendHybridKVCacheCoordinator,
    request: Request,
    num_tokens: int,
) -> tuple[list[int], int]:
    allocated = coordinator.allocate_new_blocks(
        request.request_id,
        num_tokens=num_tokens,
        num_tokens_main_model=num_tokens,
    )
    full_ids = [block.block_id for block in allocated[0]]
    ring_id = allocated[1][0].block_id
    coordinator.cache_blocks(request, num_tokens)
    coordinator.free(request.request_id)
    return full_ids, ring_id


def test_mixed_hit_reuses_full_cache_and_allocates_fresh_ring() -> None:
    coordinator = _mixed_coordinator()
    request_a = _request("a", list(range(12)), 4)
    full_ids, _ = _publish_and_release(coordinator, request_a, 12)

    request_b = _request("b", list(range(12)), 4)
    hit_blocks, hit_length, _ = coordinator.find_longest_cache_hit(
        request_b.block_hashes,
        max_cache_hit_length=12,
    )
    assert hit_length == 12
    assert [block.block_id for block in hit_blocks[0]] == full_ids
    assert hit_blocks[1] == []

    coordinator.allocate_new_computed_blocks(
        request_b.request_id,
        hit_blocks,
        num_local_computed_tokens=hit_length,
        num_external_computed_tokens=0,
    )
    coordinator.allocate_new_blocks(
        request_b.request_id,
        num_tokens=13,
        num_tokens_main_model=13,
    )
    ring_manager = coordinator.single_type_managers[1]
    assert len(ring_manager.req_to_blocks[request_b.request_id]) == 1


def test_mixed_miss_is_bounded_only_by_participating_group() -> None:
    coordinator = _mixed_coordinator()
    request_a = _request("a", list(range(12)), 4)
    _publish_and_release(coordinator, request_a, 12)

    tokens_b = list(range(12))
    tokens_b[5] = 99_999
    request_b = _request("b", tokens_b, 4)
    hit_blocks, hit_length, _ = coordinator.find_longest_cache_hit(
        request_b.block_hashes,
        max_cache_hit_length=12,
    )
    # The participating full-attention group finds one 4-token block, then
    # correctly rounds it down to the 12-token scheduler reuse boundary. The
    # request-local ring does not participate in that decision.
    assert hit_length == 0
    assert hit_blocks[0] == []
    assert hit_blocks[1] == []


def test_nonparticipating_group_still_allocates_and_frees() -> None:
    coordinator = _mixed_coordinator()
    request = _request("request", list(range(12)), 4)
    coordinator.allocate_new_blocks(
        request.request_id,
        num_tokens=12,
        num_tokens_main_model=12,
    )
    ring_manager = coordinator.single_type_managers[1]
    assert len(ring_manager.req_to_blocks[request.request_id]) == 1
    coordinator.free(request.request_id)
    assert request.request_id not in ring_manager.req_to_blocks


def test_all_nonparticipating_groups_use_no_prefix_coordinator() -> None:
    coordinator = get_kv_cache_coordinator(
        kv_cache_config=_config([_group("ring_a", _ring(3)), _group("ring_b", _ring(3))]),
        max_model_len=24,
        max_in_flight_tokens=24,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=3,
        scheduler_block_size=3,
    )
    assert isinstance(coordinator, KVCacheCoordinatorNoPrefixCache)
    hit_blocks, hit_length, uncached = coordinator.find_longest_cache_hit([], 12)
    assert hit_length == 0
    assert uncached == 0
    assert hit_blocks == ([], [])
