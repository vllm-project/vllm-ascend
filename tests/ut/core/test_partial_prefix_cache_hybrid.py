# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlockCopy,
    get_block_hash,
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
)

pytestmark = pytest.mark.cpu_test

HASH_BLOCK_SIZE = 2
MAMBA_BLOCK_SIZE = 2 * HASH_BLOCK_SIZE


@pytest.fixture(autouse=True)
def _init_hash_seed() -> None:
    init_none_hash(sha256)


def _make_request(request_id: str, token_ids: list[int]) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        block_hasher=get_request_block_hasher(HASH_BLOCK_SIZE, sha256),
    )


def _make_manager(
    *,
    full_block_size: int = HASH_BLOCK_SIZE,
    num_blocks: int = 24,
) -> KVCacheManager:
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=full_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=MAMBA_BLOCK_SIZE,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        max_in_flight_tokens=8192,
        scheduler_block_size=MAMBA_BLOCK_SIZE,
        hash_block_size=HASH_BLOCK_SIZE,
        enable_caching=True,
    )
    assert isinstance(manager.coordinator, AscendHybridKVCacheCoordinator)
    return manager


def test_ascend_hybrid_mamba_partial_hit_uses_cow() -> None:
    manager = _make_manager()
    owner = _make_request("owner", [0, 0, 1, 1, 2, 2])

    computed, num_computed, _ = manager.get_computed_blocks(owner)
    assert num_computed == 0
    assert manager.allocate_slots(owner, 6, num_computed, computed) is not None
    manager.free(owner)
    manager.new_step_starts()

    partial_hash = owner.block_hashes[2]
    partial_mamba_block = manager.block_pool.get_cached_block(
        partial_hash,
        kv_cache_group_ids=[1],
    )
    assert partial_mamba_block is not None
    assert partial_mamba_block[0].block_hash_num_tokens == 6

    consumer = _make_request("consumer", [0, 0, 1, 1, 2, 2, 3, 3])
    computed, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == 6
    assert [len(group) for group in computed.blocks] == [3, 2]

    new_blocks = manager.allocate_slots(consumer, 2, num_computed, computed)
    assert new_blocks is not None
    new_mamba_ids = new_blocks.get_block_ids()[1]
    assert len(new_mamba_ids) == 1
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_mamba_block[0].block_id,
            dst_block_id=new_mamba_ids[0],
        )
        in manager.take_kv_cache_block_copies()[0]
    )


def test_ascend_hybrid_partial_tail_owner_continuation_preserves_cache() -> None:
    manager = _make_manager(num_blocks=32)
    owner = _make_request("owner", [0, 0, 1, 1, 2, 2])
    computed, num_computed, _ = manager.get_computed_blocks(owner)
    assert manager.allocate_slots(owner, 6, num_computed, computed) is not None

    partial_hash = owner.block_hashes[2]
    cached = manager.block_pool.get_cached_block(partial_hash, kv_cache_group_ids=[1])
    assert cached is not None
    original_block_id = cached[0].block_id

    owner.num_computed_tokens = 6
    owner.append_output_token_ids([3])
    new_blocks = manager.allocate_slots(owner, 1)
    assert new_blocks is not None
    assert new_blocks.get_block_ids()[1] == []

    copies, _ = manager.take_kv_cache_block_copies()
    cow_copy = next(copy for copy in copies if copy.src_block_id == original_block_id)
    assert cow_copy.dst_block_id != original_block_id
    assert cached[0].block_hash is None

    moved = manager.block_pool.get_cached_block(partial_hash, kv_cache_group_ids=[1])
    assert moved is not None
    assert moved[0].block_id == cow_copy.dst_block_id
    assert get_block_hash(moved[0].block_hash) == partial_hash
    assert get_group_id(moved[0].block_hash) == 1
    assert moved[0].block_hash_num_tokens == 6


def test_ascend_hybrid_full_attention_partial_hit_uses_cow() -> None:
    manager = _make_manager(full_block_size=MAMBA_BLOCK_SIZE)
    owner = _make_request("owner", [0, 0, 1, 1, 2, 2])
    computed, num_computed, _ = manager.get_computed_blocks(owner)
    assert manager.allocate_slots(owner, 6, num_computed, computed) is not None
    manager.free(owner)
    manager.new_step_starts()

    partial_hash = owner.block_hashes[2]
    partial_full_block = manager.block_pool.get_cached_block(
        partial_hash,
        kv_cache_group_ids=[0],
    )
    assert partial_full_block is not None

    consumer = _make_request("consumer", [0, 0, 1, 1, 2, 2, 3, 3])
    computed, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == 6

    new_blocks = manager.allocate_slots(consumer, 2, num_computed, computed)
    assert new_blocks is not None
    new_full_ids = new_blocks.get_block_ids()[0]
    assert len(new_full_ids) == 1

    copies, retained = manager.take_kv_cache_block_copies()
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_full_block[0].block_id,
            dst_block_id=new_full_ids[0],
        )
        in copies
    )
    assert partial_full_block[0].ref_cnt == 1
    manager.block_pool.free_blocks(retained)
    assert partial_full_block[0].ref_cnt == 0
