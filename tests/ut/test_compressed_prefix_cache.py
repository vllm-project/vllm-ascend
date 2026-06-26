# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch
import vllm.v1.core.kv_cache_coordinator as kv_cache_coordinator
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashListWithBlockSize,
    get_block_hash,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    _block_hash_to_bytes,
    get_block_hashes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
    ExternalCachedBlockPool,
)
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator
from vllm_ascend.patch.platform.patch_prefix_cache_retention import (
    _sliding_window_reachable_block_mask,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash_seed():
    init_none_hash(sha256)


def _make_request(request_id: str, token_ids: list[int], hash_block_size: int) -> Request:
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )


def _make_compress_manager(
    block_size: int = 128,
    compress_ratio: int = 4,
) -> tuple[MLAAttentionSpec, BlockPool, CompressAttentionManager]:
    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
    )
    block_pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=block_size,
    )
    manager = CompressAttentionManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
    )
    return spec, block_pool, manager


def test_compressed_prefix_cache_uses_logical_block_hash() -> None:
    block_size = 128
    compress_ratio = 4
    logical_block_size = block_size * compress_ratio
    spec, block_pool, manager = _make_compress_manager(block_size, compress_ratio)

    request_a_tokens = list(range(logical_block_size))
    request_b_tokens = request_a_tokens.copy()
    request_b_tokens[block_size + 7] = 999_999

    request_a = _make_request("a", request_a_tokens, block_size)
    request_b = _make_request("b", request_b_tokens, block_size)

    manager.allocate_new_blocks(
        request_a.request_id,
        num_tokens=logical_block_size,
        num_tokens_main_model=logical_block_size,
    )
    manager.cache_blocks(request_a, num_tokens=logical_block_size)

    cached_hash = get_block_hash(manager.req_to_blocks[request_a.request_id][0].block_hash)
    expected_hash = BlockHashListWithBlockSize(
        request_a.block_hashes,
        block_size,
        logical_block_size,
    )[0]
    assert cached_hash == expected_hash

    hit_blocks = CompressAttentionManager.find_longest_cache_hit(
        block_hashes=request_b.block_hashes,
        max_length=logical_block_size,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=logical_block_size,
    )[0]

    assert hit_blocks == []


def test_compressed_prefix_cache_hits_identical_logical_block() -> None:
    block_size = 128
    compress_ratio = 4
    logical_block_size = block_size * compress_ratio
    spec, block_pool, manager = _make_compress_manager(block_size, compress_ratio)

    request = _make_request("a", list(range(logical_block_size)), block_size)
    manager.allocate_new_blocks(
        request.request_id,
        num_tokens=logical_block_size,
        num_tokens_main_model=logical_block_size,
    )
    manager.cache_blocks(request, num_tokens=logical_block_size)

    hit_blocks = CompressAttentionManager.find_longest_cache_hit(
        block_hashes=request.block_hashes,
        max_length=logical_block_size,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=logical_block_size,
    )[0]

    assert hit_blocks == manager.req_to_blocks[request.request_id]


def test_upstream_coordinator_factory_uses_compress_manager() -> None:
    spec, block_pool, _ = _make_compress_manager()

    manager = kv_cache_coordinator.get_manager_for_kv_cache_spec(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
    )

    assert isinstance(manager, CompressAttentionManager)


def test_unset_retention_keeps_sliding_window_dense() -> None:
    spec = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=512,
    )

    assert (
        _sliding_window_reachable_block_mask(
            type(None),
            start_block=0,
            end_block=16,
            alignment_tokens=2048,
            kv_cache_spec=spec,
            use_eagle=False,
            retention_interval=None,
            num_prompt_tokens=2048,
        )
        is None
    )


def test_sliding_window_retention_exact_alignment_keeps_current_tail() -> None:
    block_size = 128
    alignment_tokens = block_size * 128
    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=block_size * 4,
    )

    mask = _sliding_window_reachable_block_mask(
        type(None),
        start_block=0,
        end_block=128,
        alignment_tokens=alignment_tokens,
        kv_cache_spec=spec,
        use_eagle=True,
        retention_interval=0,
        num_prompt_tokens=alignment_tokens,
    )

    assert [idx for idx, keep in enumerate(mask or []) if keep] == [124, 125, 126, 127]


def test_sliding_window_eagle_hit_uses_post_pop_alignment() -> None:
    block_size = 128
    alignment_tokens = block_size * 128
    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=block_size,
    )

    class FakePool:
        null_block = object()

        def get_cached_block(self, block_hash, kv_cache_group_ids):
            if block_hash in (127, 128):
                return [block_hash]
            return None

    hit_blocks = SlidingWindowManager.find_longest_cache_hit(
        block_hashes=list(range(129)),
        max_length=alignment_tokens + block_size,
        kv_cache_group_ids=[0],
        block_pool=FakePool(),
        kv_cache_spec=spec,
        use_eagle=True,
        alignment_tokens=alignment_tokens,
    )[0]

    assert len(hit_blocks) == 128


def test_hybrid_coordinator_rejects_partial_compressed_prefix_hit() -> None:
    block_size = 128
    compress_ratio = 4
    logical_block_size = block_size * compress_ratio
    request_a_tokens = list(range(logical_block_size))
    request_b_tokens = request_a_tokens.copy()
    request_b_tokens[block_size + 7] = 999_999

    request_a = _make_request("a", request_a_tokens, block_size)
    request_b = _make_request("b", request_b_tokens, block_size)
    compressed_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
    )
    full_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    coordinator = AscendHybridKVCacheCoordinator(
        kv_cache_config=KVCacheConfig(
            num_blocks=16,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["compressed"], compressed_spec),
                KVCacheGroupSpec(["full"], full_spec),
            ],
        ),
        max_model_len=logical_block_size,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=block_size,
        max_num_batched_tokens=logical_block_size,
    )

    for manager in coordinator.single_type_managers:
        manager.allocate_new_blocks(
            request_a.request_id,
            num_tokens=logical_block_size,
            num_tokens_main_model=logical_block_size,
        )
        manager.cache_blocks(request_a, num_tokens=logical_block_size)

    hit_blocks, hit_length = coordinator.find_longest_cache_hit(
        request_b.block_hashes,
        max_cache_hit_length=logical_block_size,
    )

    assert hit_length == 0
    assert hit_blocks == ([], [])


def test_external_coordinator_does_not_double_apply_compress_ratio() -> None:
    block_size = 128
    compress_ratio = 4
    logical_block_size = block_size * compress_ratio
    request = _make_request("a", list(range(logical_block_size)), block_size)
    compressed_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
    )
    coordinator = AscendStoreCoordinator(
        kv_cache_groups=[KVCacheGroupSpec(["compressed"], compressed_spec)],
        scheduler_block_size=logical_block_size,
        hash_block_size=block_size,
        group_block_sizes=[block_size],
        group_cache_families=[f"c{compress_ratio}"],
    )
    chunk_hash = get_block_hashes(
        request.block_hashes,
        logical_block_size,
        block_size,
    )[0]

    _, hit_length = coordinator.find_longest_cache_hit(
        request.block_hashes,
        logical_block_size,
        ExternalCachedBlockPool({(0, _block_hash_to_bytes(chunk_hash))}),
    )

    assert hit_length == logical_block_size


def test_mtp_fallback_excludes_compressed_groups_from_eagle_drop() -> None:
    block_size = 128
    compressed_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        compress_ratio=4,
        model_version="deepseek_v4",
    )
    sliding_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=block_size,
    )
    groups = [
        KVCacheGroupSpec(["compressed"], compressed_spec),
        KVCacheGroupSpec(["sliding"], sliding_spec),
    ]
    local_coordinator = AscendHybridKVCacheCoordinator(
        kv_cache_config=KVCacheConfig(
            num_blocks=16,
            kv_cache_tensors=[],
            kv_cache_groups=groups,
        ),
        max_model_len=block_size * 4,
        use_eagle=True,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=block_size,
        max_num_batched_tokens=block_size * 4,
    )
    external_coordinator = AscendStoreCoordinator(
        kv_cache_groups=groups,
        scheduler_block_size=block_size * 4,
        hash_block_size=block_size,
        group_block_sizes=[block_size, block_size],
        group_cache_families=["c4", "c1"],
        use_eagle=True,
    )

    assert local_coordinator.eagle_group_ids == {1}
    assert external_coordinator.eagle_group_ids == {1}
