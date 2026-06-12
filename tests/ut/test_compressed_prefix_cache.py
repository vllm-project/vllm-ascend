# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashListWithBlockSize,
    get_block_hash,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import Request

from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator
from vllm_ascend.utils import vllm_version_is

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
    if vllm_version_is("0.21.0"):
        manager = CompressAttentionManager(
            spec,
            block_pool=block_pool,
            enable_caching=True,
            kv_cache_group_id=0,
        )
    else:
        manager = CompressAttentionManager(
            spec,
            block_pool=block_pool,
            enable_caching=True,
            kv_cache_group_id=0,
            scheduler_block_size=block_size,
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
