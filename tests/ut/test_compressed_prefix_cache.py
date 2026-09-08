# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

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
    is_kv_cache_spec_uniform,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
)
from vllm_ascend.utils import vllm_version_is

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash_seed():
    register_all_kvcache_specs(None)
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


def _make_full_manager(
    physical_block_size: int = 128,
    compress_ratio: int = 4,
) -> tuple[AscendMLAAttentionSpec, BlockPool, FullAttentionManager]:
    logical_block_size = physical_block_size * compress_ratio
    # vLLM #51718 replaced MLAAttentionSpec.compress_ratio with
    # AttentionSpec.tokens_per_state on main.
    ratio_kwargs = (
        {"compress_ratio": compress_ratio} if vllm_version_is("0.27.1") else {"tokens_per_state": compress_ratio}
    )
    spec = AscendMLAAttentionSpec(
        block_size=logical_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        model_version="deepseek_v4",
        **ratio_kwargs,
    )
    block_pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=physical_block_size,
    )
    manager = FullAttentionManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=logical_block_size,
    )
    return spec, block_pool, manager


def test_ascend_mla_spec_is_not_uniform_with_mamba() -> None:
    mla_spec, _, _ = _make_full_manager()
    mamba_spec = MambaSpec(
        block_size=1,
        shapes=((1,),),
        dtypes=(torch.float32,),
    )

    assert not is_kv_cache_spec_uniform(
        {
            "mla": mla_spec,
            "mamba": mamba_spec,
        }
    )


@pytest.mark.parametrize("physical_block_size", [32, 64, 128])
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_compressed_spec_separates_logical_and_storage_blocks(
    physical_block_size: int,
    compress_ratio: int,
) -> None:
    spec, _, manager = _make_full_manager(physical_block_size, compress_ratio)
    logical_block_size = physical_block_size * compress_ratio

    assert spec.block_size == logical_block_size
    assert spec.storage_block_size == physical_block_size
    assert spec.page_size_bytes == physical_block_size * torch.tensor([], dtype=torch.float32).element_size()
    assert isinstance(manager, FullAttentionManager)

    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=logical_block_size * 3),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )
    assert spec.max_memory_usage_bytes(config) == spec.page_size_bytes * 3

    request_id = f"p{physical_block_size}-c{compress_ratio}"
    expected_blocks = {
        1: 1,
        logical_block_size - 1: 1,
        logical_block_size: 1,
        logical_block_size + 1: 2,
    }
    for num_tokens, expected in expected_blocks.items():
        manager.req_to_blocks.pop(request_id, None)
        manager.allocate_new_blocks(
            request_id,
            num_tokens=num_tokens,
            num_tokens_main_model=num_tokens,
        )
        assert len(manager.req_to_blocks[request_id]) == expected


def test_compressed_prefix_cache_uses_logical_block_hash() -> None:
    physical_block_size = 128
    compress_ratio = 4
    logical_block_size = physical_block_size * compress_ratio
    spec, block_pool, manager = _make_full_manager(
        physical_block_size,
        compress_ratio,
    )

    request_a_tokens = list(range(logical_block_size))
    request_b_tokens = request_a_tokens.copy()
    request_b_tokens[physical_block_size + 7] = 999_999

    request_a = _make_request("a", request_a_tokens, physical_block_size)
    request_b = _make_request("b", request_b_tokens, physical_block_size)

    manager.allocate_new_blocks(
        request_a.request_id,
        num_tokens=logical_block_size,
        num_tokens_main_model=logical_block_size,
    )
    manager.cache_blocks(request_a, num_tokens=logical_block_size)

    cached_hash = get_block_hash(manager.req_to_blocks[request_a.request_id][0].block_hash)
    expected_hash = BlockHashListWithBlockSize(
        request_a.block_hashes,
        physical_block_size,
        logical_block_size,
    )[0]
    assert cached_hash == expected_hash

    request_b_hashes = BlockHashListWithBlockSize(
        request_b.block_hashes,
        physical_block_size,
        logical_block_size,
    )
    hit_result = FullAttentionManager.find_longest_cache_hit(
        block_hashes=request_b_hashes,
        max_length=logical_block_size,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=spec,
        drop_eagle_block=False,
        alignment_tokens=logical_block_size,
    )
    hit_blocks = hit_result[0][0]

    assert hit_blocks == []


def test_compressed_prefix_cache_hits_identical_logical_block() -> None:
    physical_block_size = 128
    compress_ratio = 4
    logical_block_size = physical_block_size * compress_ratio
    spec, block_pool, manager = _make_full_manager(
        physical_block_size,
        compress_ratio,
    )

    request = _make_request("a", list(range(logical_block_size)), physical_block_size)
    manager.allocate_new_blocks(
        request.request_id,
        num_tokens=logical_block_size,
        num_tokens_main_model=logical_block_size,
    )
    manager.cache_blocks(request, num_tokens=logical_block_size)

    logical_hashes = BlockHashListWithBlockSize(
        request.block_hashes,
        physical_block_size,
        logical_block_size,
    )
    hit_result = FullAttentionManager.find_longest_cache_hit(
        block_hashes=logical_hashes,
        max_length=logical_block_size,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=spec,
        drop_eagle_block=False,
        alignment_tokens=logical_block_size,
    )
    hit_blocks = hit_result[0][0]

    assert hit_blocks == manager.req_to_blocks[request.request_id]


def test_hybrid_coordinator_rejects_partial_compressed_prefix_hit() -> None:
    physical_block_size = 128
    logical_block_size = physical_block_size * 4
    request_a_tokens = list(range(logical_block_size))
    request_b_tokens = request_a_tokens.copy()
    request_b_tokens[physical_block_size + 7] = 999_999

    request_a = _make_request("a", request_a_tokens, physical_block_size)
    request_b = _make_request("b", request_b_tokens, physical_block_size)
    # vLLM #51718 replaced MLAAttentionSpec.compress_ratio with
    # AttentionSpec.tokens_per_state on main.
    ratio_kwargs = {"compress_ratio": 4} if vllm_version_is("0.27.1") else {"tokens_per_state": 4}
    compressed_spec = MLAAttentionSpec(
        block_size=logical_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        model_version="deepseek_v4",
        **ratio_kwargs,
    )
    full_spec = FullAttentionSpec(
        block_size=physical_block_size,
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
        hash_block_size=physical_block_size,
        scheduler_block_size=logical_block_size,
        max_num_batched_tokens=logical_block_size,
    )

    for manager in coordinator.single_type_managers:
        manager.allocate_new_blocks(
            request_a.request_id,
            num_tokens=logical_block_size,
            num_tokens_main_model=logical_block_size,
        )
        manager.cache_blocks(request_a, num_tokens=logical_block_size)

    per_group_blocks, per_group_hits = coordinator.find_longest_cache_hit_per_group(
        request_a.block_hashes,
        max_cache_hit_length=logical_block_size,
    )
    assert isinstance(per_group_hits, tuple)
    assert per_group_hits == (logical_block_size, logical_block_size)
    assert all(per_group_blocks)

    hit_result = coordinator.find_longest_cache_hit(
        request_b.block_hashes,
        max_cache_hit_length=logical_block_size,
    )
    hit_blocks, hit_length, _ = hit_result

    assert hit_length == 0
    assert hit_blocks == ([], [])


def test_hybrid_coordinator_truncates_every_full_attention_group() -> None:
    hash_block_size = 2
    block_size = 2 * hash_block_size
    coordinator = AscendHybridKVCacheCoordinator(
        kv_cache_config=KVCacheConfig(
            num_blocks=32,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["full_a"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                ),
                KVCacheGroupSpec(
                    ["full_b"],
                    FullAttentionSpec(
                        block_size=2 * block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                ),
                KVCacheGroupSpec(
                    ["mamba"],
                    MambaSpec(
                        block_size=block_size,
                        shapes=((1,),),
                        dtypes=(torch.float32,),
                        mamba_cache_mode="align",
                    ),
                ),
            ],
        ),
        max_model_len=8192,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=hash_block_size,
        scheduler_block_size=block_size,
        max_num_batched_tokens=8192,
    )
    request = _make_request(
        "a",
        [index // hash_block_size for index in range(24)],
        hash_block_size,
    )

    for group_id in (0, 1):
        group_block_size = block_size * (1 + group_id)
        num_full_blocks = len(request.prompt_token_ids) // group_block_size
        blocks = coordinator.block_pool.get_new_blocks(num_full_blocks)
        coordinator.block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=0,
            num_full_blocks=num_full_blocks,
            block_size=group_block_size,
            kv_cache_group_id=group_id,
        )

    mamba_block = coordinator.block_pool.get_new_blocks(1)[0]
    coordinator.block_pool.cache_partial_block(
        request=request,
        block=mamba_block,
        num_tokens=6,
        kv_cache_group_id=2,
        block_size=block_size,
    )

    hit_blocks, hit_length, _ = coordinator.find_longest_cache_hit(
        request.block_hashes,
        max_cache_hit_length=len(request.prompt_token_ids),
    )

    assert hit_length == 6
    assert [len(blocks) for blocks in hit_blocks] == [2, 1, 2]
