# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import vllm.v1.core.kv_cache_utils as vllm_kv_cache_utils
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashListWithBlockSize,
    generate_scheduler_kv_cache_config,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

import vllm_ascend.patch.platform.patch_kv_cache_utils as kv_cache_utils_patch
from vllm_ascend.core.kv_cache_interface import (
    AscendDCPReplicatedDraftAttentionSpec,
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
    _is_deepseek_v4_kv_cache_spec,
    get_kv_cache_coordinator,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_resolve_kv_cache_block_sizes,
    _get_kimi_k3_dspark_mixed_kv_cache_groups,
    _get_kimi_k3_replicated_dspark_kv_cache_config,
    _get_kv_cache_config_deepseek_v4,
    _get_kv_cache_config_deepseek_v4_main,
    _unify_kv_cache_spec_page_size,
    group_and_unify_kv_cache_specs,
)
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager
from vllm_ascend.utils import get_kv_cache_tensor_layers, vllm_version_is


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("scheduler_size", [16, 32])
def test_real_tail_coordinator_preserves_prefix_hit_and_private_lifecycle(wrapped, scheduler_size):
    from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager

    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    full = FullAttentionSpec(block_size=scheduler_size, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    tail = AscendIndexerKPoolTailSpec(
        block_size=4, sliding_window=4, compress_ratio=4, num_kv_heads=1, head_size=8, dtype=torch.float32
    )
    group_tail = UniformTypeKVCacheSpecs.from_specs({"tail": tail}) if wrapped else tail
    cfg = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["full"], full), KVCacheGroupSpec(["tail"], group_tail)],
    )
    coordinator = AscendHybridKVCacheCoordinator(
        cfg,
        max_model_len=4096,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=16,
        scheduler_block_size=scheduler_size,
        max_in_flight_tokens=64,
    )
    full_mgr, tail_mgr = coordinator.single_type_managers
    assert isinstance(tail_mgr, KpoolTailManager)
    assert len(coordinator.attention_groups) == 1
    full_mgr.allocate_new_blocks("old", 64, 64)
    tail_mgr.allocate_new_blocks("old", 64, 64)
    hashes = [bytes([i + 1]) * 32 for i in range(4)]
    request = SimpleNamespace(request_id="old", block_hashes=hashes, num_prompt_tokens=65, shared_prefix_boundary=None)
    # Exercise the real coordinator, including main's replay_boundary keyword.
    coordinator.cache_blocks(request, 64)
    if not vllm_version_is("0.28.0"):
        assert full_mgr.cache_hit_alignment_tokens == scheduler_size
        assert tail_mgr.cache_hit_alignment_tokens == scheduler_size
    assert tail_mgr.req_to_blocks["old"][0].block_hash is None
    full_mgr.free("old")
    tail_mgr.free("old")
    result = coordinator.find_longest_cache_hit(hashes, 64)
    assert result[1] == 64
    assert len(result[0]) == 2
    assert len(result[0][0]) == 64 // scheduler_size
    assert result[0][1] == []
    tail_mgr.allocate_external_computed_blocks("new", 64, 0)
    assert len(tail_mgr.req_to_blocks["new"]) == 1
    tail_mgr.free("new")


def _make_kv_cache_tensor(size: int, layer_names: list[str]) -> KVCacheTensor:
    """Build a KVCacheTensor; vLLM #51718 renamed shared_by -> layers on main."""
    if vllm_version_is("0.28.0"):
        return KVCacheTensor(size=size, shared_by=layer_names)
    return KVCacheTensor(size=size, layers=layer_names, layer_stride=0, block_stride=0, offset=0)


@pytest.mark.skipif(vllm_version_is("0.28.0"), reason="separate write-mask alignment is a main API")
def test_partial_hash_alignment_reaches_real_managers_and_cached_blocks():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    specs = [
        FullAttentionSpec(block_size=32, num_kv_heads=1, head_size=8, dtype=torch.bfloat16),
        MambaSpec(block_size=64, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align"),
        SlidingWindowMLASpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, sliding_window=16),
        AscendIndexerKPoolTailSpec(
            block_size=4, sliding_window=4, compress_ratio=4, num_kv_heads=1, head_size=8, dtype=torch.float32
        ),
    ]
    cfg = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([str(index)], spec) for index, spec in enumerate(specs)],
    )
    coordinator = AscendHybridKVCacheCoordinator(
        cfg,
        max_model_len=4096,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=16,
        scheduler_block_size=64,
        max_in_flight_tokens=64,
    )
    assert coordinator.enable_partial_hash_hits
    assert coordinator._cache_hit_alignment_tokens == 16
    assert all(manager.cache_hit_alignment_tokens == 16 for manager in coordinator.single_type_managers)
    hashes = [bytes([index + 1]) * 32 for index in range(8)]
    request = SimpleNamespace(
        request_id="partial", block_hashes=hashes, num_prompt_tokens=129, shared_prefix_boundary=None
    )
    for manager in coordinator.single_type_managers:
        manager.allocate_new_blocks(request.request_id, 128, 128)
    coordinator.cache_blocks(request, 128)
    assert coordinator.single_type_managers[-1].req_to_blocks[request.request_id][0].block_hash is None
    coordinator.free(request.request_id)
    hit = coordinator.find_longest_cache_hit(hashes, 128)
    assert hit[1] == 128
    assert len(hit[0][0]) == 4
    assert hit[0][-1] == []
    # The 16-token SWA boundary lies inside the 64-token scheduler segment.
    # With the stale scheduler write-mask this first block is not cached.
    swa_hit = SlidingWindowManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=16,
        kv_cache_group_ids=[2],
        block_pool=coordinator.block_pool,
        kv_cache_spec=specs[2],
        drop_eagle_block=False,
        alignment_tokens=16,
        dcp_world_size=1,
        pcp_world_size=1,
    )
    assert swa_hit[1] == 16
    assert len(swa_hit[0][0]) == 1


def _ratio_kwargs(ratio: int) -> dict[str, int]:
    """vLLM #51718 renamed compress_ratio to tokens_per_state on main."""
    return {"compress_ratio": ratio} if vllm_version_is("0.28.0") else {"tokens_per_state": ratio}


def _make_hybrid_kv_cache_config(
    full_block_size: int = 16,
    mamba_block_size: int = 16,
) -> KVCacheConfig:
    full_spec = FullAttentionSpec(
        block_size=full_block_size,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=mamba_block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            _make_kv_cache_tensor(full_spec.page_size_bytes * 10, ["attn"]),
            _make_kv_cache_tensor(mamba_spec.page_size_bytes * 10, ["mamba"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=full_spec),
            KVCacheGroupSpec(layer_names=["mamba"], kv_cache_spec=mamba_spec),
        ],
    )


def _make_kimi_k3_dspark_kv_cache_specs(
    *,
    block_size: int = 384,
    mamba_block_size: int | None = None,
    page_size: int = 488448,
    target_layer_count: int = 24,
    draft_layer_count: int = 5,
    mamba_layer_count: int = 69,
    draft_uses_mla: bool = False,
    draft_replication_size: int = 1,
) -> dict:
    target_mla_spec = AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
        cache_dtype_str="auto",
    )
    if draft_uses_mla:
        draft_attention_spec = AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            page_size_padded=page_size,
            cache_dtype_str="auto",
            non_causal_multi_token_decode=True,
        )
    else:
        draft_attention_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=64,
            dtype=torch.bfloat16,
            page_size_padded=page_size,
        )
        if draft_replication_size > 1:
            draft_attention_spec = AscendDCPReplicatedDraftAttentionSpec.from_full_attention_spec(
                draft_attention_spec,
                draft_replication_size,
            )
    mamba_spec = MambaSpec(
        block_size=mamba_block_size or block_size,
        shapes=((10, 2304), (6, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page_size,
        mamba_cache_mode="align",
        num_speculative_blocks=7,
    )
    specs = {
        f"language_model.model.layers.{layer_idx}.self_attn.attn": target_mla_spec
        for layer_idx in range(target_layer_count)
    }
    specs.update(
        {
            f"model.layers.{layer_idx}.self_attn.attn": draft_attention_spec
            for layer_idx in range(93, 93 + draft_layer_count)
        }
    )
    specs.update(
        {f"language_model.model.layers.{layer_idx}.self_attn": mamba_spec for layer_idx in range(mamba_layer_count)}
    )
    return specs


def _make_deepseek_v4_kv_cache_config() -> KVCacheConfig:
    c4_spec = MLAAttentionSpec(
        block_size=128 * 4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        **_ratio_kwargs(4),
        model_version="deepseek_v4",
    )
    c128_spec = MLAAttentionSpec(
        block_size=128 * 128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        **_ratio_kwargs(128),
        model_version="deepseek_v4",
    )
    c4_group_spec = UniformTypeKVCacheSpecs.from_specs({"c4_attn": c4_spec})
    c128_group_spec = UniformTypeKVCacheSpecs.from_specs({"c128_attn": c128_spec})
    assert c4_group_spec is not None
    assert c128_group_spec is not None
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            _make_kv_cache_tensor(c4_spec.page_size_bytes * 10, ["c4_attn"]),
            _make_kv_cache_tensor(c128_spec.page_size_bytes * 10, ["c128_attn"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["c4_attn"], kv_cache_spec=c4_group_spec),
            KVCacheGroupSpec(layer_names=["c128_attn"], kv_cache_spec=c128_group_spec),
        ],
    )


def _make_vllm_config(
    *,
    enable_prefix_caching: bool,
    dcp: int,
    block_size: int = 16,
    prefix_match_unit: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
            mamba_cache_mode="align",
            prefix_match_unit=prefix_match_unit,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
        ),
        kv_transfer_config=None,
    )


def _make_coordinator_for_effective_block_size(
    *,
    dcp_world_size: int,
    enable_caching: bool,
) -> AscendHybridKVCacheCoordinator:
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.dcp_world_size = dcp_world_size
    coordinator.enable_caching = enable_caching
    return coordinator


@pytest.mark.parametrize("prefix_match_unit,expected_hash", [(None, 128), (64, 64)])
def test_dcp1_hybrid_hash_unit_divides_every_group(monkeypatch, prefix_match_unit, expected_hash):
    config = _make_vllm_config(enable_prefix_caching=True, dcp=1, block_size=128)
    config.cache_config.prefix_match_unit = prefix_match_unit
    cache = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=FullAttentionSpec(
                    block_size=128,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.bfloat16,
                )
            ),
            SimpleNamespace(
                kv_cache_spec=MambaSpec(
                    block_size=3072,
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                )
            ),
        ],
    )
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_utils._orig_resolve_kv_cache_block_sizes",
        lambda *_: (3072, 3072),
    )
    assert _ascend_resolve_kv_cache_block_sizes(cache, config) == (3072, expected_hash)
    config.cache_config.prefix_match_unit = 96
    with pytest.raises(ValueError, match="Invalid prefix_match_unit"):
        _ascend_resolve_kv_cache_block_sizes(cache, config)


def test_ascend_mla_page_size_includes_scale_storage() -> None:
    spec = AscendMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        scale_dim=1,
        scale_dtype=torch.float16,
    )

    expected_page_size = 16 * (128 * 2 + 2)
    assert spec.unpadded_page_size_bytes == expected_page_size
    assert spec.real_page_size_bytes == expected_page_size
    assert spec.page_size_bytes == expected_page_size


def test_ascend_mla_merge_preserves_upstream_layout_fields() -> None:
    legacy_layout_kwargs = {"indexes_kv_by_block_stride": True} if vllm_version_is("0.28.0") else {}
    spec = AscendMLAAttentionSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        cache_dtype_str="fp8_ds_mla",
        page_size_padded=(512 // 4) * (128 * 2 + 2) + 128,
        model_version="deepseek_v4",
        scale_dim=1,
        scale_dtype=torch.float16,
        **_ratio_kwargs(4),
        **legacy_layout_kwargs,
    )

    merged = AscendMLAAttentionSpec.merge([spec, replace(spec)])

    assert merged.block_size == spec.block_size
    assert merged.real_page_size_bytes == (512 // 4) * (128 * 2 + 2)
    assert merged.page_size_bytes == spec.page_size_padded
    ratio_field = "compress_ratio" if vllm_version_is("0.28.0") else "tokens_per_state"
    assert getattr(merged, ratio_field) == getattr(spec, ratio_field)
    assert merged.model_version == spec.model_version
    if vllm_version_is("0.28.0"):
        assert merged.indexes_kv_by_block_stride == spec.indexes_kv_by_block_stride
    assert merged.scale_dim == spec.scale_dim
    assert merged.scale_dtype == spec.scale_dtype


@pytest.mark.parametrize(
    ("enable_prefix_caching", "expected_hash_block_size"),
    [
        pytest.param(False, math.lcm(16, 32) * 2, id="dcp-without-prefix-caching"),
        pytest.param(True, math.gcd(16, 32), id="dcp-with-prefix-caching"),
    ],
)
def test_resolve_kv_cache_block_sizes_with_cp_hybrid_groups(
    enable_prefix_caching: bool,
    expected_hash_block_size: int,
) -> None:
    kv_cache_config = _make_hybrid_kv_cache_config(full_block_size=16, mamba_block_size=32)
    vllm_config = _make_vllm_config(
        enable_prefix_caching=enable_prefix_caching,
        dcp=2,
    )

    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        kv_cache_config,
        vllm_config,
    )

    expected_scheduler_block_size = math.lcm(16, 32) * 2
    assert scheduler_block_size == expected_scheduler_block_size
    assert hash_block_size == expected_hash_block_size


@pytest.mark.parametrize(
    ("full_block_size", "prefix_match_unit", "expected_hash_block_size"),
    [
        pytest.param(128, None, 128, id="default-prefix-match-unit"),
        pytest.param(384, None, 384, id="non-power-of-two-full-block"),
        pytest.param(384, 8, 8, id="configured-prefix-match-unit"),
        pytest.param(384, 32, 32, id="prefix-match-unit-larger-than-tail"),
    ],
)
def test_glm5_next_hashes_exclude_private_tail_after_engine_min_block_update(
    full_block_size: int,
    prefix_match_unit: int | None,
    expected_hash_block_size: int,
) -> None:
    main_spec = MLAAttentionSpec(
        block_size=full_block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        model_version="glm5_next",
    )
    indexer_spec = MLAAttentionSpec(
        block_size=full_block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        model_version="glm5_next",
        **_ratio_kwargs(16),
    )
    state_spec = AscendIndexerKPoolTailSpec(
        block_size=16,
        sliding_window=16,
        compress_ratio=16,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.float32,
        model_version="glm5_next",
        cache_role="indexer_tail",
    )
    full_group_spec = UniformTypeKVCacheSpecs.from_specs({"layer.main": main_spec, "layer.indexer": indexer_spec})
    state_group_spec = UniformTypeKVCacheSpecs.from_specs({"layer.state": state_spec})
    mamba_spec = MambaSpec(
        block_size=full_block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    mamba_group_spec = UniformTypeKVCacheSpecs.from_specs({"layer.mamba": mamba_spec})
    assert full_group_spec is not None
    assert state_group_spec is not None
    assert mamba_group_spec is not None

    worker_config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer.main", "layer.indexer"],
                kv_cache_spec=full_group_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["layer.state"],
                kv_cache_spec=state_group_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["layer.mamba"],
                kv_cache_spec=mamba_group_spec,
            ),
        ],
    )
    scheduler_config = generate_scheduler_kv_cache_config([worker_config])
    scheduler_full_spec = scheduler_config.kv_cache_groups[0].kv_cache_spec
    assert isinstance(scheduler_full_spec, MLAAttentionSpec)
    assert scheduler_full_spec.head_size == main_spec.head_size
    ratio_field = "compress_ratio" if vllm_version_is("0.28.0") else "tokens_per_state"
    assert getattr(scheduler_full_spec, ratio_field) == 1

    vllm_config = _make_vllm_config(
        enable_prefix_caching=True,
        dcp=1,
        block_size=full_block_size,
        prefix_match_unit=prefix_match_unit,
    )
    # Match EngineCore: the global block size becomes the smallest unwrapped
    # scheduler group size. The private tail must not limit prefix hashing.
    vllm_config.cache_config.block_size = min(
        group.kv_cache_spec.block_size for group in scheduler_config.kv_cache_groups
    )
    assert vllm_config.cache_config.block_size == 16
    assert state_spec.block_size == state_spec.compress_ratio == 16
    assert not state_spec.prefix_cacheable
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        scheduler_config,
        vllm_config,
    )
    assert (scheduler_block_size, hash_block_size) == (
        full_block_size,
        expected_hash_block_size,
    )

    hashes_per_full_block = full_block_size // hash_block_size
    base_hashes = [index.to_bytes(2, "little") for index in range(2 * hashes_per_full_block)]
    full_group_hashes = BlockHashListWithBlockSize(
        base_hashes,
        hash_block_size,
        scheduler_full_spec.block_size,
    )
    assert len(full_group_hashes) == 2
    # vLLM hashes are chained across prefix blocks, so the last state-sized
    # hash in a full block is already that full block's hash.
    assert full_group_hashes[0] == base_hashes[hashes_per_full_block - 1]


def test_glm5_next_single_cacheable_group_ignores_private_tail_block_size() -> None:
    """A standalone MTP runner has a full group and a private tail only."""
    full_spec = MLAAttentionSpec(
        block_size=384,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        model_version="glm5_next",
    )
    tail_spec = AscendIndexerKPoolTailSpec(
        block_size=16,
        sliding_window=16,
        compress_ratio=16,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.float32,
        model_version="glm5_next",
        cache_role="indexer_tail",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.full"], kv_cache_spec=full_spec),
            KVCacheGroupSpec(layer_names=["layer.tail"], kv_cache_spec=tail_spec),
        ],
    )
    vllm_config = _make_vllm_config(
        enable_prefix_caching=True,
        dcp=1,
        block_size=16,
    )

    # Match EngineCore: the tail participates in the global minimum before
    # resolve filters request-private groups from prefix-cache planning.
    assert vllm_config.cache_config.block_size == tail_spec.block_size
    assert not tail_spec.prefix_cacheable

    assert _ascend_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config) == (384, 384)


def test_deepseek_v4_groups_use_logical_sizes_and_full_attention_manager() -> None:
    c128_spec = MLAAttentionSpec(
        block_size=128 * 128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        **_ratio_kwargs(128),
        model_version="deepseek_v4",
    )
    c4_spec = MLAAttentionSpec(
        block_size=128 * 4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        **_ratio_kwargs(4),
        model_version="deepseek_v4",
    )
    swa_spec = SlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        sliding_window=512,
    )

    grouped_specs = group_and_unify_kv_cache_specs(
        {
            "c128": c128_spec,
            "swa": swa_spec,
            "c4": c4_spec,
        }
    )

    assert grouped_specs is not None
    assert [group.block_size for group in grouped_specs[:2]] == [512, 16384]
    for group in grouped_specs[:2]:
        spec = next(iter(group.kv_cache_specs.values()))
        assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager


@pytest.mark.parametrize(
    ("block_size", "page_size", "draft_uses_mla"),
    [
        pytest.param(384, 488448, False, id="gqa-tp16"),
        pytest.param(768, 976896, False, id="gqa-tp8"),
        pytest.param(384, 488448, True, id="mla-tp16"),
    ],
)
def test_kimi_k3_dspark_uses_four_mixed_kv_groups(block_size, page_size, draft_uses_mla) -> None:
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(
        _make_kimi_k3_dspark_kv_cache_specs(
            block_size=block_size,
            page_size=page_size,
            draft_uses_mla=draft_uses_mla,
        )
    )

    assert groups is not None
    assert [len(group.layer_names) for group in groups] == [29, 23, 23, 23]
    assert all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in groups)
    mixed_specs = groups[0].kv_cache_spec.kv_cache_specs
    expected_mla_count = 29 if draft_uses_mla else 24
    assert sum(isinstance(spec, MLAAttentionSpec) for spec in mixed_specs.values()) == expected_mla_count
    assert all(isinstance(spec, FullAttentionSpec) for spec in mixed_specs.values())


def test_kimi_k3_gqa_mixed_groups_preserve_scheduler_and_mamba_contracts() -> None:
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(_make_kimi_k3_dspark_kv_cache_specs())
    assert groups is not None
    max_model_len = 133120
    vllm_config = _make_vllm_config(enable_prefix_caching=True, dcp=1, block_size=384)
    worker_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )

    assert worker_config.has_mamba_layers
    assert worker_config.needs_kv_cache_zeroing
    worker_mamba_widths = {
        group.kv_cache_spec.max_num_blocks_per_req(vllm_config, max_model_len) for group in groups[1:]
    }
    assert worker_mamba_widths == {354}

    scheduler_config = generate_scheduler_kv_cache_config([worker_config])
    assert isinstance(scheduler_config.kv_cache_groups[0].kv_cache_spec, MLAAttentionSpec)
    assert all(isinstance(group.kv_cache_spec, MambaSpec) for group in scheduler_config.kv_cache_groups[1:])
    scheduler_mamba_widths = {
        group.kv_cache_spec.max_num_blocks_per_req(vllm_config, max_model_len)
        for group in scheduler_config.kv_cache_groups[1:]
    }
    assert scheduler_mamba_widths == worker_mamba_widths
    assert scheduler_config.needs_kv_cache_zeroing


@pytest.mark.skipif(not vllm_version_is("0.28.0"), reason="shared_by planner is only installed on v0.28.0")
def test_kimi_k3_gqa_mixed_groups_use_expected_physical_layout(monkeypatch) -> None:
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(_make_kimi_k3_dspark_kv_cache_specs())
    assert groups is not None
    page_size = 488448
    expected_num_blocks = 100
    available_memory = page_size * 29 * expected_num_blocks
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_utils.may_override_num_blocks",
        lambda _config, num_blocks: num_blocks,
    )

    num_blocks, tensors = _get_kv_cache_config_deepseek_v4(
        SimpleNamespace(),
        groups,
        available_memory,
    )

    assert num_blocks == expected_num_blocks
    assert len(tensors) == 29
    assert [len(get_kv_cache_tensor_layers(tensor)) for tensor in tensors] == [4] * 23 + [1] * 6
    assert all(tensor.size == page_size * expected_num_blocks for tensor in tensors)
    assert sum(tensor.size for tensor in tensors) == available_memory


def test_kimi_k3_none_mamba_uses_separate_scheduler_block_size() -> None:
    page_size = 940032
    specs = _make_kimi_k3_dspark_kv_cache_specs(block_size=768, page_size=page_size)
    for name, spec in list(specs.items()):
        if isinstance(spec, MambaSpec):
            specs[name] = replace(
                spec,
                block_size=200000,
                shapes=((6, 4608), (12, 128, 128)),
                mamba_cache_mode="none",
                num_speculative_blocks=3,
            )
        elif name.startswith("model.layers."):
            specs[name] = replace(spec, num_kv_heads=2)
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(specs)
    assert groups is not None
    assert [len(g.layer_names) for g in groups] == [29, 23, 23, 23]
    assert [g.kv_cache_spec.block_size for g in groups] == [768, 200000, 200000, 200000]
    config = _make_vllm_config(enable_prefix_caching=False, dcp=1, block_size=768)
    config.cache_config.mamba_cache_mode = "none"
    assert {g.kv_cache_spec.max_num_blocks_per_req(config, 200000) for g in groups[1:]} == {4}


def test_kimi_k3_mixed_attention_still_requires_same_block_size() -> None:
    specs = _make_kimi_k3_dspark_kv_cache_specs()
    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is not None
    name = "model.layers.93.self_attn.attn"
    specs[name] = replace(specs[name], block_size=192)
    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is None


def test_kimi_k3_dcp_replicated_draft_uses_minimal_physical_layout(
    monkeypatch,
) -> None:
    replication_size = 8
    page_size = 488448
    specs = _make_kimi_k3_dspark_kv_cache_specs(
        page_size=page_size,
        draft_replication_size=replication_size,
    )
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(specs)
    assert groups is not None
    expected_num_blocks = 100
    draft_page_size = 384 * 1 * (64 + 64) * 2 * replication_size
    bytes_per_block = page_size * 24 + draft_page_size * 5
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_utils.may_override_num_blocks",
        lambda _config, num_blocks: num_blocks,
    )
    config = _get_kimi_k3_replicated_dspark_kv_cache_config(
        SimpleNamespace(cache_config=SimpleNamespace(prefix_cache_retention_interval=None)),
        groups,
        bytes_per_block * expected_num_blocks,
    )

    assert config is not None
    assert config.num_blocks == expected_num_blocks
    assert len(config.kv_cache_tensors) == 29
    assert [len(tensor.shared_by) for tensor in config.kv_cache_tensors] == ([4] * 23 + [1] * 6)
    assert [tensor.size for tensor in config.kv_cache_tensors[:24]] == [page_size * expected_num_blocks] * 24
    assert [tensor.size for tensor in config.kv_cache_tensors[24:]] == [draft_page_size * expected_num_blocks] * 5
    assert sum(tensor.size for tensor in config.kv_cache_tensors) == (bytes_per_block * expected_num_blocks)


def test_kimi_k3_dcp_replicated_pages_bypass_rectangular_unification() -> None:
    specs = _make_kimi_k3_dspark_kv_cache_specs(draft_replication_size=4)
    for name, spec in list(specs.items()):
        if isinstance(spec, AscendDCPReplicatedDraftAttentionSpec):
            specs[name] = replace(spec, page_size_padded=122112)

    unified = _unify_kv_cache_spec_page_size(specs)

    assert unified is not specs
    assert {spec.block_size for spec in unified.values()} == {384}
    assert {
        spec.page_size_bytes for spec in unified.values() if isinstance(spec, AscendDCPReplicatedDraftAttentionSpec)
    } == {4 * 384 * 128 * 2}


def test_kimi_k3_gqa_mixed_grouping_falls_back_on_unrecognized_layer() -> None:
    specs = _make_kimi_k3_dspark_kv_cache_specs()
    specs["unrecognized.layer"] = next(iter(specs.values()))

    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is None


def test_kimi_k3_dcp_replicated_pages_allow_independent_mamba_block_size() -> None:
    specs = _make_kimi_k3_dspark_kv_cache_specs(
        block_size=128,
        mamba_block_size=3072,
        page_size=3907584,
        target_layer_count=3,
        draft_layer_count=5,
        mamba_layer_count=2,
        draft_replication_size=2,
    )

    unified = _unify_kv_cache_spec_page_size(specs)

    attention_block_sizes = {spec.block_size for spec in unified.values() if isinstance(spec, FullAttentionSpec)}
    mamba_block_sizes = {spec.block_size for spec in unified.values() if isinstance(spec, MambaSpec)}
    assert attention_block_sizes == {128}
    assert mamba_block_sizes == {3072}
    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(unified) is not None


def test_kimi_k3_dspark_group_count_is_derived_from_layer_ratio() -> None:
    groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(
        _make_kimi_k3_dspark_kv_cache_specs(
            target_layer_count=20,
            draft_layer_count=4,
            mamba_layer_count=70,
        )
    )

    assert groups is not None
    assert [len(group.layer_names) for group in groups] == [24, 24, 23, 23]


def test_kimi_k3_dspark_mixed_grouping_falls_back_on_unaligned_pages() -> None:
    specs = _make_kimi_k3_dspark_kv_cache_specs()
    draft_layer = "model.layers.93.self_attn.attn"
    specs[draft_layer] = replace(specs[draft_layer], page_size_padded=976896)

    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is None


def test_deepseek_v4_scheduler_lcm_uses_logical_group_sizes() -> None:
    kv_cache_config = _make_deepseek_v4_kv_cache_config()
    vllm_config = _make_vllm_config(
        enable_prefix_caching=True,
        dcp=1,
        block_size=128,
    )

    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        kv_cache_config,
        vllm_config,
    )

    assert scheduler_block_size == 16384
    assert hash_block_size == 512


@pytest.mark.skipif(vllm_version_is("0.28.0"), reason="vLLM #51718 only changed the main planner")
def test_deepseek_v4_main_restores_ascend_shared_tuple_planner(monkeypatch) -> None:
    kv_cache_config = _make_deepseek_v4_kv_cache_config()
    planned_tensor = _make_kv_cache_tensor(4096, ["c4_attn", "c128_attn"])
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(prefix_cache_retention_interval=None),
        model_config=SimpleNamespace(max_model_len=4096),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=1,
    )

    planner = MagicMock(return_value=(7, [planned_tensor]))
    monkeypatch.setattr(kv_cache_utils_patch, "_get_kv_cache_config_deepseek_v4_main", planner)

    result = kv_cache_utils_patch._ascend_get_kv_cache_config_from_groups(
        vllm_config,
        kv_cache_config.kv_cache_groups,
        available_memory=1 << 30,
    )

    planner.assert_called_once_with(vllm_config, kv_cache_config.kv_cache_groups, 1 << 30)
    assert result.num_blocks == 7
    assert result.kv_cache_tensors == [planned_tensor]

    needed_memory = kv_cache_utils_patch._ascend_max_memory_usage_bytes_from_groups(
        vllm_config,
        kv_cache_config.kv_cache_groups,
    )
    full_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_spec, UniformTypeKVCacheSpecs)
    layer_tuple_bytes = sum(spec.page_size_bytes for spec in full_spec.kv_cache_specs.values())
    num_layer_tuples = max(
        kv_cache_utils_patch._get_max_layers_per_page_size(group.kv_cache_spec)
        for group in kv_cache_config.kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )
    expected_memory = sum(
        num_layer_tuples * group.kv_cache_spec.max_memory_usage_pages(vllm_config) * layer_tuple_bytes
        for group in kv_cache_config.kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )
    assert needed_memory == expected_memory


@pytest.mark.skipif(vllm_version_is("0.28.0"), reason="vLLM #51718 introduced shared backing on main")
def test_deepseek_v4_main_planner_uses_shared_backing_geometry(monkeypatch) -> None:
    kv_cache_config = _make_deepseek_v4_kv_cache_config()
    groups = kv_cache_config.kv_cache_groups
    first_group_spec = groups[0].kv_cache_spec
    assert isinstance(first_group_spec, UniformTypeKVCacheSpecs)
    page_size = next(iter(first_group_spec.kv_cache_specs.values())).page_size_bytes
    assert all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        and {spec.page_size_bytes for spec in group.kv_cache_spec.kv_cache_specs.values()} == {page_size}
        for group in groups
    )
    expected_num_blocks = 7
    available_memory = page_size * expected_num_blocks
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_utils.may_override_num_blocks",
        lambda _config, num_blocks: num_blocks,
    )

    num_blocks, tensors = _get_kv_cache_config_deepseek_v4_main(
        SimpleNamespace(),
        groups,
        available_memory,
    )

    assert num_blocks == expected_num_blocks
    assert [tensor.layers for tensor in tensors] == [["c4_attn"], ["c128_attn"]]
    assert {tensor.size for tensor in tensors} == {available_memory}
    for tensor in tensors:
        assert tensor.offset == 0
        assert tensor.layer_stride == available_memory
        assert tensor.block_stride == page_size


@pytest.mark.skipif(vllm_version_is("0.28.0"), reason="vLLM #51718 only re-plans ranks on main")
def test_deepseek_v4_main_rank_replan_preserves_num_blocks() -> None:
    small_page_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float16,
        **_ratio_kwargs(4),
        model_version="deepseek_v4",
    )
    large_page_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        **_ratio_kwargs(4),
        model_version="deepseek_v4",
    )
    group_spec = UniformTypeKVCacheSpecs.from_specs(
        {
            "small_attn": small_page_spec,
            "large_attn": large_page_spec,
            "mtp_attn": large_page_spec,
        }
    )
    assert group_spec is not None
    kv_cache_groups = [
        KVCacheGroupSpec(
            layer_names=["small_attn", "large_attn", "mtp_attn"],
            kv_cache_spec=group_spec,
        )
    ]
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
        )
    )

    ascend_bytes_per_block = kv_cache_utils_patch._ascend_pool_bytes_per_block(kv_cache_groups)
    upstream_bytes_per_block = kv_cache_utils_patch._orig_pool_bytes_per_block(kv_cache_groups)
    assert ascend_bytes_per_block != upstream_bytes_per_block
    assert vllm_kv_cache_utils._pool_bytes_per_block is kv_cache_utils_patch._ascend_pool_bytes_per_block

    expected_num_blocks = 7
    replanned_config = kv_cache_utils_patch._ascend_get_kv_cache_config_from_groups(
        vllm_config,
        kv_cache_groups,
        expected_num_blocks * ascend_bytes_per_block,
    )

    assert replanned_config.num_blocks == expected_num_blocks
    tuple_stride = (small_page_spec.page_size_bytes + large_page_spec.page_size_bytes) * expected_num_blocks
    backing_size = tuple_stride * 2
    tensors_by_layers = {tuple(tensor.layers): tensor for tensor in replanned_config.kv_cache_tensors}
    assert set(tensors_by_layers) == {("small_attn",), ("large_attn",), ("mtp_attn",)}
    assert {tensor.size for tensor in tensors_by_layers.values()} == {backing_size}

    small_tensor = tensors_by_layers[("small_attn",)]
    assert small_tensor.offset == 0
    assert small_tensor.layer_stride == tuple_stride
    assert small_tensor.block_stride == small_page_spec.page_size_bytes

    large_page_offset = small_page_spec.page_size_bytes * expected_num_blocks
    large_tensor = tensors_by_layers[("large_attn",)]
    assert large_tensor.offset == large_page_offset
    assert large_tensor.layer_stride == tuple_stride
    assert large_tensor.block_stride == large_page_spec.page_size_bytes

    mtp_tensor = tensors_by_layers[("mtp_attn",)]
    assert mtp_tensor.offset == tuple_stride + large_page_offset
    assert mtp_tensor.layer_stride == 0
    assert mtp_tensor.block_stride == large_page_spec.page_size_bytes


@pytest.mark.parametrize(
    ("spec_factory", "dcp", "enable_caching", "expected"),
    [
        pytest.param(
            lambda: FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=64,
                dtype=torch.float16,
            ),
            2,
            True,
            32,
            id="full-attention-scales-with-dcp",
        ),
        pytest.param(
            lambda: MambaSpec(
                block_size=16,
                shapes=((1,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="none",
            ),
            2,
            True,
            16,
            id="mamba-keeps-physical-block-size-with-prefix-caching",
        ),
        pytest.param(
            lambda: FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=64,
                dtype=torch.float16,
            ),
            1,
            True,
            16,
            id="full-attention-no-cp",
        ),
    ],
)
def test_get_effective_block_size(
    spec_factory,
    dcp: int,
    enable_caching: bool,
    expected: int,
) -> None:
    coordinator = _make_coordinator_for_effective_block_size(
        dcp_world_size=dcp,
        enable_caching=enable_caching,
    )

    assert coordinator._get_effective_block_size(spec_factory()) == expected


def test_get_kv_cache_coordinator_delegates_single_group(monkeypatch) -> None:
    sentinel = object()
    kv_cache_config = _make_hybrid_kv_cache_config(full_block_size=16, mamba_block_size=16)
    single_group_config = KVCacheConfig(
        num_blocks=kv_cache_config.num_blocks,
        kv_cache_tensors=kv_cache_config.kv_cache_tensors[:1],
        kv_cache_groups=kv_cache_config.kv_cache_groups[:1],
    )

    def _fake_orig(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator._orig_get_kv_cache_coordinator",
        _fake_orig,
    )

    coordinator = get_kv_cache_coordinator(
        single_group_config,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=16,
    )

    assert coordinator is sentinel


def test_get_kv_cache_coordinator_delegates_hybrid_without_caching(monkeypatch) -> None:
    sentinel = object()
    kv_cache_config = _make_hybrid_kv_cache_config(full_block_size=16, mamba_block_size=16)

    def _fake_orig(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator._orig_get_kv_cache_coordinator",
        _fake_orig,
    )

    coordinator = get_kv_cache_coordinator(
        kv_cache_config,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=2,
        pcp_world_size=1,
        hash_block_size=16,
    )

    assert coordinator is sentinel


def test_get_kv_cache_coordinator_uses_ascend_for_deepseek_v4(monkeypatch) -> None:
    sentinel = object()
    kv_cache_config = _make_deepseek_v4_kv_cache_config()

    def _fake_orig(*args, **kwargs):
        raise AssertionError("DeepSeek V4 should use AscendHybridKVCacheCoordinator")

    def _fake_ascend_coordinator(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator._orig_get_kv_cache_coordinator",
        _fake_orig,
    )
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_coordinator.AscendHybridKVCacheCoordinator",
        _fake_ascend_coordinator,
    )

    coordinator = get_kv_cache_coordinator(
        kv_cache_config,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=128,
    )

    assert coordinator is sentinel


class _FakeEagleManager:
    def __init__(self) -> None:
        self.use_eagle = False


def test_verify_and_split_propagates_eagle_to_managers() -> None:
    """Regression for DeepSeek-V4 prefix-cache hit rate 0% with MTP/EAGLE.

    The eagle bit must reach each single-type manager: the SWA write path
    (``cache_blocks`` -> ``reachable_block_mask``) keys the retained checkpoint
    tail on ``manager.use_eagle``, while the read path
    (``find_longest_cache_hit``) applies ``drop_eagle_block`` to the same
    groups. If the manager keeps the default ``use_eagle=False`` the retained
    tail is one block short of the eagle peek boundary, the SWA group never
    hits, and the min-over-groups hybrid hit collapses to 0%.
    """
    kv_cache_config = _make_deepseek_v4_kv_cache_config()

    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.kv_cache_config = kv_cache_config
    coordinator.dcp_world_size = 1
    coordinator.enable_caching = True
    # The c128 group (index 1) carries the EAGLE/MTP layers.
    coordinator.eagle_group_ids = {1}

    coordinator.single_type_managers = (_FakeEagleManager(), _FakeEagleManager())

    coordinator.verify_and_split_kv_cache_groups()

    assert coordinator.single_type_managers[1].use_eagle is True
    assert coordinator.single_type_managers[0].use_eagle is False


def test_verify_and_split_propagates_eagle_to_merged_spec_siblings() -> None:
    """Upstream ``_annotate_eagle_groups_deepseek_v4`` flags only the single
    group holding the MTP layer, but the read path merges same-spec groups and
    applies ``drop_eagle_block`` to the whole merged group. So every sibling
    sharing that spec must also get ``use_eagle=True`` on the write path, else
    ``get_cached_block`` (which needs the block cached for *all* group ids)
    misses and the hit collapses to 0%.
    """
    base_config = _make_deepseek_v4_kv_cache_config()
    # Reuse the c128 spec object so the two c128 groups compare equal and merge
    # into one attention group in verify_and_split.
    c128_group_spec = base_config.kv_cache_groups[1].kv_cache_spec
    kv_cache_config = KVCacheConfig(
        num_blocks=base_config.num_blocks,
        kv_cache_tensors=base_config.kv_cache_tensors,
        kv_cache_groups=[
            base_config.kv_cache_groups[0],  # c4   -> gid 0 (distinct spec)
            base_config.kv_cache_groups[1],  # c128 -> gid 1
            KVCacheGroupSpec(layer_names=["c128_attn_mtp"], kv_cache_spec=c128_group_spec),  # gid 2
        ],
    )

    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.kv_cache_config = kv_cache_config
    coordinator.dcp_world_size = 1
    coordinator.enable_caching = True
    # Only the MTP sibling (gid 2) is flagged, exactly as upstream does.
    coordinator.eagle_group_ids = {2}

    coordinator.single_type_managers = (
        _FakeEagleManager(),
        _FakeEagleManager(),
        _FakeEagleManager(),
    )

    coordinator.verify_and_split_kv_cache_groups()

    # Both gid 1 and gid 2 share the c128 spec and merge, so both must be eagle.
    assert coordinator.single_type_managers[1].use_eagle is True
    assert coordinator.single_type_managers[2].use_eagle is True
    assert coordinator.single_type_managers[0].use_eagle is False


def test_deepseek_v4_detection_handles_non_mapping_nested_specs() -> None:
    kv_cache_spec = SimpleNamespace(
        kv_cache_specs=[
            SimpleNamespace(model_version="deepseek_v4"),
        ]
    )
    unknown_spec = SimpleNamespace(kv_cache_specs=object())

    assert _is_deepseek_v4_kv_cache_spec(kv_cache_spec)
    assert not _is_deepseek_v4_kv_cache_spec(unknown_spec)


def test_ascend_mamba_manager_uses_logical_block_size_with_prefix_caching() -> None:
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    block_pool = BlockPool(
        10,
        True,
        16,
        False,
        MagicMock(),
    )

    manager_kwargs = dict(
        kv_cache_spec=mamba_spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=1,
        dcp_world_size=2,
        pcp_world_size=1,
    )
    manager_kwargs["scheduler_block_size"] = mamba_spec.block_size
    manager = AscendMambaManager(**manager_kwargs)

    assert manager.block_size == mamba_spec.block_size


def test_ascend_mamba_cache_lookup_ignores_dcp_sharding() -> None:
    """Mamba states are replicated, unlike DCP-sharded attention KV cache."""
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )

    # The patch module replaces vLLM's public MambaManager symbol with the
    # Ascend subclass, so patch the subclass's direct base explicitly.
    base_mamba_manager = AscendMambaManager.__base__
    assert base_mamba_manager is not None
    with patch.object(
        base_mamba_manager,
        "find_longest_cache_hit",
        return_value=((), 0),
    ) as find_cache_hit:
        AscendMambaManager.find_longest_cache_hit(
            block_hashes=[],
            max_length=0,
            kv_cache_group_ids=[1],
            block_pool=MagicMock(),
            kv_cache_spec=mamba_spec,
            alignment_tokens=16,
            dcp_world_size=8,
            pcp_world_size=1,
            drop_eagle_block=False,
        )

    assert find_cache_hit.call_args.kwargs["dcp_world_size"] == 1


def test_swa_reachable_block_mask_sparse_with_lcm_alignment() -> None:
    """Regression: when ``scheduler_block_size`` is aligned to ``lcm_block_size``
    (instead of the raw-block-size LCM), ``SlidingWindowManager.reachable_block_mask``
    must produce a sparse mask rather than returning ``None``.

    Before the fix, ``alignment_tokens`` was the LCM of raw block_sizes (e.g. 32),
    making ``need >= per_segment`` always true for Ascend's SWA configuration and
    the mask returned ``None`` (cache everything). After the fix the alignment is
    ``lcm_block_size`` (e.g. 4096), which is large enough that only the tail
    blocks within each segment need caching.
    """
    spec = SlidingWindowMLASpec(
        block_size=32,  # Ascend SWA block_size (--block-size 32)
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
        sliding_window=128,  # DeepSeek V4 window
        **_ratio_kwargs(1),
    )
    alignment_tokens = 4096  # lcm_block_size

    mask = SlidingWindowManager.reachable_block_mask(
        start_block=0,
        end_block=256,  # 256 × 32 = 8192 tokens (2 × alignment_tokens)
        alignment_tokens=alignment_tokens,
        kv_cache_spec=spec,
        use_eagle=False,
        retention_interval=None,
    )

    # Must produce a sparse mask, not None.
    assert mask is not None, "should produce sparse mask with lcm alignment"

    true_blocks = sum(mask)

    # need = cdiv(window−1, block_size) = cdiv(127, 32) = 4
    # per_segment = alignment_tokens // block_size = 4096 // 32 = 128
    # Each 128-block segment caches the last 4 blocks (= 0 % sparse padding).
    total_blocks = len(mask)
    expected = 4 * (total_blocks // 128)
    assert true_blocks == expected, (
        f"expected {expected} cached blocks ({4}/{128} per segment), got {true_blocks}/{total_blocks}"
    )
    assert true_blocks > 0 and true_blocks < total_blocks, f"mask should be sparse, got {true_blocks}/{total_blocks}"


@pytest.mark.parametrize("replication_size", [1, 2, 8])
@pytest.mark.parametrize("block_size", [128, 384])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("head_size_v", [64, 128])
def test_replicated_draft_page_sizes_match_base_spec(replication_size, block_size, padded, head_size_v):
    base = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=64,
        head_size_v=head_size_v,
        dtype=torch.bfloat16,
    )
    if padded:
        base = replace(base, page_size_padded=base.unpadded_page_size_bytes + 4096)
    spec = AscendDCPReplicatedDraftAttentionSpec.from_full_attention_spec(base, replication_size)
    assert spec.block_size == block_size
    assert spec.lane_page_size_bytes == base.unpadded_page_size_bytes
    assert spec.page_size_bytes == replication_size * base.unpadded_page_size_bytes
    assert spec.real_page_size_bytes == replication_size * base.real_page_size_bytes
    assert spec.unpadded_page_size_bytes == replication_size * base.unpadded_page_size_bytes
