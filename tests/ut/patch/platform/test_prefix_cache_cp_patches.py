# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
    _is_deepseek_v4_kv_cache_spec,
    get_kv_cache_coordinator,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_resolve_kv_cache_block_sizes,
)
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager


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
            KVCacheTensor(size=full_spec.page_size_bytes * 10, shared_by=["attn"]),
            KVCacheTensor(size=mamba_spec.page_size_bytes * 10, shared_by=["mamba"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=full_spec),
            KVCacheGroupSpec(layer_names=["mamba"], kv_cache_spec=mamba_spec),
        ],
    )


def _make_deepseek_v4_kv_cache_config() -> KVCacheConfig:
    c4_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        model_version="deepseek_v4",
    )
    c128_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=128,
        model_version="deepseek_v4",
    )
    c4_group_spec = UniformTypeKVCacheSpecs.from_specs({"c4_attn": c4_spec})
    c128_group_spec = UniformTypeKVCacheSpecs.from_specs({"c128_attn": c128_spec})
    assert c4_group_spec is not None
    assert c128_group_spec is not None
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=c4_spec.page_size_bytes * 10, shared_by=["c4_attn"]),
            KVCacheTensor(size=c128_spec.page_size_bytes * 10, shared_by=["c128_attn"]),
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
    pcp: int,
    block_size: int = 16,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
    )


def _make_coordinator_for_effective_block_size(
    *,
    dcp_world_size: int,
    pcp_world_size: int,
    enable_caching: bool,
) -> AscendHybridKVCacheCoordinator:
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.dcp_world_size = dcp_world_size
    coordinator.pcp_world_size = pcp_world_size
    coordinator.enable_caching = enable_caching
    return coordinator


@pytest.mark.parametrize(
    ("enable_prefix_caching", "expected_hash_block_size"),
    [
        pytest.param(False, math.lcm(16, 32) * 2 * 2, id="cp-without-prefix-caching"),
        pytest.param(True, math.gcd(16, 32), id="cp-with-prefix-caching"),
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
        pcp=2,
    )

    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        kv_cache_config,
        vllm_config,
    )

    expected_scheduler_block_size = math.lcm(16, 32) * 2 * 2
    assert scheduler_block_size == expected_scheduler_block_size
    assert hash_block_size == expected_hash_block_size


@pytest.mark.parametrize(
    ("spec_factory", "dcp", "pcp", "enable_caching", "expected"),
    [
        pytest.param(
            lambda: FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=64,
                dtype=torch.float16,
            ),
            2,
            2,
            True,
            64,
            id="full-attention-scales-with-cp",
        ),
        pytest.param(
            lambda: MambaSpec(
                block_size=16,
                shapes=((1,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="none",
            ),
            2,
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
    pcp: int,
    enable_caching: bool,
    expected: int,
) -> None:
    coordinator = _make_coordinator_for_effective_block_size(
        dcp_world_size=dcp,
        pcp_world_size=pcp,
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
        pcp_world_size=2,
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
    coordinator.pcp_world_size = 1
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
    coordinator.pcp_world_size = 1
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
        pcp_world_size=2,
    )
    manager_kwargs["scheduler_block_size"] = mamba_spec.block_size
    manager = AscendMambaManager(**manager_kwargs)

    assert manager.block_size == mamba_spec.block_size
