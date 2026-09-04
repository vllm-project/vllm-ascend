# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import vllm.v1.core.kv_cache_utils as vllm_kv_cache_utils
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
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
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    get_storage_block_size,
)
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
    _is_deepseek_v4_kv_cache_spec,
    get_kv_cache_coordinator,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_resolve_kv_cache_block_sizes,
    _get_kimi_k3_dspark_mixed_kv_cache_groups,
    _get_kv_cache_config_deepseek_v4,
    _get_kv_cache_config_deepseek_v4_main,
    group_and_unify_kv_cache_specs,
)
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager
from vllm_ascend.utils import get_kv_cache_tensor_layers, vllm_version_is


def _make_kv_cache_tensor(size: int, layer_names: list[str]) -> KVCacheTensor:
    """Build a KVCacheTensor; vLLM #51718 renamed shared_by -> layers on main."""
    if vllm_version_is("0.27.1"):
        return KVCacheTensor(size=size, shared_by=layer_names)
    return KVCacheTensor(size=size, layers=layer_names, layer_stride=0, block_stride=0, offset=0)


def _ratio_kwargs(ratio: int) -> dict[str, int]:
    """vLLM #51718 renamed compress_ratio to tokens_per_state on main."""
    return {"compress_ratio": ratio} if vllm_version_is("0.27.1") else {"tokens_per_state": ratio}


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
    page_size: int = 488448,
    target_layer_count: int = 24,
    draft_layer_count: int = 5,
    mamba_layer_count: int = 69,
    draft_uses_mla: bool = False,
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
    mamba_spec = MambaSpec(
        block_size=block_size,
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
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
            mamba_cache_mode="align",
            prefix_match_unit=None,
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
    legacy_layout_kwargs = {"indexes_kv_by_block_stride": True} if vllm_version_is("0.27.1") else {}
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
    ratio_field = "compress_ratio" if vllm_version_is("0.27.1") else "tokens_per_state"
    assert getattr(merged, ratio_field) == getattr(spec, ratio_field)
    assert merged.model_version == spec.model_version
    if vllm_version_is("0.27.1"):
        assert merged.indexes_kv_by_block_stride == spec.indexes_kv_by_block_stride
    assert merged.scale_dim == spec.scale_dim
    assert merged.scale_dtype == spec.scale_dtype


def test_ascend_mla_storage_geometry_survives_upstream_optional_field() -> None:
    spec = AscendMLAAttentionSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        **_ratio_kwargs(4),
    )
    assert get_storage_block_size(spec) == 128
    resized = replace(spec, block_size=1024)
    assert get_storage_block_size(resized) == 256
    merged = AscendMLAAttentionSpec.merge([spec, replace(spec)])
    assert get_storage_block_size(merged) == 128
    assert merged.page_size_bytes == 128 * 128 * 2
    if vllm_version_is("0.27.1"):
        assert spec.storage_block_size == 128
    else:
        # #53906's optional override must not be populated with Ascend's
        # derived physical size, or upstream metadata building changes lanes.
        assert spec.storage_block_size is None
        assert resized.storage_block_size is None
        assert merged.storage_block_size is None


@pytest.mark.parametrize("spec_cls", [MLAAttentionSpec, AscendSFAIndexerCacheSpec])
def test_optional_mla_storage_size_defaults_to_logical_block(spec_cls) -> None:
    spec = spec_cls(block_size=32, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    assert get_storage_block_size(spec) == 32
    uniform_spec = UniformTypeKVCacheSpecs(kv_cache_specs={"layer.0": spec, "layer.1": spec}, block_size=32)
    assert get_storage_block_size(uniform_spec) == 32
    if not vllm_version_is("0.27.1"):
        assert spec.storage_block_size is None
        explicit = replace(spec, storage_block_size=16)
        assert get_storage_block_size(explicit) == 16


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


@pytest.mark.skipif(vllm_version_is("0.27.1"), reason="vLLM #53896 introduced the packed-group hook")
def test_deepseek_v4_groups_patch_the_live_packed_group_hook() -> None:
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
    vllm_config = _make_vllm_config(enable_prefix_caching=True, dcp=1)
    vllm_config.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    vllm_config.speculative_config = None

    groups = vllm_kv_cache_utils.get_kv_cache_groups(
        vllm_config,
        {"c128": c128_spec, "swa": swa_spec, "c4": c4_spec},
    )

    assert vllm_kv_cache_utils._get_packed_kv_cache_groups is kv_cache_utils_patch._ascend_get_packed_kv_cache_groups
    assert [group.layer_names for group in groups[:2]] == [["c4"], ["c128"]]
    assert groups[2].layer_names == ["swa"]


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


@pytest.mark.skipif(not vllm_version_is("0.27.1"), reason="shared_by planner is only installed on v0.27.1")
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


def test_kimi_k3_gqa_mixed_grouping_falls_back_on_unrecognized_layer() -> None:
    specs = _make_kimi_k3_dspark_kv_cache_specs()
    specs["unrecognized.layer"] = next(iter(specs.values()))

    assert _get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is None


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


@pytest.mark.skipif(vllm_version_is("0.27.1"), reason="vLLM #51718 only changed the main planner")
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


@pytest.mark.skipif(vllm_version_is("0.27.1"), reason="vLLM #51718 introduced shared backing on main")
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


@pytest.mark.skipif(vllm_version_is("0.27.1"), reason="vLLM #51718 only re-plans ranks on main")
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
