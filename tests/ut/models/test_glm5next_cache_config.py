# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GLM-Next cache grouping, physical layout, and capacity accounting."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import (
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendMLAAttentionSpec
from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_max_memory_usage,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers


@pytest.mark.parametrize("pool", [4, 16])
@pytest.mark.parametrize("speculative_blocks", [0, 5])
def test_platform_builder_accounts_for_both_glm_page_classes(pool, speculative_blocks):
    planning = pytest.importorskip("vllm.v1.core.kv_cache_planning")
    if not hasattr(planning, "_pool_bytes_per_block_from_builder"):
        pytest.skip("Requires the upstream builder pool-accounting hook")
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from vllm_ascend.platform import NPUPlatform

    config = make_config()
    config.attention_config = SimpleNamespace(hisparse_config=None)
    specs = {
        name: replace(spec, num_speculative_blocks=speculative_blocks) if isinstance(spec, MambaSpec) else spec
        for name, spec in make_specs(pool).items()
    }
    groups = get_glm5_next_kv_cache_groups(config, specs)
    physical = get_glm5_next_kv_cache_config(config, groups, get_glm5_next_pool_bytes_per_block(groups))
    assert len({tensor.size for tensor in physical.kv_cache_tensors}) == 2
    expected = sum(tensor.size for tensor in physical.kv_cache_tensors)
    builder = resolve_obj_by_qualname(NPUPlatform.get_kv_cache_config_builder_cls(config))()
    config.cache_config.num_gpu_blocks_override = 7
    assert planning._pool_bytes_per_block_from_builder(builder, config, groups) == expected
    assert config.cache_config.num_gpu_blocks_override == 7


def test_platform_builder_preserves_default_non_glm_accounting():
    pytest.importorskip("vllm.v1.core.kv_cache_planning")
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    groups = [KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=make_specs()["model.layers.3.attn"])]
    groups[0].kv_cache_spec = replace(groups[0].kv_cache_spec, model_version=None)
    assert AscendKVCacheConfigBuilder()._get_pool_bytes_per_block(make_config(), groups) is None


def _ratio_kwargs(ratio: int) -> dict[str, int]:
    return {"tokens_per_state": ratio}


def make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        max_in_flight_tokens=32,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            mamba_cache_mode="none",
            enable_prefix_caching=False,
        ),
    )


def make_specs(pool: int = 16):
    specs = {
        "model.layers.3.attn": MLAAttentionSpec(
            block_size=512,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            model_version="glm5_next",
        ),
        "model.layers.3.indexer.k_cache": MLAAttentionSpec(
            block_size=512,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            **_ratio_kwargs(pool),
        ),
        "model.layers.3.indexer.tail_cache": AscendIndexerKPoolTailSpec(
            block_size=pool,
            sliding_window=pool,
            compress_ratio=pool,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float32,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        ),
    }
    for layer_idx in range(3):
        specs[f"model.layers.{layer_idx}.linear_attn"] = MambaSpec(
            block_size=512,
            shapes=((3, 16), (1, 16, 16)),
            dtypes=(torch.bfloat16, torch.float32),
        )
    return specs


@pytest.fixture(autouse=True)
def register_cache_specs():
    # Match production: vLLM registers built-in specs before the Ascend hook.
    register_all_kvcache_specs(None)


@pytest.mark.parametrize("pool", [4, 16])
def test_groups_share_block_ids_and_pack_two_page_classes(pool):
    config = make_config()
    specs = make_specs(pool)
    groups = get_glm5_next_kv_cache_groups(config, dict(reversed(list(specs.items()))))
    layout = _get_glm5_next_cache_layout(groups)
    assert layout is not None
    assert groups[0].layer_names == [
        "model.layers.3.attn",
        "model.layers.3.indexer.k_cache",
    ]
    assert len(groups) == 5  # full, state, and three interleaved KDA groups
    assert layout.main_slot_count == layout.small_slot_count == 1

    bytes_per_block = layout.main_page_size + layout.small_page_size
    budget = 20 * bytes_per_block
    plan = get_glm5_next_kv_cache_config(config, groups, budget)
    assert plan.num_blocks == 20
    assert len(plan.kv_cache_tensors) == 2
    assert all(isinstance(tensor, KVCacheTensor) for tensor in plan.kv_cache_tensors)
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {
        20 * layout.main_page_size,
        20 * layout.small_page_size,
    }
    assert all(
        tensor.offset == 0 and tensor.layer_stride == 0 and tensor.block_stride == tensor.size // plan.num_blocks
        for tensor in plan.kv_cache_tensors
    )

    placements = {
        layer_name: tensor for tensor in plan.kv_cache_tensors for layer_name in get_kv_cache_tensor_layers(tensor)
    }
    main = placements[layout.mla_names[0]]
    indexer = placements[layout.indexer_names[0]]
    state = placements[layout.tail_names[0]]
    assert main.offset == 0
    assert indexer.offset == 0
    assert state is indexer
    assert set(get_kv_cache_tensor_layers(main)) == {
        layout.mla_names[0],
        *(group.layer_names[0] for group in layout.mamba_groups),
    }
    assert set(get_kv_cache_tensor_layers(indexer)) == {
        layout.indexer_names[0],
        layout.tail_names[0],
    }

    # Scheduler groups consume disjoint IDs from the shared global BlockPool.
    required_blocks = sum(
        (group.kv_cache_spec.max_memory_usage_bytes(config) + group.kv_cache_spec.page_size_bytes - 1)
        // group.kv_cache_spec.page_size_bytes
        for group in groups
    )
    assert get_glm5_next_pool_bytes_per_block(groups) == bytes_per_block
    assert get_glm5_next_max_memory_usage(config, groups) == required_blocks * bytes_per_block


def test_standalone_mtp_layout_has_no_mamba_groups():
    specs = {name: spec for name, spec in make_specs().items() if not isinstance(spec, MambaSpec)}
    groups = get_glm5_next_kv_cache_groups(make_config(), specs)
    layout = _get_glm5_next_cache_layout(groups)
    assert len(groups) == 2
    assert layout is not None
    assert layout.mamba_groups == ()


def test_pipeline_projection_supports_a_mamba_only_worker():
    config = make_config()
    groups = get_glm5_next_kv_cache_groups(config, make_specs())
    local_mamba_name = groups[2].layer_names[0]
    projected_groups = [
        KVCacheGroupSpec(
            [local_mamba_name] if group is groups[2] else [],
            group.kv_cache_spec,
        )
        for group in groups
    ]

    layout = _get_glm5_next_cache_layout(projected_groups)
    assert layout is not None
    assert layout.mla_names == layout.indexer_names == layout.tail_names == ()
    assert layout.main_slot_count == 1
    assert layout.small_slot_count == 0

    budget = 10 * layout.main_page_size
    plan = get_glm5_next_kv_cache_config(config, projected_groups, available_memory=budget)
    assert plan.num_blocks == 10
    assert len(plan.kv_cache_tensors) == 1
    assert get_kv_cache_tensor_layers(plan.kv_cache_tensors[0]) == [local_mamba_name]


def test_missing_paired_cache_is_rejected():
    specs = make_specs()
    del specs["model.layers.3.indexer.tail_cache"]
    with pytest.raises(ValueError, match="requires"):
        get_glm5_next_kv_cache_groups(make_config(), specs)


def test_misaligned_logical_block_is_rejected():
    specs = make_specs(pool=16)
    object.__setattr__(
        specs["model.layers.3.indexer.k_cache"],
        "tokens_per_state",
        15,
    )
    with pytest.raises(ValueError, match="divisible"):
        get_glm5_next_kv_cache_groups(make_config(), specs)


@pytest.mark.parametrize("use_sparse", [False, True])
@pytest.mark.parametrize("packed_main", [False, True])
def test_worker_preserves_glm5_next_physical_pool_budget(monkeypatch, use_sparse, packed_main):
    from vllm_ascend.worker.worker import NPUWorker

    config = make_config()
    config.model_config.max_model_len = 1048576
    config.model_config.hf_text_config = SimpleNamespace(model_type="glm5_next", index_topk=2048, index_kpool=16)
    config.cache_config.get_resolved_kv_cache_layout = lambda: SimpleNamespace(
        is_layer_compact=True, is_block_compact=True
    )
    specs = make_specs()
    if packed_main:
        specs["model.layers.3.attn"] = AscendMLAAttentionSpec(
            block_size=512,
            num_kv_heads=1,
            head_size=528,
            dtype=torch.int8,
            cache_sparse_sfa_c8=True,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        )
    groups = get_glm5_next_kv_cache_groups(config, specs)
    # Supply exactly enough physical memory for the configured context. The
    # generic discount incorrectly makes this valid shared pool fail admission.
    required = get_glm5_next_max_memory_usage(config, groups)
    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = config
    worker.model_config = config.model_config
    worker.get_kv_cache_spec = lambda: specs
    worker.model_runner = SimpleNamespace(
        use_sparse=use_sparse,
        use_compress=False,
        supports_standardized_shared_kv_backing=True,
    )
    monkeypatch.setattr("vllm_ascend.worker.worker.get_kv_cache_groups", lambda *_args: groups)
    budget = worker._scale_kv_cache_memory_for_multi_group(required)

    assert budget == required
    plan = get_glm5_next_kv_cache_config(config, groups, budget)
    assert plan.num_blocks * get_glm5_next_pool_bytes_per_block(groups) == required
    assert sum(tensor.size for tensor in plan.kv_cache_tensors) == required
