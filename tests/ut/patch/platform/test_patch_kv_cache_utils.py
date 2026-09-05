# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import (
    SlidingWindowManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_get_kv_cache_config_from_groups,
    _ascend_get_kv_cache_groups,
    _ascend_max_memory_usage_bytes_from_groups,
    _ascend_pool_bytes_per_block,
    _dspark_native_layout_enabled,
    _is_mla_regular_swa_groups,
    _partition_mla_regular_swa_specs,
    _use_dspark_native_layout,
)


def _config(*, use_dspark=True, disable_hybrid=False):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=disable_hybrid,
        ),
        speculative_config=SimpleNamespace(use_dspark=lambda: use_dspark),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        model_config=SimpleNamespace(max_model_len=257000),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        # Async scheduling doubles --max-num-batched-tokens=80.
        max_in_flight_tokens=160,
    )


def _glm_dspark_specs():
    return {
        # Put the indexer first deliberately. The grouping wrapper must keep
        # the target-spec order and therefore preserve baseline manager choice.
        "model.layers.0.self_attn.indexer": AscendSFAIndexerCacheSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            scale_dim=0,
            cache_sparse_li_c8=False,
        ),
        "model.layers.0.self_attn": AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=656,
            dtype=torch.int8,
            cache_sparse_sfa_c8=True,
        ),
        "model.dspark.layers.0.self_attn": SlidingWindowSpec(
            block_size=128,
            num_kv_heads=8,
            head_size=192,
            dtype=torch.bfloat16,
            sliding_window=1024,
        ),
    }


def _register_test_specs():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()


def test_glm_dspark_keeps_native_pages_and_sliding_window_manager():
    _register_test_specs()
    config = _config()
    specs = _glm_dspark_specs()

    groups = _ascend_get_kv_cache_groups(config, specs)

    assert _is_mla_regular_swa_groups(groups)
    assert len(groups) == 2
    assert all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in groups
    )
    assert groups[0].layer_names == [
        "model.layers.0.self_attn.indexer",
        "model.layers.0.self_attn",
    ]
    assert groups[1].layer_names == ["model.dspark.layers.0.self_attn"]
    draft_spec = specs["model.dspark.layers.0.self_attn"]
    assert KVCacheSpecRegistry.get_manager_class(draft_spec) is SlidingWindowManager
    assert draft_spec.page_size_bytes == 786432
    assert specs["model.layers.0.self_attn"].page_size_bytes == 83968
    assert (
        specs["model.layers.0.self_attn.indexer"].page_size_bytes == 32768
    )
    assert all(
        spec.page_size_padded is None
        for spec in specs.values()
        if hasattr(spec, "page_size_padded")
    )


def test_glm_dspark_allocates_one_native_layout_tensor_per_layer():
    _register_test_specs()
    config = _config()
    specs = _glm_dspark_specs()
    groups = _ascend_get_kv_cache_groups(config, specs)
    bytes_per_block = sum(spec.page_size_bytes for spec in specs.values())

    assert _ascend_pool_bytes_per_block(config, groups) == bytes_per_block
    expected_blocks = 17
    cache_config = _ascend_get_kv_cache_config_from_groups(
        config,
        groups,
        bytes_per_block * expected_blocks + bytes_per_block // 2,
    )

    assert cache_config.num_blocks == expected_blocks
    assert len(cache_config.kv_cache_tensors) == len(specs)
    tensor_by_layer = {
        tensor.shared_by[0]: tensor
        for tensor in cache_config.kv_cache_tensors
    }
    assert all(
        len(tensor.shared_by) == 1
        for tensor in cache_config.kv_cache_tensors
    )
    for layer_name, spec in specs.items():
        tensor = tensor_by_layer[layer_name]
        assert tensor.size == spec.page_size_bytes * expected_blocks
        assert tensor.offset == 0
        assert tensor.block_stride == 0

    scheduler_config = generate_scheduler_kv_cache_config([cache_config])
    assert isinstance(
        scheduler_config.kv_cache_groups[0].kv_cache_spec,
        AscendSFAIndexerCacheSpec,
    )
    assert isinstance(
        scheduler_config.kv_cache_groups[1].kv_cache_spec,
        SlidingWindowSpec,
    )

    expected_pages = sum(
        group.kv_cache_spec.max_memory_usage_pages(config)
        for group in groups
    )
    assert _ascend_max_memory_usage_bytes_from_groups(
        config, groups
    ) == bytes_per_block * expected_pages


def test_257k_request_fits_candidate_block_capacity():
    _register_test_specs()
    config = _config()
    specs = {}
    for index in range(78):
        if index < 21:
            specs[f"model.layers.{index}.self_attn.indexer"] = (
                AscendSFAIndexerCacheSpec(
                    block_size=128,
                    num_kv_heads=1,
                    head_size=128,
                    dtype=torch.bfloat16,
                    scale_dim=0,
                    cache_sparse_li_c8=False,
                )
            )
        specs[f"model.layers.{index}.self_attn"] = AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=656,
            dtype=torch.int8,
            cache_sparse_sfa_c8=True,
        )
    for index in range(5):
        specs[f"model.dspark.layers.{index}.self_attn"] = SlidingWindowSpec(
            block_size=128,
            num_kv_heads=8,
            head_size=192,
            dtype=torch.bfloat16,
            sliding_window=1024,
        )

    groups = _ascend_get_kv_cache_groups(config, specs)
    bytes_per_block = _ascend_pool_bytes_per_block(config, groups)
    available_blocks = 2752
    cache_config = _ascend_get_kv_cache_config_from_groups(
        config, groups, bytes_per_block * available_blocks
    )
    required_blocks = sum(
        group.kv_cache_spec.max_memory_usage_pages(config)
        for group in groups
    )

    assert cache_config.num_blocks == available_blocks
    assert bytes_per_block == 11_169_792
    assert len(cache_config.kv_cache_tensors) == 104
    assert sum(
        tensor.size for tensor in cache_config.kv_cache_tensors
    ) == bytes_per_block * available_blocks
    assert required_blocks == 2008 + 11
    assert required_blocks < cache_config.num_blocks


def test_compressed_target_does_not_use_dspark_native_layout_fast_path():
    specs = _glm_dspark_specs()
    specs["model.layers.0.self_attn"] = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=656,
        dtype=torch.int8,
        cache_sparse_sfa_c8=True,
        compress_ratio=2,
    )

    assert _partition_mla_regular_swa_specs(specs) is None


def test_native_layout_preserves_main_first_target_order():
    specs = _glm_dspark_specs()
    main_name = "model.layers.0.self_attn"
    indexer_name = "model.layers.0.self_attn.indexer"
    draft_name = "model.dspark.layers.0.self_attn"
    main_first_specs = {
        main_name: specs[main_name],
        indexer_name: specs[indexer_name],
        draft_name: specs[draft_name],
    }

    groups = _ascend_get_kv_cache_groups(_config(), main_first_specs)

    assert groups[0].layer_names == [main_name, indexer_name]


def test_native_layout_gate_rejects_non_dspark_and_disabled_hybrid():
    groups = _ascend_get_kv_cache_groups(_config(), _glm_dspark_specs())

    assert _dspark_native_layout_enabled(_config())
    assert _use_dspark_native_layout(_config(), groups)
    assert not _dspark_native_layout_enabled(_config(use_dspark=False))
    assert not _use_dspark_native_layout(_config(use_dspark=False), groups)
    assert not _dspark_native_layout_enabled(_config(disable_hybrid=True))
    assert not _use_dspark_native_layout(
        _config(disable_hybrid=True), groups
    )
