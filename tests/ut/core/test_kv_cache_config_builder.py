# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# ruff: noqa: E402
"""Dispatch contract of ``AscendKVCacheConfigBuilder``.

Covers hybrid HiSparse dispatch and device layout, ordinary grouping precedence,
and TurboQuant pool accounting.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip(
    "vllm.v1.core.kv_cache_config_builder",
    reason="the pluggable KV cache builder requires the v0.33.0 lane or newer",
)

import torch
from vllm.config import KVTransferConfig
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _get_kv_cache_bytes_per_block,
    _max_memory_usage_bytes_from_groups,
    _pool_bytes_per_block_from_builder,
)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.core import kv_cache_config_builder as builder_module
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendSFAIndexerCacheSpec,
    AscendSlidingWindowMLASpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import cache_planner
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LAYERWISE_REUSE_BACKING_BYTES,
    get_layerwise_reuse_backing_bytes,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import (
    SLOT_BYTES,
    TURBOQUANT_CACHE_DTYPE,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache


@pytest.fixture(autouse=True)
def registered_specs():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()


def make_runtime():
    return SimpleNamespace(
        model_config=SimpleNamespace(supports_kv_cache_grouping=False, hf_config=SimpleNamespace(model_type="qwen3")),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
    )


def make_hisparse_runtime(model_type="mixed_sparse_model", layout=None):
    runtime = make_runtime()
    runtime.model_config.hf_config.model_type = model_type
    runtime.model_config.hf_config.index_topk = 2048
    runtime.model_config.hf_text_config = SimpleNamespace(index_topk=2048)
    runtime.model_config.max_model_len = 2048
    runtime.attention_config.hisparse_config = SimpleNamespace(device_buffer_size=None, eager_host_mirror=True)
    runtime.speculative_config = None
    runtime.kv_transfer_config = None
    runtime.max_in_flight_tokens = 1024
    runtime.parallel_config = SimpleNamespace(decode_context_parallel_size=1)
    runtime.cache_config = SimpleNamespace(
        kv_cache_layout=layout,
        mamba_cache_mode="align",
        get_resolved_kv_cache_layout=lambda: layout,
    )
    return runtime


def sparse_attention_specs():
    return {
        "main": MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=656,
            dtype=torch.float8_e4m3fn,
            cache_role=SparseCacheRole.SPARSE,
            state_content_bytes=656,
            block_stride_alignment=128 * 656,
            is_index_group_leader=True,
        ),
        "indexer": AscendSFAIndexerCacheSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float8_e4m3fn,
            cache_role=SparseCacheRole.INDEXER,
            cache_sparse_li_c8=True,
            scale_dim=1,
            scale_dtype=torch.float32,
        ),
    }


def generic_specs():
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
    return {"layer.a": spec, "layer.b": spec}


def test_kimi_k31_grouping_precedes_generic_hybrid_promotion(monkeypatch):
    runtime = make_runtime()
    runtime.model_config.hf_config.model_type = "kimi_k31"
    runtime.scheduler_config.disable_hybrid_kv_cache_manager = True
    specs = generic_specs()
    paired = Mock(side_effect=ValueError("Paired SFA cache requires the hybrid KV cache manager."))
    promote = Mock()
    monkeypatch.setattr(builder_module, "paired_sfa_cache_groups", paired)
    monkeypatch.setattr(builder_module, "unify_hybrid_kv_cache_specs", promote)
    with pytest.raises(ValueError, match="requires the hybrid KV cache manager"):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
    paired.assert_called_once_with(runtime, specs)
    promote.assert_not_called()


def test_kimi_k31_builder_uses_paired_grouping_and_pool_accounting(monkeypatch):
    runtime = make_runtime()
    runtime.model_config.hf_config.model_type = "kimi_k31"
    specs = generic_specs()
    groups = [KVCacheGroupSpec(list(specs), specs["layer.a"])]
    paired = Mock(return_value=groups)
    bytes_per_block = 14487552
    pool = Mock(return_value=bytes_per_block)
    num_blocks = 2
    expected = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)
    place = Mock(return_value=expected)
    monkeypatch.setattr(builder_module, "paired_sfa_cache_groups", paired)
    monkeypatch.setattr(builder_module, "paired_sfa_pool_bytes", pool)
    monkeypatch.setattr(builder_module, "paired_sfa_cache_config", place)
    builder = AscendKVCacheConfigBuilder()
    assert builder.get_kv_cache_groups(runtime, specs) is groups
    assert builder._get_pool_bytes_per_block(runtime, groups) == bytes_per_block
    paired.assert_called_once_with(runtime, specs)
    pool.assert_called_once_with(groups)
    assert builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks) is expected
    place.assert_called_once_with(runtime, groups, num_blocks * bytes_per_block)


@pytest.mark.parametrize("model_type", ["kimi_k31", "mixed_sparse_model"])
def test_hybrid_hisparse_uses_native_indexer_page_and_separate_recurrent_groups(model_type):
    runtime = make_hisparse_runtime(model_type)
    state = MambaSpec(
        block_size=128,
        shapes=((32, 64), (16, 64, 64)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    groups = AscendKVCacheConfigBuilder().get_kv_cache_groups(
        runtime, {**sparse_attention_specs(), "state.0": state, "state.1": state}
    )

    host, pooled_indexer, resident, hot, *states = groups
    assert host.host_resident and host.role is KVCacheGroupRole.HISPARSE_SOURCE
    assert host.layer_names == ["main"]
    assert pooled_indexer.role is KVCacheGroupRole.HISPARSE_INDEXER and not pooled_indexer.host_resident
    assert pooled_indexer.kv_cache_spec.block_size == 128
    indexer = pooled_indexer.kv_cache_spec.kv_cache_specs["indexer"]
    assert (indexer.block_table_block_size or indexer.block_size) == 128
    assert isinstance(resident.kv_cache_spec, HiSparseResidentSpec)
    assert isinstance(hot.kv_cache_spec, HiSparseHotSpec)
    assert hot.kv_cache_spec.get_block_stride_alignment() == hot.kv_cache_spec.page_size_bytes == 128 * 656
    assert [group.layer_names for group in states] == [["state.0"], ["state.1"]]
    assert all(group.kv_cache_spec == state for group in states)
    assert all(not group.host_resident for group in states)


@pytest.mark.parametrize("method", ["ngram", "ngram_gpu"])
def test_hybrid_hisparse_reserves_verify_rows_with_ngram(method):
    runtime = make_hisparse_runtime("kimi_k31")
    specs = {
        **sparse_attention_specs(),
        "state": MambaSpec(block_size=128, shapes=((32,),), dtypes=(torch.float32,)),
    }
    builder = AscendKVCacheConfigBuilder()
    target_groups = builder.get_kv_cache_groups(runtime, specs)

    runtime.speculative_config = SimpleNamespace(
        method=method,
        num_speculative_tokens=1,
        max_num_new_slots_for_drafting=0,
        parallel_drafting=False,
        draft_model_config=runtime.model_config,
    )
    groups = builder.get_kv_cache_groups(runtime, specs)

    host, indexer, resident, hot, *states = groups
    base_host, base_indexer, base_resident, base_hot, *base_states = target_groups
    assert (host, indexer, resident, states) == (base_host, base_indexer, base_resident, base_states)
    assert isinstance(base_hot.kv_cache_spec, HiSparseHotSpec)
    assert base_hot.kv_cache_spec.blocks_per_request == 32
    assert hot == replace(base_hot, kv_cache_spec=replace(base_hot.kv_cache_spec, blocks_per_request=48))


@pytest.mark.parametrize("method", [None, "ngram", "ngram_gpu"])
def test_hybrid_hisparse_rejects_draft_caches(method):
    runtime = make_hisparse_runtime("kimi_k31")
    specs = sparse_attention_specs()
    specs["state"] = MambaSpec(block_size=128, shapes=((32,),), dtypes=(torch.float32,))
    specs["draft"] = replace(specs["main"], is_draft=True)
    if method is not None:
        runtime.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=1)

    with pytest.raises(ValueError, match="does not support draft caches"):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)


@pytest.mark.parametrize("method", ["dspark", "mtp", "eagle"])
def test_hybrid_hisparse_rejects_other_speculative_methods(method):
    runtime = make_hisparse_runtime("kimi_k31")
    runtime.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=1)
    specs = {
        **sparse_attention_specs(),
        "state": MambaSpec(block_size=128, shapes=((32,),), dtypes=(torch.float32,)),
    }

    with pytest.raises(ValueError, match=rf"only supports ngram and ngram_gpu speculative decoding, got '{method}'"):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)


def test_hybrid_hisparse_rejects_unsupported_attention():
    runtime = make_hisparse_runtime("kimi_k31")
    state = MambaSpec(block_size=128, shapes=((32,),), dtypes=(torch.float32,))
    specs = {**sparse_attention_specs(), "state": state, **generic_specs()}

    with pytest.raises(ValueError, match=r"Unsupported cache spec in Ascend hybrid SFA HiSparse: layer\.a"):
        AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)


def test_hybrid_hisparse_groups_recurrent_layers_by_resolved_device_stride():
    runtime = make_hisparse_runtime()
    sparse = sparse_attention_specs()
    state = MambaSpec(
        block_size=128,
        shapes=((25_000,),),
        dtypes=(torch.bfloat16,),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    specs = {f"{name}.{i}": spec for i in range(6) for name, spec in {**sparse, "state": state}.items()}
    builder = AscendKVCacheConfigBuilder()
    baseline = builder.get_kv_cache_groups(runtime, specs)

    layout = KVCacheLayout.BLHNC
    runtime = make_hisparse_runtime(layout=layout)
    selected = builder.get_kv_cache_groups(runtime, specs)

    # The six Indexer pages occupy 101,376 bytes. Aligning to a full MLA
    # page already costs 167,936 bytes, enough for a third state per block.
    baseline_device = [group for group in baseline if not group.host_resident]
    selected_device = [group for group in selected if not group.host_resident]
    assert _get_kv_cache_bytes_per_block(baseline_device, layout) == _get_kv_cache_bytes_per_block(
        selected_device, layout
    )
    assert _max_memory_usage_bytes_from_groups(builder, runtime, selected) < _max_memory_usage_bytes_from_groups(
        builder, runtime, baseline
    )
    old_states = [group for group in baseline if isinstance(group.kv_cache_spec, MambaSpec)]
    new_states = [group for group in selected if isinstance(group.kv_cache_spec, MambaSpec)]
    assert [len(group.layer_names) for group in old_states] == [2, 2, 2]
    assert [len(group.layer_names) for group in new_states] == [3, 3]
    assert {name for group in new_states for name in group.layer_names} == {f"state.{i}" for i in range(6)}
    (indexer,) = [group for group in selected if group.role is KVCacheGroupRole.HISPARSE_INDEXER]
    assert indexer.kv_cache_spec.block_size == 128


@pytest.mark.parametrize("model_type", ["kimi_k31", "mixed_sparse_model"])
def test_hisparse_without_recurrent_state_preserves_upstream_grouping(model_type):
    runtime = make_hisparse_runtime(model_type)
    specs = sparse_attention_specs()
    expected = DefaultKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
    assert AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs) == expected


@pytest.mark.parametrize(
    "model_type, include_recurrent",
    [("mixed_sparse_model", True), ("kimi_k31", False), ("kimi_k31", True)],
)
def test_hisparse_placement_uses_the_core_device_block_stride(monkeypatch, model_type, include_recurrent):
    from vllm.v1.hisparse import layout as hisparse_layout

    layout = KVCacheLayout.BLHNC
    runtime = make_hisparse_runtime(model_type, layout=layout)
    runtime.cache_config.num_gpu_blocks_override = None
    runtime.cache_config.prefix_cache_retention_interval = None
    runtime.kv_transfer_config = KVTransferConfig(
        kv_connector="HiSparseConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"host_pool_gib": 1},
    )
    monkeypatch.setattr(hisparse_layout, "use_shared_hisparse_host_pool", lambda _: False)
    specs = sparse_attention_specs()
    if include_recurrent:
        specs["state"] = MambaSpec(block_size=128, shapes=((65_536,),), dtypes=(torch.float32,))
    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(runtime, specs)

    device_groups = [group for group in groups if not group.host_resident]
    stride = _get_kv_cache_bytes_per_block(device_groups, layout)
    assert _pool_bytes_per_block_from_builder(builder, runtime, device_groups) == stride
    assert _max_memory_usage_bytes_from_groups(builder, runtime, groups) > 0

    num_blocks = 64
    config = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks)
    assert config.num_blocks == num_blocks
    assert config.hisparse_host_num_blocks > 0
    assert {layer for tensor in config.kv_cache_tensors if tensor.host_resident for layer in tensor.layers} == {"main"}
    device_layers = {layer for tensor in config.kv_cache_tensors if not tensor.host_resident for layer in tensor.layers}
    assert "indexer" in device_layers
    if include_recurrent:
        assert "state" in device_layers
        (state_group,) = [group for group in device_groups if isinstance(group.kv_cache_spec, MambaSpec)]
        assert state_group.layer_names == ["state"]
    assert all(
        tensor.block_stride == stride and tensor.size == stride * num_blocks
        for tensor in config.kv_cache_tensors
        if not tensor.host_resident
    )


@pytest.mark.parametrize("draft_sharded", [True, False])
def test_kimi_grouping_requires_matching_dcp_layout(draft_sharded):
    target = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
    draft = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, dcp_sharded=draft_sharded
    )
    state = MambaSpec(block_size=16, shapes=((32, 16),), dtypes=(torch.bfloat16,))
    assert target.page_size_bytes == draft.page_size_bytes == state.page_size_bytes
    specs = {
        "language_model.model.layers.0.attn": target,
        "language_model.model.layers.1.state": state,
        "model.layers.0.attn": draft,
    }
    groups = builder_module._get_kimi_k3_dspark_mixed_kv_cache_groups(specs)
    if draft_sharded:
        assert groups is not None
        assert set(groups[0].layer_names) == {"language_model.model.layers.0.attn", "model.layers.0.attn"}
    else:
        assert groups is None


def test_generic_uniform_spec_falls_back_to_default_grouping():
    runtime = make_runtime()
    specs = generic_specs()
    result = AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
    expected = DefaultKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
    assert result == expected
    assert len(result) == 1
    assert set(result[0].layer_names) == {"layer.a", "layer.b"}


def test_layerwise_offload_budget_uses_one_block_backing(monkeypatch):
    runtime = make_runtime()
    runtime.cache_config = SimpleNamespace(
        num_gpu_blocks_override=None,
        prefix_cache_retention_interval=0,
        get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
    )
    runtime.model_config.get_total_num_hidden_layers = lambda: 4
    runtime.kv_transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={"backend": "memcache", "use_layerwise": True, "layerwise_num_shared_buffers": 1},
    )
    builder = AscendKVCacheConfigBuilder()
    spec = generic_specs()["layer.a"]
    groups = [KVCacheGroupSpec([f"model.layers.{layer}.attn" for layer in range(4)], spec)]
    rules = {FullAttentionSpec: True}
    monkeypatch.setattr(builder_module, "get_spec_reuse_rules", lambda _: rules)
    monkeypatch.setattr(cache_planner, "get_spec_reuse_rules", lambda _: rules)

    backing_bytes_per_block = builder._get_pool_bytes_per_block(runtime, groups)
    assert backing_bytes_per_block is not None and backing_bytes_per_block == 2 * spec.page_size_bytes
    native_config = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=3)
    assert native_config.num_blocks == 3
    assert get_layerwise_reuse_backing_bytes(native_config) == 3 * backing_bytes_per_block
    assert LAYERWISE_REUSE_BACKING_BYTES not in runtime.kv_transfer_config.kv_connector_extra_config


def test_glm_next_layout_takes_precedence_over_ascend_grouping(monkeypatch):
    sentinel = [KVCacheGroupSpec(["glm"], generic_specs()["layer.a"])]
    glm_mock = Mock(return_value=sentinel)
    ascend_mock = Mock()
    monkeypatch.setattr(builder_module, "is_glm5_next_cache_spec", lambda spec: True)
    monkeypatch.setattr(builder_module, "get_glm5_next_kv_cache_groups", glm_mock)
    monkeypatch.setattr(builder_module, "group_and_unify_kv_cache_specs", ascend_mock)
    assert AscendKVCacheConfigBuilder().get_kv_cache_groups(make_runtime(), generic_specs()) is sentinel
    ascend_mock.assert_not_called()


def test_ascend_layout_takes_precedence_over_kimi_dspark(monkeypatch):
    sentinel = [KVCacheGroupSpec(["tuple"], generic_specs()["layer.a"])]
    ascend_mock = Mock(return_value=sentinel)
    kimi_mock = Mock(return_value=None)
    monkeypatch.setattr(builder_module, "is_glm5_next_cache_spec", lambda spec: False)
    monkeypatch.setattr(builder_module, "is_deepseek_v41_cache", lambda spec: True)
    monkeypatch.setattr(builder_module, "group_cache_specs", Mock(return_value=[]))
    monkeypatch.setattr(builder_module, "make_cache_groups", ascend_mock)
    monkeypatch.setattr(builder_module, "_get_kimi_k3_dspark_mixed_kv_cache_groups", kimi_mock)
    assert AscendKVCacheConfigBuilder().get_kv_cache_groups(make_runtime(), generic_specs()) is sentinel
    kimi_mock.assert_not_called()


def test_turboquant_groups_use_packed_pool_bytes():
    tq_spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=SLOT_BYTES,
        dtype=torch.uint8,
        sliding_window=128,
        cache_dtype_str=TURBOQUANT_CACHE_DTYPE,
        model_version="deepseek_v4",
    )
    uniform = UniformTypeKVCacheSpecs.from_specs({"l0": tq_spec, "l1": tq_spec})
    groups = [KVCacheGroupSpec(layer_names=["l0", "l1"], kv_cache_spec=uniform)]
    builder = AscendKVCacheConfigBuilder()
    assert tq_cache.uses_turboquant_groups(groups)
    pool_bytes = builder._get_pool_bytes_per_block(make_runtime(), groups)
    assert pool_bytes == tq_cache.pool_bytes_per_block(groups)
    assert pool_bytes >= sum(spec.page_size_bytes for spec in uniform.kv_cache_specs.values())
