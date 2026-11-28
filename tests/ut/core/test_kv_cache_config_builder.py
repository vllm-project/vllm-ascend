# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# ruff: noqa: E402
"""Dispatch contract of ``AscendKVCacheConfigBuilder``.

Locks the grouping precedence (GLM-Next, Qwen4-Exp MTP, Ascend shared-tuple
layouts, Kimi-K3 DSpark, then vLLM's generic grouping) and the TurboQuant pool
accounting, all of which now live in the builder rather than in a monkey patch.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip(
    "vllm.v1.core.kv_cache_config_builder",
    reason="the pluggable KV cache builder requires the v0.33.0 lane or newer",
)

import torch
from vllm.config import KVTransferConfig
from vllm.v1.core import kv_cache_config_builder as upstream_builder_module
from vllm.v1.core import kv_cache_planning
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.core import kv_cache_config_builder as builder_module
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
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
    specs, groups = generic_specs(), [Mock()]
    paired = Mock(return_value=groups)
    pool = Mock(return_value=14487552)
    monkeypatch.setattr(builder_module, "paired_sfa_cache_groups", paired)
    monkeypatch.setattr(builder_module, "paired_sfa_pool_bytes", pool)
    builder = AscendKVCacheConfigBuilder()
    assert builder.get_kv_cache_groups(runtime, specs) is groups
    assert builder._get_pool_bytes_per_block(runtime, groups) == 14487552
    paired.assert_called_once_with(runtime, specs)
    pool.assert_called_once_with(groups)


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
    runtime = make_runtime()
    runtime.cache_config = SimpleNamespace(prefix_cache_retention_interval=4096)
    config = builder_module._get_kv_cache_config_deepseek_v4_main(runtime, groups, num_blocks=3)
    assert config.num_blocks == 3 and config.prefix_cache_retention_interval == 4096
    assert config.kv_cache_tensors == tq_cache.cache_config(groups, num_blocks=3)[1]


@pytest.mark.parametrize("override", [None, 11])
def test_core_applies_capacity_and_converges_worker_layouts(monkeypatch, override):
    builder = AscendKVCacheConfigBuilder()
    config = make_runtime()
    config.cache_config = SimpleNamespace(
        num_gpu_blocks_override=override, prefix_cache_retention_interval=None, enable_extensible_kv_cache=False
    )
    config.model_config.original_max_model_len = config.model_config.max_model_len = 16
    config.parallel_config = SimpleNamespace(tensor_parallel_size=1, decode_context_parallel_size=1)
    spec = MLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, model_version="deepseek_v4"
    )
    uniform = UniformTypeKVCacheSpecs.from_specs({"target": spec})
    groups = [KVCacheGroupSpec(["target"], uniform)]
    specs = {"target": uniform}
    monkeypatch.setattr(upstream_builder_module, "get_kv_cache_config_builder", lambda _: builder)
    if hasattr(kv_cache_planning, "get_kv_cache_config_builder"):
        monkeypatch.setattr(kv_cache_planning, "get_kv_cache_config_builder", lambda _: builder)
    p = builder._get_pool_bytes_per_block(config, groups)
    results = kv_cache_planning.get_kv_cache_configs(config, [specs, specs], [100 * p, 80 * p], groups)
    expected_n = override if override is not None else 80
    expected = builder.get_kv_cache_config_from_groups(config, groups, expected_n)
    assert [result.num_blocks for result in results] == [expected_n, expected_n]
    assert all(result.kv_cache_tensors == expected.kv_cache_tensors for result in results)
    # A later materialization follows Core's current N without reapplying override.
    config.cache_config.num_gpu_blocks_override = 99
    replanned = builder.get_kv_cache_config_from_groups(config, groups, num_blocks=3)
    assert replanned.num_blocks == 3
    assert {tensor.size for tensor in replanned.kv_cache_tensors} == {3 * p}
