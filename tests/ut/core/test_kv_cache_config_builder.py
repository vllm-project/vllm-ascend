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
from vllm.v1.core import kv_cache_config_builder as upstream_builder_module
from vllm.v1.core import kv_cache_planning
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core import kv_cache_config_builder as builder_module
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendSlidingWindowMLASpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.models.glm5next.cache_config import get_glm5_next_kv_cache_groups
from vllm_ascend.models.qwen4_exp.cache_config import get_qwen4_exp_kv_cache_config
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


def qwen_groups(raw_width=32):
    main = FullAttentionSpec(block_size=256, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    return [
        KVCacheGroupSpec(["model.layers.3.self_attn.attn"], main),
        KVCacheGroupSpec(
            ["model.layers.3.self_attn.indexer.raw_key_cache"],
            CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=raw_width, head_size_v=0, dtype=torch.bfloat16),
        ),
        KVCacheGroupSpec(
            ["model.layers.3.self_attn.indexer.compressed_key_cache"],
            MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=32, dtype=torch.bfloat16, tokens_per_state=4),
        ),
        KVCacheGroupSpec(
            ["model.layers.0.linear_attn"],
            MambaSpec(block_size=256, shapes=((64, 4), (128, 128)), dtypes=(torch.bfloat16, torch.bfloat16)),
        ),
        KVCacheGroupSpec(
            ["model.ple"],
            MambaSpec(block_size=256, shapes=((128, 5),), dtypes=(torch.bfloat16,), tp_replicated=True),
        ),
    ]


def materialization_config(override=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3")),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=override,
            prefix_cache_retention_interval=None,
            mamba_cache_mode="none",
            enable_extensible_kv_cache=False,
        ),
        kv_transfer_config=None,
    )


@pytest.mark.parametrize("override", [None, 7])
def test_qwen_materialization_retains_budget_planner(monkeypatch, override):
    groups = qwen_groups()
    builder = AscendKVCacheConfigBuilder()
    config = materialization_config(override)
    num_blocks = 3
    budget = num_blocks * builder._get_pool_bytes_per_block(config, groups)
    planner = Mock(wraps=get_qwen4_exp_kv_cache_config)
    monkeypatch.setattr(builder_module, "get_qwen4_exp_kv_cache_config", planner)

    result = builder.get_kv_cache_config_from_groups(config, groups, num_blocks)

    planner.assert_called_once_with(config, groups, budget)
    # Qwen4 deliberately keeps its own capacity/override handling.
    assert result.num_blocks == (override if override is not None else num_blocks)
    expected = get_qwen4_exp_kv_cache_config(config, groups, budget)
    assert result.kv_cache_tensors == expected.kv_cache_tensors
    assert result.kv_transfer_config is None


@pytest.mark.parametrize("version", ["deepseek_v4", "deepseek_v41"])
def test_deepseek_materialization_does_not_reapply_override(version):
    spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.bfloat16,
        sliding_window=32,
        cache_dtype_str="bfloat16",
        model_version=version,
    )
    name = "model.layers.0.self_attn.swa_cache"
    uniform = UniformTypeKVCacheSpecs.from_specs({name: spec})
    groups = [KVCacheGroupSpec([name], uniform)]
    result = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(
        materialization_config(override=99), groups, 3
    )
    assert result.num_blocks == 3
    assert result.kv_cache_tensors[0].size == 3 * spec.page_size_bytes


@pytest.mark.parametrize("layout_kind", ["qwen", "dsv4"])
@pytest.mark.parametrize("override", [None, 11])
def test_core_applies_capacity_and_converges_worker_layouts(monkeypatch, override, layout_kind):
    builder = AscendKVCacheConfigBuilder()
    config = materialization_config(override)
    config.model_config = SimpleNamespace(
        original_max_model_len=16, max_model_len=16, hf_config=SimpleNamespace(model_type="qwen3")
    )
    config.parallel_config = SimpleNamespace(tensor_parallel_size=1, decode_context_parallel_size=1)
    config.attention_config = SimpleNamespace(hisparse_config=None)
    if layout_kind == "qwen":
        groups = qwen_groups(raw_width=1)
    else:
        spec = MLAAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, model_version="deepseek_v4"
        )
        uniform = UniformTypeKVCacheSpecs.from_specs({"target": spec})
        groups = [KVCacheGroupSpec(["target"], uniform)]
    specs = {name: group.kv_cache_spec for group in groups for name in group.layer_names}
    monkeypatch.setattr(upstream_builder_module, "get_kv_cache_config_builder", lambda _: builder)
    # Newer upstream imports the resolver at module scope; older supported code
    # resolves it inside get_kv_cache_configs.
    if hasattr(kv_cache_planning, "get_kv_cache_config_builder"):
        monkeypatch.setattr(kv_cache_planning, "get_kv_cache_config_builder", lambda _: builder)
    p = builder._get_pool_bytes_per_block(config, groups)
    # Enough blocks for one request plus the reserved null block on both ranks.
    results = kv_cache_planning.get_kv_cache_configs(config, [specs, specs], [100 * p, 80 * p], groups)
    expected_n = override if override is not None else 80
    expected = builder.get_kv_cache_config_from_groups(config, groups, expected_n)
    assert [r.num_blocks for r in results] == [expected_n, expected_n]
    assert all(r.kv_cache_tensors == expected.kv_cache_tensors for r in results)


def test_glm_materialization_preserves_core_count_and_both_backings():
    config = materialization_config(override=99)
    config.model_config = SimpleNamespace(max_model_len=2048, hf_config=SimpleNamespace(model_type="glm5_next"))
    config.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    config.parallel_config = SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1)
    config.max_in_flight_tokens = 32
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
            tokens_per_state=16,
            dtype=torch.bfloat16,
            model_version="glm5_next",
        ),
        "model.layers.3.indexer.tail_cache": AscendIndexerKPoolTailSpec(
            block_size=16,
            sliding_window=16,
            compress_ratio=16,
            ring_capacity=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float32,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
        ),
    }
    groups = get_glm5_next_kv_cache_groups(config, specs)
    builder = AscendKVCacheConfigBuilder()
    result = builder.get_kv_cache_config_from_groups(config, groups, 3)
    assert result.num_blocks == 3
    assert len(result.kv_cache_tensors) == 2
    assert sum(t.size for t in result.kv_cache_tensors) == 3 * builder._get_pool_bytes_per_block(config, groups)


def test_turboquant_materialization_uses_core_count():
    spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=SLOT_BYTES,
        dtype=torch.uint8,
        sliding_window=128,
        cache_dtype_str=TURBOQUANT_CACHE_DTYPE,
        model_version="deepseek_v4",
    )
    uniform = UniformTypeKVCacheSpecs.from_specs({"model.layers.0.attn": spec})
    groups = [KVCacheGroupSpec(list(uniform.kv_cache_specs), uniform)]
    config = materialization_config(override=99)
    result = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(config, groups, 3)
    assert result.num_blocks == 3
    assert {t.size for t in result.kv_cache_tensors} == {3 * tq_cache.pool_bytes_per_block(groups)}
