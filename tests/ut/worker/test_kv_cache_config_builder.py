# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Tests for the Ascend KV cache config builder.

Covers the DeepSeekV4 grouping + non-packed shared-tensor layout in
``vllm_ascend.worker.kv_cache_config_builder`` (wired via
``NPUPlatform.get_kv_cache_config_builder_cls``, vLLM PR #53558).
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import vllm.v1.core.kv_cache_planning as kv_cache_planning
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.worker.kv_cache_config_builder import (
    AscendKVCacheConfigBuilder,
    _ascend_get_kv_cache_config_deepseek_v4,
    _ascend_get_kv_cache_groups_uniform_groups,
    _ascend_group_and_unify_kv_cache_specs,
    _has_deepseek_v4,
)


def _make_c4_spec() -> MLAAttentionSpec:
    """DeepSeekV4 C4 (compress_ratio=4) MLA cache spec on Ascend."""
    return MLAAttentionSpec(
        block_size=128 * 4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        model_version="deepseek_v4",
    )


def _make_c128_spec() -> MLAAttentionSpec:
    """DeepSeekV4 C128 (compress_ratio=128) MLA cache spec on Ascend."""
    return MLAAttentionSpec(
        block_size=128 * 128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=128,
        model_version="deepseek_v4",
    )


def _make_swa_spec() -> SlidingWindowMLASpec:
    """DeepSeekV4 sliding-window MLA cache spec on Ascend."""
    return SlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        sliding_window=512,
        model_version="deepseek_v4",
    )


def _make_deepseek_v4_specs(n_c4: int = 2, n_c128: int = 2, n_swa: int = 2) -> dict[str, KVCacheSpec]:
    """Build a DeepSeekV4 per-worker layer-spec dict."""
    specs: dict[str, KVCacheSpec] = {}
    for i in range(n_c4):
        specs[f"c4_{i}"] = _make_c4_spec()
    for i in range(n_c128):
        specs[f"c128_{i}"] = _make_c128_spec()
    for i in range(n_swa):
        specs[f"swa_{i}"] = _make_swa_spec()
    return specs


def _make_vllm_config(*, num_gpu_blocks_override: int | None = None, max_model_len: int = 128) -> SimpleNamespace:
    """Minimal mock VllmConfig for planning tests."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=num_gpu_blocks_override,
            prefix_cache_retention_interval=None,
            block_size=128,
        ),
        model_config=SimpleNamespace(
            original_max_model_len=max_model_len,
            max_model_len=max_model_len,
            kv_cache_config_builder_cls=None,
        ),
        speculative_config=None,
    )


def _monkeypatch_approximate_gcd(monkeypatch, value: int | None = None) -> None:
    """Pin the vLLM approximate-GCD helper so SWA splitting is deterministic."""
    monkeypatch.setattr(
        kv_cache_planning,
        "_approximate_gcd",
        (lambda values, lower_bound=None: value)
        if value is not None
        else (lambda values, lower_bound=None: lower_bound if lower_bound is not None else values[0]),
    )


def _collect_covered_layers(cfg) -> set[str]:
    """Layers covered by at least one emitted KV cache tensor."""
    return {name for tensor in cfg.kv_cache_tensors for name in tensor.shared_by}


# ---------------------------------------------------------------------------
# _has_deepseek_v4
# ---------------------------------------------------------------------------
def test_has_deepseek_v4_true_for_dsv4_layout() -> None:
    assert _has_deepseek_v4([_make_deepseek_v4_specs()])


def test_has_deepseek_v4_false_for_plain_attention() -> None:
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=torch.float16)
    assert not _has_deepseek_v4([{"layer0": full}])


def test_has_deepseek_v4_detects_nested_uniform_spec() -> None:
    uniform = UniformTypeKVCacheSpecs(block_size=512, kv_cache_specs={"c4": _make_c4_spec()})
    assert _has_deepseek_v4([{"layer0": uniform}])


# ---------------------------------------------------------------------------
# _ascend_group_and_unify_kv_cache_specs
# ---------------------------------------------------------------------------
def test_ascend_group_and_unify_kv_cache_specs_groups_by_block_size() -> None:
    specs = _make_deepseek_v4_specs(n_c4=2, n_c128=2, n_swa=2)
    grouped = _ascend_group_and_unify_kv_cache_specs(specs)

    assert grouped is not None
    # MLA groups come first (C4, then C128), SWA last; each is a uniform group.
    assert [group.block_size for group in grouped] == [512, 128 * 128, 128]
    assert set(grouped[0].kv_cache_specs) == {"c4_0", "c4_1"}
    assert set(grouped[1].kv_cache_specs) == {"c128_0", "c128_1"}
    assert set(grouped[2].kv_cache_specs) == {"swa_0", "swa_1"}
    assert all(isinstance(spec, SlidingWindowMLASpec) for spec in grouped[2].kv_cache_specs.values())


def test_ascend_group_and_unify_kv_cache_specs_no_swa_returns_none() -> None:
    specs = {"c4": _make_c4_spec(), "c128": _make_c128_spec()}
    assert _ascend_group_and_unify_kv_cache_specs(specs) is None


# ---------------------------------------------------------------------------
# _ascend_get_kv_cache_groups_uniform_groups
# ---------------------------------------------------------------------------
def test_ascend_get_kv_cache_groups_uniform_groups(monkeypatch) -> None:
    specs = _make_deepseek_v4_specs(n_c4=2, n_c128=2, n_swa=2)
    grouped = _ascend_group_and_unify_kv_cache_specs(specs)
    assert grouped is not None
    _monkeypatch_approximate_gcd(monkeypatch)

    groups = _ascend_get_kv_cache_groups_uniform_groups(grouped)

    assert len(groups) == 3
    assert groups[0].layer_names == ["c4_0", "c4_1"]
    assert groups[1].layer_names == ["c128_0", "c128_1"]
    assert groups[2].layer_names == ["swa_0", "swa_1"]
    # Every layer lands in exactly one group.
    assert sorted(name for group in groups for name in group.layer_names) == sorted(specs)


def test_ascend_get_kv_cache_groups_uniform_groups_splits_swa(monkeypatch) -> None:
    specs = _make_deepseek_v4_specs(n_c4=2, n_c128=2, n_swa=4)
    grouped = _ascend_group_and_unify_kv_cache_specs(specs)
    assert grouped is not None
    _monkeypatch_approximate_gcd(monkeypatch, value=2)

    groups = _ascend_get_kv_cache_groups_uniform_groups(grouped)

    # SWA layers are split into num_layer_tuples-sized sub-groups.
    swa_groups = groups[2:]
    assert len(swa_groups) == 2
    assert [sorted(group.layer_names) for group in swa_groups] == [["swa_0", "swa_2"], ["swa_1", "swa_3"]]


# ---------------------------------------------------------------------------
# _ascend_get_kv_cache_config_deepseek_v4
# ---------------------------------------------------------------------------
def test_ascend_get_kv_cache_config_deepseek_v4_layout(monkeypatch) -> None:
    specs = _make_deepseek_v4_specs(n_c4=2, n_c128=2, n_swa=2)
    grouped = _ascend_group_and_unify_kv_cache_specs(specs)
    assert grouped is not None
    _monkeypatch_approximate_gcd(monkeypatch, value=2)
    groups = _ascend_get_kv_cache_groups_uniform_groups(grouped)

    available_memory = 1 << 30  # 1 GiB
    cfg = _ascend_get_kv_cache_config_deepseek_v4(_make_vllm_config(), groups, available_memory)

    assert cfg.num_blocks > 0
    # One non-packed tensor per (tuple_idx, page_size) bucket; every layer covered.
    assert _collect_covered_layers(cfg) == {name for group in groups for name in group.layer_names}
    # num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples),
    # with num_layer_tuples == 2 for the 2-layer-per-group fixture.
    page_sizes = sorted(groups[0].kv_cache_spec.get_page_sizes())
    assert cfg.num_blocks == available_memory // (sum(page_sizes) * 2)
    # Tensor size is page_size * num_blocks.
    for tensor in cfg.kv_cache_tensors:
        assert tensor.size % cfg.num_blocks == 0
    # Config identity is preserved.
    assert [group.layer_names for group in cfg.kv_cache_groups] == [group.layer_names for group in groups]
    assert cfg.prefix_cache_retention_interval is None


def test_ascend_get_kv_cache_config_deepseek_v4_mtp_gets_own_tensor() -> None:
    c4 = _make_c4_spec()
    mtp = _make_c4_spec()
    c4_group = KVCacheGroupSpec(
        layer_names=["c4_0"],
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=c4.block_size, kv_cache_specs={"c4_0": c4}),
    )
    mtp_group = KVCacheGroupSpec(
        layer_names=["model.layers.0.mtp"],
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=mtp.block_size, kv_cache_specs={"model.layers.0.mtp": mtp}),
    )

    available_memory = 1 << 30
    cfg = _ascend_get_kv_cache_config_deepseek_v4(
        _make_vllm_config(), [c4_group, mtp_group], available_memory
    )

    mtp_tensors = [tensor for tensor in cfg.kv_cache_tensors if tensor.shared_by == ["model.layers.0.mtp"]]
    assert len(mtp_tensors) == 1
    assert mtp_tensors[0].size == mtp.page_size_bytes * cfg.num_blocks
    assert _collect_covered_layers(cfg) == {"c4_0", "model.layers.0.mtp"}


def test_ascend_get_kv_cache_config_deepseek_v4_num_gpu_blocks_override(monkeypatch) -> None:
    grouped = _ascend_group_and_unify_kv_cache_specs(_make_deepseek_v4_specs())
    assert grouped is not None
    _monkeypatch_approximate_gcd(monkeypatch, value=2)
    groups = _ascend_get_kv_cache_groups_uniform_groups(grouped)

    cfg = _ascend_get_kv_cache_config_deepseek_v4(
        _make_vllm_config(num_gpu_blocks_override=42), groups, 1 << 30
    )

    assert cfg.num_blocks == 42


# ---------------------------------------------------------------------------
# AscendKVCacheConfigBuilder.build_kv_cache_configs
# ---------------------------------------------------------------------------
def test_build_kv_cache_configs_deepseek_v4_end_to_end(monkeypatch) -> None:
    worker_specs = [_make_deepseek_v4_specs(), _make_deepseek_v4_specs()]
    vllm_config = _make_vllm_config()
    available_memory = [1 << 30, 1 << 30]

    # Memory-sufficiency checks depend on profiled memory; keep the Ascend
    # grouping/layout logic the focus of this test.
    monkeypatch.setattr(kv_cache_planning, "_check_enough_kv_cache_memory", lambda *args, **kwargs: None)

    configs = AscendKVCacheConfigBuilder().build_kv_cache_configs(vllm_config, worker_specs, available_memory)

    assert len(configs) == 2
    assert configs[0].num_blocks == configs[1].num_blocks
    assert configs[0].num_blocks > 0
    for cfg in configs:
        expected_layers = {name for group in cfg.kv_cache_groups for name in group.layer_names}
        assert _collect_covered_layers(cfg) == expected_layers
        assert cfg.prefix_cache_retention_interval is None


def test_build_kv_cache_configs_non_deepseek_v4_falls_back(monkeypatch) -> None:
    from vllm.v1.core.kv_cache_config_builder import KVCacheConfigBuilder

    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=torch.float16)
    worker_specs = [{"layer0": full}]
    vllm_config = _make_vllm_config()
    available_memory = [1024]

    calls = []

    def _fake_base_build(self, vllm_config_, kv_cache_specs, available_memory_):
        calls.append((vllm_config_, kv_cache_specs, available_memory_))
        return "fallback"

    monkeypatch.setattr(KVCacheConfigBuilder, "build_kv_cache_configs", _fake_base_build)

    result = AscendKVCacheConfigBuilder().build_kv_cache_configs(vllm_config, worker_specs, available_memory)

    assert result == "fallback"
    assert calls == [(vllm_config, worker_specs, available_memory)]


def test_platform_resolution_loads_ascend_builder() -> None:
    """Mirror vLLM PR #53558's resolve_builder test for the Ascend platform hook."""
    from vllm.v1.core.kv_cache_config_builder import resolve_builder

    with patch("vllm.platforms.current_platform") as mock_platform:
        mock_platform.get_kv_cache_config_builder_cls.return_value = (
            "vllm_ascend.worker.kv_cache_config_builder.AscendKVCacheConfigBuilder"
        )
        builder = resolve_builder(_make_vllm_config())

    assert isinstance(builder, AscendKVCacheConfigBuilder)
