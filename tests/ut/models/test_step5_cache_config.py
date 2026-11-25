# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Unit tests for the Step5 KV cache planner (vllm_ascend.models.step5.cache_config).

The planner mirrors upstream's block-outermost Step5 layout on Ascend's
layer-outer LBNHC layout: one envelope-sized aliased backing whose per-id
cost is the widest group, the sparse summary sharing its DSA parent's group
and block-id chain, and the CSA tail ring keeping its own
``CircularBufferSpec`` group on the main page class, stacked from byte 0
like every other group so per-id spans of different groups can only
intersect at identical block ids.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.models.step5.cache_config import (
    _padded_page,
    get_step5_kv_cache_config_from_groups,
    get_step5_kv_cache_groups,
    get_step5_max_memory_usage,
    get_step5_pool_bytes_per_block,
    is_step5_groups,
    is_step5_kv_cache_spec,
)

BLOCK = 128
# 96 KiB main DSA page: 2 (k+v) x 4 kv heads x 48 head_size x bf16 x 128 tok.
MAIN_PAGE = 98304
# 4 KiB FP8 sparse summary page: 2 x 1 head x 16 head_size x fp8 x 128 tok.
SUMMARY_PAGE = 4096
# The CSA tail ring mirrors the main page class (96 KiB, capacity 96 rows):
# 1 head x 384 head_size x bf16 x 128 tok.
RING_PAGE = MAIN_PAGE
RING_ALIGNMENT = 1024

L0 = "model.layers.0.self_attn"
L1 = "model.layers.1.self_attn"
L2 = "model.layers.2.self_attn"
RING = f"{L0}.csa_tail_ring"
SUMMARY0 = f"{L0}.sparse_summary_cache"
SUMMARY2 = f"{L2}.sparse_summary_cache"


def _main_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=4,
        head_size=48,
        head_size_v=48,
        dtype=torch.bfloat16,
    )


def _sw_spec() -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=BLOCK,
        num_kv_heads=4,
        head_size=48,
        head_size_v=48,
        dtype=torch.bfloat16,
        sliding_window=4096,
    )


def _summary_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float8_e4m3fn,
    )


def _ring_spec() -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=384,
        head_size_v=0,
        dtype=torch.bfloat16,
        block_stride_alignment=RING_ALIGNMENT,
    )


def _step5_kv_cache_spec() -> dict[str, KVCacheSpec]:
    """Two DSA layers with summaries, one sliding-window layer, one ring."""
    return {
        L0: _main_spec(),
        L2: _main_spec(),
        L1: _sw_spec(),
        SUMMARY0: _summary_spec(),
        SUMMARY2: _summary_spec(),
        RING: _ring_spec(),
    }


def _fake_orig_get_kv_cache_groups(vllm_config, kv_cache_spec):
    """Upstream-style grouping: full layers together, one SW group."""
    assert all(not name.endswith((".csa_tail_ring", ".sparse_summary_cache")) for name in kv_cache_spec)
    full = [name for name, spec in kv_cache_spec.items() if type(spec) is FullAttentionSpec]
    sliding = [name for name, spec in kv_cache_spec.items() if isinstance(spec, SlidingWindowSpec)]
    groups = []
    if full:
        uniform = UniformTypeKVCacheSpecs.from_specs({name: kv_cache_spec[name] for name in full})
        groups.append(KVCacheGroupSpec(full, uniform))
    if sliding:
        uniform = UniformTypeKVCacheSpecs.from_specs({name: kv_cache_spec[name] for name in sliding})
        groups.append(KVCacheGroupSpec(sliding, uniform))
    return groups


def _planned_groups() -> list[KVCacheGroupSpec]:
    groups = get_step5_kv_cache_groups(None, _step5_kv_cache_spec(), _fake_orig_get_kv_cache_groups)
    assert groups is not None
    return groups


def _vllm_config(max_model_len: int = 8192, num_gpu_blocks_override: int | None = None):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=8192,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=num_gpu_blocks_override,
            prefix_cache_retention_interval=None,
        ),
    )


def _planner_vllm_config():
    """Config for builder/planner dispatch: they read hf_config.model_type."""
    config = _vllm_config()
    config.model_config.hf_config = SimpleNamespace(model_type="step5")
    # The generic planner probes vllm_config.attention_config.hisparse_config
    # before consulting the builder's pool-bytes hook.
    config.attention_config = SimpleNamespace(hisparse_config=None)
    return config


def test_is_step5_detection_matches_aux_suffixes() -> None:
    assert is_step5_kv_cache_spec(_step5_kv_cache_spec())
    assert is_step5_kv_cache_spec({RING: _ring_spec()})
    assert is_step5_kv_cache_spec({SUMMARY0: _summary_spec()})
    assert not is_step5_kv_cache_spec({L0: _main_spec(), L1: _sw_spec()})

    assert is_step5_groups(_planned_groups())
    assert not is_step5_groups([KVCacheGroupSpec([L0], _main_spec())])


def test_grouping_merges_summary_into_parent_group_and_keeps_ring_separate() -> None:
    groups = _planned_groups()

    assert [group.layer_names for group in groups] == [
        # Parents lead the merged group so the scheduler representative spec
        # is the DSA FullAttentionSpec, not the 4 KiB summary.
        [L0, L2, SUMMARY0, SUMMARY2],
        [L1],
        [RING],
    ]
    merged = groups[0].kv_cache_spec
    assert isinstance(merged, UniformTypeKVCacheSpecs)
    assert merged.kv_cache_specs[L0].page_size_bytes == MAIN_PAGE
    assert merged.kv_cache_specs[SUMMARY0].page_size_bytes == SUMMARY_PAGE
    # The ring keeps its own group and stays a CircularBufferSpec.
    assert isinstance(groups[2].kv_cache_spec, UniformTypeKVCacheSpecs)
    ring_spec = groups[2].kv_cache_spec.kv_cache_specs[RING]
    assert isinstance(ring_spec, CircularBufferSpec)
    assert ring_spec.page_size_bytes == RING_PAGE


def test_grouping_returns_none_without_step5_aux_layers() -> None:
    calls = []

    def tracking_orig(vllm_config, kv_cache_spec):
        calls.append(kv_cache_spec)
        return []

    specs = {L0: _main_spec(), L1: _sw_spec()}
    assert get_step5_kv_cache_groups(None, specs, tracking_orig) is None
    assert calls == []


def test_grouping_isolates_orphan_summary_layers() -> None:
    orphan = "model.layers.9.self_attn.sparse_summary_cache"
    specs = {
        L0: _main_spec(),
        SUMMARY0: _summary_spec(),
        RING: _ring_spec(),
        orphan: _summary_spec(),
    }
    groups = get_step5_kv_cache_groups(None, specs, _fake_orig_get_kv_cache_groups)
    assert groups is not None
    assert [group.layer_names for group in groups] == [
        [L0, SUMMARY0],
        [orphan],
        [RING],
    ]


def test_grouping_splits_aux_layers_with_mixed_geometry() -> None:
    """Aux layers the wrapper cannot merge fall back to one group each."""
    small = FullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float8_e4m3fn,
    )
    orphan_a = "model.layers.8.self_attn.sparse_summary_cache"
    orphan_b = "model.layers.9.self_attn.sparse_summary_cache"
    specs = {
        L0: _main_spec(),
        SUMMARY0: _summary_spec(),
        orphan_a: _summary_spec(),
        orphan_b: small,
    }
    groups = get_step5_kv_cache_groups(None, specs, _fake_orig_get_kv_cache_groups)
    assert groups is not None
    assert [group.layer_names for group in groups] == [
        [L0, SUMMARY0],
        [orphan_a],
        [orphan_b],
    ]


def test_padded_page_pads_to_block_stride_alignment() -> None:
    odd = FullAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=63,
        dtype=torch.bfloat16,
        block_stride_alignment=RING_ALIGNMENT,
    )
    # 2 x 1 x 63 x 2 x 128 = 32256 bytes; next 1024 multiple is 32768.
    assert odd.page_size_bytes == 32256
    assert _padded_page(odd) == 32768
    assert _padded_page(_main_spec()) == MAIN_PAGE


def test_pool_bytes_per_block_is_widest_group_width() -> None:
    groups = _planned_groups()
    # Merged DSA group: 2 x 96 KiB + 2 x 4 KiB; SW and ring are narrower.
    assert get_step5_pool_bytes_per_block(groups) == 2 * MAIN_PAGE + 2 * SUMMARY_PAGE


def test_max_memory_usage_sums_group_demands_times_envelope() -> None:
    vllm_config = _vllm_config(max_model_len=8192)
    # DSA/summary group: 64 ids at 8192 tokens; SW saturates at its admission
    # cap cdiv(min(4095 + 8192, 8192), 128) + 1 = 65; ring holds 1 bucket.
    expected = (64 + 65 + 1) * (2 * MAIN_PAGE + 2 * SUMMARY_PAGE)
    assert get_step5_max_memory_usage(vllm_config, _planned_groups()) == expected


def test_max_memory_usage_skips_host_resident_groups() -> None:
    vllm_config = _vllm_config(max_model_len=8192)
    groups = _planned_groups()
    sw_name, sw_spec = groups[1].layer_names, groups[1].kv_cache_spec
    groups[1] = KVCacheGroupSpec(sw_name, sw_spec, host_resident=True)
    expected = (64 + 1) * (2 * MAIN_PAGE + 2 * SUMMARY_PAGE)
    assert get_step5_max_memory_usage(vllm_config, groups) == expected


def test_config_from_groups_aliases_one_envelope_backing() -> None:
    vllm_config = _vllm_config(num_gpu_blocks_override=3)
    groups = _planned_groups()
    envelope = 2 * MAIN_PAGE + 2 * SUMMARY_PAGE
    config = get_step5_kv_cache_config_from_groups(vllm_config, groups, 10**9)

    assert config.num_blocks == 3
    assert config.kv_cache_groups == groups
    assert config.prefix_cache_retention_interval is None
    backing_size = envelope * 3
    by_layer = {tensor.layers[0]: tensor for tensor in config.kv_cache_tensors}
    assert set(by_layer) == {L0, L2, SUMMARY0, SUMMARY2, L1, RING}
    assert all(tensor.size == backing_size for tensor in config.kv_cache_tensors)

    # Main/summary layers stack from byte 0 in planner order.
    assert by_layer[L0].offset == 0
    assert by_layer[L2].offset == MAIN_PAGE * 3
    assert by_layer[SUMMARY0].offset == 2 * MAIN_PAGE * 3
    assert by_layer[SUMMARY2].offset == (2 * MAIN_PAGE + SUMMARY_PAGE) * 3
    assert by_layer[L1].offset == 0
    # The ring stack starts at byte 0 like every other group: it shares the
    # main page class, so the same-class start dedups instead of colliding.
    assert by_layer[RING].offset == 0

    assert by_layer[L0].layer_stride == MAIN_PAGE * 3
    assert by_layer[L0].block_stride == MAIN_PAGE
    assert by_layer[SUMMARY0].layer_stride == SUMMARY_PAGE * 3
    assert by_layer[RING].layer_stride == RING_PAGE * 3
    assert by_layer[RING].block_stride == RING_PAGE


def test_config_from_groups_derives_num_blocks_from_available_memory() -> None:
    vllm_config = _vllm_config()
    groups = _planned_groups()
    envelope = 2 * MAIN_PAGE + 2 * SUMMARY_PAGE
    config = get_step5_kv_cache_config_from_groups(vllm_config, groups, envelope * 7 + 123)
    assert config.num_blocks == 7


def test_config_from_groups_ring_only_layout_stacks_from_byte_zero() -> None:
    # A ring-only layout: the ring IS the envelope, and its stack starts at
    # byte 0 like every other group -- no per-class start shift remains now
    # that the ring shares the attention stride class.
    ring_group = KVCacheGroupSpec([RING], _ring_spec())
    vllm_config = _vllm_config()
    config = get_step5_kv_cache_config_from_groups(vllm_config, [ring_group], RING_PAGE * 3)

    assert config.num_blocks == 3
    (tensor,) = config.kv_cache_tensors
    assert tensor.size == RING_PAGE * 3
    assert tensor.offset == 0
    assert tensor.layer_stride == RING_PAGE * 3
    assert tensor.block_stride == RING_PAGE


def test_builder_and_generic_planner_dispatch_step5_groups_to_step5_planner() -> None:
    planning = pytest.importorskip("vllm.v1.core.kv_cache_planning")
    vllm_config = _planner_vllm_config()
    groups = _planned_groups()
    envelope = 2 * MAIN_PAGE + 2 * SUMMARY_PAGE

    builder = AscendKVCacheConfigBuilder.__new__(AscendKVCacheConfigBuilder)
    assert builder._get_pool_bytes_per_block(vllm_config, groups) == envelope
    expected_usage = (64 + 65 + 1) * envelope
    assert planning._max_memory_usage_bytes_from_groups(builder, vllm_config, groups) == expected_usage


def test_builder_pool_bytes_keeps_default_layout_without_step5_aux_layers() -> None:
    """Groups without Step5 auxiliary layers must not hit the step5 divisor:
    the builder leaves the layout to the upstream default (no custom
    per-block bytes)."""
    builder = AscendKVCacheConfigBuilder.__new__(AscendKVCacheConfigBuilder)
    vllm_config = _planner_vllm_config()

    plain_groups = [KVCacheGroupSpec([L0], _main_spec()), KVCacheGroupSpec([L1], _sw_spec())]
    assert builder._get_pool_bytes_per_block(vllm_config, plain_groups) is None

    # Adding one summary layer flips the same groups onto the step5 envelope.
    merged = UniformTypeKVCacheSpecs.from_specs({L0: _main_spec(), SUMMARY0: _summary_spec()})
    step_groups = [KVCacheGroupSpec([L0, SUMMARY0], merged), KVCacheGroupSpec([L1], _sw_spec())]
    assert builder._get_pool_bytes_per_block(vllm_config, step_groups) == MAIN_PAGE + SUMMARY_PAGE
