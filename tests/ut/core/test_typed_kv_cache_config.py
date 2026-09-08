# SPDX-License-Identifier: Apache-2.0

import copy
import math
import pickle
from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)

from vllm_ascend.core.typed_kv_cache import get_typed_kv_cache_plan
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _enable_typed_kv_cache_config,
)


def _make_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=128,
            _ascend_typed_attention_block_size=128,
            enable_prefix_caching=False,
            num_gpu_blocks_override=None,
            mamba_cache_mode="align",
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(watermark=0.0),
        model_config=SimpleNamespace(max_model_len=32768),
    )


def _make_uniform_qwen35_config() -> KVCacheConfig:
    attention = FullAttentionSpec(
        block_size=640,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
    )
    mamba = MambaSpec(
        block_size=640,
        shapes=((3, 6144), (16, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode="align",
    )
    group_layers = [
        [f"attn-{index}" for index in range(6)],
        [f"mamba-a-{index}" for index in range(6)],
        [f"mamba-b-{index}" for index in range(6)],
        [f"mamba-c-{index}" for index in range(6)],
    ]
    groups = [KVCacheGroupSpec(group_layers[0], attention)] + [
        KVCacheGroupSpec(names, mamba) for names in group_layers[1:]
    ]
    tensors = [
        KVCacheTensor(
            size=1_431_306_240,
            shared_by=[group[index] for group in group_layers],
        )
        for index in range(6)
    ]
    return KVCacheConfig(1092, tensors, groups)


def test_typed_plan_propagates_to_scheduler_and_worker_config() -> None:
    config = _make_uniform_qwen35_config()
    _enable_typed_kv_cache_config(_make_vllm_config(), config)

    plan = get_typed_kv_cache_plan(config)
    assert plan is not None
    assert plan.is_addressed
    assert plan.superpage_size_bytes == 0
    assert plan.num_superpages == 0
    assert [spec.block_size_tokens for spec in plan.specs] == [
        128,
        640,
        640,
        640,
    ]
    assert [spec.page_size_bytes for spec in plan.specs] == [
        262_144,
        1_085_440,
        1_085_440,
        1_085_440,
    ]
    assert plan.total_managed_bytes == 1_431_306_240
    assert config.num_blocks == min(plan.num_blocks(spec.group_id) for spec in plan.specs)
    assert {tensor.size for tensor in config.kv_cache_tensors} == {plan.total_managed_bytes}
    for spec, address_table in zip(plan.specs, plan.page_address_tables_bytes):
        assert len(address_table) == plan.num_blocks(spec.group_id)
        assert address_table[0] == 0
        assert all(offset % spec.page_size_bytes == 0 for offset in address_table)
        assert address_table[-1] + spec.page_size_bytes <= plan.total_managed_bytes
        assert plan.physical_block_id(spec.group_id, 1) == (address_table[1] // spec.page_size_bytes)

    worker_copy = pickle.loads(pickle.dumps(copy.deepcopy(config)))
    scheduler_copy = generate_scheduler_kv_cache_config([config])
    assert get_typed_kv_cache_plan(worker_copy) == plan
    assert get_typed_kv_cache_plan(scheduler_copy) == plan


def test_address_table_mode_rejects_prefix_cache() -> None:
    config = _make_uniform_qwen35_config()
    vllm_config = _make_vllm_config()
    vllm_config.cache_config.enable_prefix_caching = True

    try:
        _enable_typed_kv_cache_config(vllm_config, config)
    except ValueError as error:
        assert "jenga_lcm_prefix" in str(error)
    else:
        raise AssertionError("typed mode accepted prefix caching")


def test_jenga_lcm_prefix_mode_builds_exact_two_level_plan(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_ASCEND_TYPED_KV_CACHE_MODE",
        "jenga_lcm_prefix",
    )
    config = _make_uniform_qwen35_config()
    vllm_config = _make_vllm_config()
    vllm_config.cache_config.enable_prefix_caching = True

    _enable_typed_kv_cache_config(vllm_config, config)

    plan = get_typed_kv_cache_plan(config)
    assert plan is not None
    assert not plan.is_addressed
    assert not plan.is_partitioned
    assert plan.superpage_size_bytes == math.lcm(*(spec.page_size_bytes for spec in plan.specs))
    assert plan.num_superpages >= 2
    assert all(plan.superpage_size_bytes % spec.page_size_bytes == 0 for spec in plan.specs)


def test_jenga_lcm_prefix_mode_requires_prefix_cache(monkeypatch) -> None:
    monkeypatch.setenv(
        "VLLM_ASCEND_TYPED_KV_CACHE_MODE",
        "jenga_lcm_prefix",
    )

    try:
        _enable_typed_kv_cache_config(
            _make_vllm_config(),
            _make_uniform_qwen35_config(),
        )
    except ValueError as error:
        assert "requires prefix caching" in str(error)
    else:
        raise AssertionError("jenga_lcm_prefix accepted a no-prefix config")


def test_jenga_lcm_prefix_mode_rejects_nonzero_scheduler_watermark(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_ASCEND_TYPED_KV_CACHE_MODE",
        "jenga_lcm_prefix",
    )
    vllm_config = _make_vllm_config()
    vllm_config.cache_config.enable_prefix_caching = True
    vllm_config.scheduler_config.watermark = 0.01

    try:
        _enable_typed_kv_cache_config(
            vllm_config,
            _make_uniform_qwen35_config(),
        )
    except ValueError as error:
        assert "watermark=0" in str(error)
    else:
        raise AssertionError("jenga_lcm_prefix accepted a nonzero watermark")


def test_typed_mode_rejects_ascend_310p(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_kv_cache_utils.is_310p",
        lambda: True,
    )

    try:
        _enable_typed_kv_cache_config(
            _make_vllm_config(),
            _make_uniform_qwen35_config(),
        )
    except ValueError as error:
        assert "310P" in str(error)
    else:
        raise AssertionError("typed mode accepted Ascend 310P")


def test_static_partition_is_a_fixed_baseline_typed_plan(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_TYPED_KV_CACHE_MODE", "static_partition")
    config = _make_uniform_qwen35_config()
    vllm_config = _make_vllm_config()

    _enable_typed_kv_cache_config(vllm_config, config)

    plan = get_typed_kv_cache_plan(config)
    assert plan is not None and plan.is_addressed
    assert not plan.is_partitioned
    assert plan.total_managed_bytes == 1_431_306_240
    blocks_per_request = [
        math.ceil(group.kv_cache_spec.max_memory_usage_bytes(vllm_config) / group.kv_cache_spec.page_size_bytes)
        for group in config.kv_cache_groups
    ]
    assert min((plan.num_blocks(group_id) - 1) // blocks for group_id, blocks in enumerate(blocks_per_request)) >= 1
    data_intervals = []
    for spec, table in zip(plan.specs, plan.page_address_tables_bytes):
        for offset in table[1:]:
            data_intervals.append((offset, offset + spec.page_size_bytes))
    data_intervals.sort()
    assert all(left[1] <= right[0] for left, right in zip(data_intervals, data_intervals[1:]))
