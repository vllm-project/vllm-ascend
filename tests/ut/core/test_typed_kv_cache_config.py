# SPDX-License-Identifier: Apache-2.0

import copy
import math
import pickle
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)

from vllm_ascend.core.typed_kv_cache import TypedAddressPool, TypedWorkerKVCacheSpecs, get_typed_kv_cache_plan
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _enable_typed_kv_cache_config,
)


def _make_vllm_config(
    *,
    mamba_cache_mode: str = "align",
    speculative_config: SimpleNamespace | None = None,
    target_model_type: str = "qwen3_5_text",
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=128,
            _ascend_typed_attention_block_size=128,
            enable_prefix_caching=False,
            num_gpu_blocks_override=None,
            mamba_cache_mode=mamba_cache_mode,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        speculative_config=speculative_config,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(watermark=0.0),
        model_config=SimpleNamespace(
            max_model_len=32768,
            hf_text_config=SimpleNamespace(model_type=target_model_type),
        ),
    )


def _make_speculative_config(
    method: str,
    draft_model_type: str,
    *,
    num_speculative_tokens: int = 3,
    parallel_drafting: bool = False,
    dynamic_tokens: list[tuple[int, int, int]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=draft_model_type),
        ),
        num_speculative_tokens=num_speculative_tokens,
        num_speculative_tokens_per_batch_size=dynamic_tokens,
        parallel_drafting=parallel_drafting,
    )


def _make_uniform_qwen35_config(
    *,
    mamba_cache_mode: str = "align",
    num_speculative_blocks: int = 0,
) -> KVCacheConfig:
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
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
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


@pytest.mark.parametrize(
    ("mode", "method", "draft_model_type"),
    [
        ("address_table", "qwen3_next_mtp", "qwen3_next_mtp"),
        ("address_table", "qwen3_5_mtp", "qwen3_5_mtp"),
        ("address_table", "mtp", "qwen3_next_mtp"),
        ("static_partition", "mtp", "qwen3_5_mtp"),
    ],
)
def test_typed_qwen_mtp_accepts_aliases_and_normalized_method(
    monkeypatch,
    mode: str,
    method: str,
    draft_model_type: str,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_TYPED_KV_CACHE_MODE", mode)
    num_speculative_tokens = 3
    speculative_config = _make_speculative_config(
        method,
        draft_model_type,
        num_speculative_tokens=num_speculative_tokens,
    )
    target_model_type = "qwen3_next" if draft_model_type == "qwen3_next_mtp" else "qwen3_5_text"
    vllm_config = _make_vllm_config(
        mamba_cache_mode="none",
        speculative_config=speculative_config,
        target_model_type=target_model_type,
    )
    config = _make_uniform_qwen35_config(
        mamba_cache_mode="none",
        num_speculative_blocks=num_speculative_tokens,
    )

    _enable_typed_kv_cache_config(vllm_config, config)

    plan = get_typed_kv_cache_plan(config)
    assert plan is not None and plan.is_addressed
    mamba_specs = [
        group.kv_cache_spec for group in config.kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    assert mamba_specs
    assert all(spec.mamba_cache_mode == "none" for spec in mamba_specs)
    assert all(spec.num_speculative_blocks == num_speculative_tokens for spec in mamba_specs)


@pytest.mark.parametrize(
    (
        "mode",
        "enable_prefix_caching",
        "mamba_cache_mode",
        "method",
        "draft_model_type",
        "parallel_drafting",
        "dynamic_tokens",
        "error_match",
    ),
    [
        (
            "jenga_lcm_prefix",
            True,
            "align",
            "mtp",
            "qwen3_5_mtp",
            False,
            None,
            "only supports address_table or static_partition",
        ),
        (
            "address_table",
            False,
            "align",
            "mtp",
            "qwen3_5_mtp",
            False,
            None,
            "Mamba cache mode 'none'",
        ),
        (
            "address_table",
            False,
            "none",
            "eagle",
            "qwen3_5_mtp",
            False,
            None,
            "only supports Qwen3-Next/Qwen3.5 MTP",
        ),
        (
            "address_table",
            False,
            "none",
            "mtp",
            "nemotron_h_mtp",
            False,
            None,
            "only supports Qwen3-Next/Qwen3.5 MTP",
        ),
        (
            "address_table",
            False,
            "none",
            "mtp",
            "qwen3_next_mtp",
            True,
            None,
            "does not support parallel drafting",
        ),
        (
            "static_partition",
            False,
            "none",
            "mtp",
            "qwen3_5_mtp",
            False,
            [(1, 8, 1)],
            "does not support dynamic speculative-token counts",
        ),
    ],
)
def test_typed_qwen_mtp_rejects_unsupported_modes_and_features(
    monkeypatch,
    mode: str,
    enable_prefix_caching: bool,
    mamba_cache_mode: str,
    method: str,
    draft_model_type: str,
    parallel_drafting: bool,
    dynamic_tokens: list[tuple[int, int, int]] | None,
    error_match: str,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_TYPED_KV_CACHE_MODE", mode)
    num_speculative_tokens = 3
    speculative_config = _make_speculative_config(
        method,
        draft_model_type,
        num_speculative_tokens=num_speculative_tokens,
        parallel_drafting=parallel_drafting,
        dynamic_tokens=dynamic_tokens,
    )
    target_model_type = "qwen3_next" if draft_model_type == "qwen3_next_mtp" else "qwen3_5_text"
    vllm_config = _make_vllm_config(
        mamba_cache_mode=mamba_cache_mode,
        speculative_config=speculative_config,
        target_model_type=target_model_type,
    )
    vllm_config.cache_config.enable_prefix_caching = enable_prefix_caching
    config = _make_uniform_qwen35_config(
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_tokens,
    )

    with pytest.raises(ValueError, match=error_match):
        _enable_typed_kv_cache_config(vllm_config, config)


def test_typed_qwen_mtp_rejects_mamba_speculative_block_mismatch() -> None:
    num_speculative_tokens = 3
    vllm_config = _make_vllm_config(
        mamba_cache_mode="none",
        speculative_config=_make_speculative_config(
            "mtp",
            "qwen3_5_mtp",
            num_speculative_tokens=num_speculative_tokens,
        ),
    )
    config = _make_uniform_qwen35_config(
        mamba_cache_mode="none",
        num_speculative_blocks=num_speculative_tokens - 1,
    )

    with pytest.raises(ValueError, match="num_speculative_blocks to equal num_speculative_tokens"):
        _enable_typed_kv_cache_config(vllm_config, config)


def test_typed_qwen_mtp_rejects_width_exceeding_ascendc_limit() -> None:
    num_speculative_tokens = 16
    vllm_config = _make_vllm_config(
        mamba_cache_mode="none",
        speculative_config=_make_speculative_config(
            "mtp",
            "qwen3_5_mtp",
            num_speculative_tokens=num_speculative_tokens,
        ),
    )
    config = _make_uniform_qwen35_config(
        mamba_cache_mode="none",
        num_speculative_blocks=num_speculative_tokens,
    )

    with pytest.raises(ValueError, match="between 1 and 15"):
        _enable_typed_kv_cache_config(vllm_config, config)


def test_typed_qwen_mtp_rejects_mismatched_target_family() -> None:
    num_speculative_tokens = 3
    vllm_config = _make_vllm_config(
        mamba_cache_mode="none",
        speculative_config=_make_speculative_config(
            "mtp",
            "qwen3_next_mtp",
            num_speculative_tokens=num_speculative_tokens,
        ),
        target_model_type="qwen3_5_text",
    )
    config = _make_uniform_qwen35_config(
        mamba_cache_mode="none",
        num_speculative_blocks=num_speculative_tokens,
    )

    with pytest.raises(ValueError, match="draft and target model families must match"):
        _enable_typed_kv_cache_config(vllm_config, config)


@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 5, 15])
def test_static_partition_maps_every_qwen_mtp_state_page(
    monkeypatch,
    num_speculative_tokens: int,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_TYPED_KV_CACHE_MODE", "static_partition")
    vllm_config = _make_vllm_config(
        mamba_cache_mode="none",
        speculative_config=_make_speculative_config(
            "mtp",
            "qwen3_5_mtp",
            num_speculative_tokens=num_speculative_tokens,
        ),
    )
    config = _make_uniform_qwen35_config(
        mamba_cache_mode="none",
        num_speculative_blocks=num_speculative_tokens,
    )
    _enable_typed_kv_cache_config(vllm_config, config)
    plan = get_typed_kv_cache_plan(config)
    assert plan is not None and plan.is_addressed

    mamba_group_id = 1
    state_page_count = 1 + num_speculative_tokens
    pool = TypedAddressPool(plan)
    state_blocks = pool.for_group(mamba_group_id).get_new_blocks(state_page_count)
    logical_ids = [block.block_id for block in state_blocks]
    physical_ids = [plan.physical_block_id(mamba_group_id, block_id) for block_id in logical_ids]
    kernel_address_table = plan.kernel_page_address_table(mamba_group_id)

    assert physical_ids == [kernel_address_table[block_id] for block_id in logical_ids]
    assert len(set(physical_ids)) == state_page_count
    assert any(logical_id != physical_id for logical_id, physical_id in zip(logical_ids, physical_ids))


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


@pytest.mark.parametrize("mode", ["address_table", "static_partition"])
def test_typed_config_uses_rpc_native_size_without_engine_worker_state(monkeypatch, mode) -> None:
    from vllm_ascend.patch.platform import patch_kv_cache_utils as patch

    monkeypatch.setenv("VLLM_ASCEND_ENABLE_TYPED_KV_CACHE", "1")
    monkeypatch.setenv("VLLM_ASCEND_TYPED_KV_CACHE_MODE", mode)
    config = _make_vllm_config()
    del config.cache_config._ascend_typed_attention_block_size
    config.cache_config.block_size = 896
    configs = [_make_uniform_qwen35_config() for _ in range(4)]
    worker_specs = pickle.loads(
        pickle.dumps(
            [TypedWorkerKVCacheSpecs({"attention": configs[i].kv_cache_groups[0].kv_cache_spec}, 128) for i in range(4)]
        )
    )
    monkeypatch.setattr(patch, "_orig_get_kv_cache_configs", lambda *_: configs)
    result = patch._ascend_get_kv_cache_configs(config, worker_specs, [4 * 1024**3] * 4)
    assert len(result) == 4
    for item in result:
        plan = get_typed_kv_cache_plan(item)
        assert plan is not None
        assert plan.specs[0].block_size_tokens == 128
    assert not hasattr(config.cache_config, "_ascend_typed_attention_block_size")
