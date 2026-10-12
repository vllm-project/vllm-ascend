from types import SimpleNamespace
from unittest.mock import Mock

import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)

from vllm_ascend.patch.platform import patch_kv_cache_utils


def _qsa_specs() -> dict[str, KVCacheSpec]:
    main = FullAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    raw = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=32,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    specs: dict[str, KVCacheSpec] = {}
    for index in range(2):
        source = f"model.layers.{index * 4 + 3}.self_attn"
        specs[f"{source}.indexer.compressed_key_cache"] = compressed
        specs[f"{source}.indexer.raw_key_cache"] = raw
        specs[f"{source}.attn"] = main
    return specs


def test_qsa_grouping_delegates_to_upstream_vllm(monkeypatch) -> None:
    specs = _qsa_specs()
    vllm_config = SimpleNamespace()
    expected = [Mock(spec=KVCacheGroupSpec)]
    upstream_grouping = Mock(return_value=expected)
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_groups",
        upstream_grouping,
    )

    actual = patch_kv_cache_utils._get_ascend_kv_cache_groups(
        vllm_config,
        specs,
    )

    assert actual is expected
    upstream_grouping.assert_called_once_with(vllm_config, specs)


def test_qsa_config_uses_upstream_vllm_planner(monkeypatch) -> None:
    groups = [Mock(spec=KVCacheGroupSpec)]
    expected = SimpleNamespace()
    upstream_planner = Mock(return_value=expected)
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_get_kv_cache_config_from_groups",
        upstream_planner,
    )
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_is_deepseek_v4_groups",
        lambda _: False,
    )
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_get_glm5_next_cache_layout",
        lambda _: None,
    )
    vllm_config = SimpleNamespace(kv_transfer_config=None)

    actual = patch_kv_cache_utils._ascend_get_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory=123456,
    )

    assert actual is expected
    upstream_planner.assert_called_once_with(vllm_config, groups, 123456)
    assert actual.kv_transfer_config is None
