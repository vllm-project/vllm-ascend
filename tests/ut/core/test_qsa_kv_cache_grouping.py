from types import SimpleNamespace
from unittest.mock import Mock

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
)

from vllm_ascend.patch.platform import patch_kv_cache_utils


def _ordinary_specs() -> dict[str, KVCacheSpec]:
    ordinary = FullAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    return {"model.layers.0.self_attn": ordinary}


def test_non_qsa_grouping_delegates_to_upstream_vllm(monkeypatch) -> None:
    specs = _ordinary_specs()
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


def test_non_qsa_config_uses_upstream_vllm_planner(monkeypatch) -> None:
    ordinary = next(iter(_ordinary_specs().values()))
    groups = [
        KVCacheGroupSpec(
            layer_names=["model.layers.0.self_attn"],
            kv_cache_spec=ordinary,
        )
    ]
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
