from types import SimpleNamespace
from unittest.mock import Mock

import torch
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
)

from vllm_ascend.core import kv_cache_config_builder


def _ordinary_specs() -> dict[str, KVCacheSpec]:
    ordinary = FullAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    return {"model.layers.0.self_attn": ordinary}


def _ordinary_config() -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        kv_transfer_config=None,
    )


def test_non_qsa_grouping_delegates_to_upstream_vllm(monkeypatch) -> None:
    specs = _ordinary_specs()
    vllm_config = _ordinary_config()
    expected = [Mock(spec=KVCacheGroupSpec)]
    upstream_grouping = Mock(return_value=expected)
    monkeypatch.setattr(
        DefaultKVCacheConfigBuilder,
        "get_kv_cache_groups",
        upstream_grouping,
    )

    actual = kv_cache_config_builder.AscendKVCacheConfigBuilder().get_kv_cache_groups(
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
        kv_cache_config_builder,
        "_upstream_get_kv_cache_config_from_groups",
        upstream_planner,
    )
    monkeypatch.setattr(
        kv_cache_config_builder,
        "_is_deepseek_v4_groups",
        lambda _: False,
    )
    monkeypatch.setattr(
        kv_cache_config_builder,
        "_get_glm5_next_cache_layout",
        lambda _: None,
    )

    vllm_config = _ordinary_config()
    actual = kv_cache_config_builder.get_ascend_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory=123456,
    )

    assert actual is expected
    upstream_planner.assert_called_once_with(vllm_config, groups, 123456)
    assert actual.kv_transfer_config is None
