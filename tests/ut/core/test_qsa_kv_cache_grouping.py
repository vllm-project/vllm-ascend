from types import SimpleNamespace
from unittest.mock import Mock

import torch
from vllm.v1.kv_cache_interface import KVCacheSpec

from tests.kv_cache_groups import cache_group, full_attention_spec_type
from vllm_ascend.core.kv_cache_layers import HAS_LAYER_CONFIG

if HAS_LAYER_CONFIG:
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    from vllm_ascend.core import kv_cache_layer_config_builder as planner
else:
    from vllm_ascend.patch.platform import patch_kv_cache_utils


def _ordinary_specs() -> dict[str, KVCacheSpec]:
    ordinary = full_attention_spec_type(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    return {"model.layers.0.self_attn": ordinary}


def test_non_qsa_grouping_delegates_to_upstream_vllm(monkeypatch) -> None:
    specs = _ordinary_specs()
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3")))
    expected = [cache_group(list(specs), specs)]
    if HAS_LAYER_CONFIG:
        specs = {n: FullAttentionManager.layer_config(s) for n, s in specs.items()}
        vllm_config.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    upstream_grouping = Mock(return_value=expected)
    if HAS_LAYER_CONFIG:
        monkeypatch.setattr(planner.DefaultKVCacheConfigBuilder, "get_kv_cache_groups", upstream_grouping)
        actual = planner.AscendKVCacheConfigBuilder().get_kv_cache_groups(vllm_config, specs)
    else:
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
        cache_group(
            layer_names=["model.layers.0.self_attn"],
            kv_cache_spec=ordinary,
        )
    ]
    expected = SimpleNamespace()
    upstream_planner = Mock(return_value=expected)
    if HAS_LAYER_CONFIG:
        monkeypatch.setattr(planner.DefaultKVCacheConfigBuilder, "get_kv_cache_config_from_groups", upstream_planner)
    else:
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
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3")),
        kv_transfer_config=None,
    )

    if HAS_LAYER_CONFIG:
        actual = planner.AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(vllm_config, groups, 123456)
    else:
        actual = patch_kv_cache_utils._ascend_get_kv_cache_config_from_groups(
            vllm_config,
            groups,
            available_memory=123456,
        )

    assert actual is expected
    upstream_planner.assert_called_once_with(vllm_config, groups, 123456)
    assert actual.kv_transfer_config is None
