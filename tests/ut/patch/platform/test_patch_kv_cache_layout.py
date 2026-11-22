# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.patch.platform import patch_kv_cache_layout as layout_patch


def make_case():
    target = AscendMLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, model_version="glm5_next"
    )
    draft = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=32, dtype=torch.bfloat16, dcp_sharded=False)
    config = SimpleNamespace(
        cache_config=CacheConfig(),
        speculative_config=SimpleNamespace(method="dflash"),
        kv_transfer_config=None,
    )
    return config, target, draft


def layout_inputs(specs):
    # The newer Core wraps each spec with its cache-manager class.
    if "layer_configs" in inspect.signature(layout_patch._orig_resolve_kv_cache_layout).parameters:
        return [SimpleNamespace(spec=spec, manager_cls=SimpleNamespace(uses_slot_mapping=True)) for spec in specs]
    return specs


@pytest.mark.parametrize(
    "requested, expected", [("HND", "LBHNC"), ("LBHNC", "LBHNC"), ("NHD", "LBNHC"), ("LBNHC", "LBNHC")]
)
@pytest.mark.parametrize("model_version", [None, "glm5_next", "deepseek_v4"])
def test_dflash_private_pools_allow_layer_layout(monkeypatch, requested, expected, model_version):
    config, target, draft = make_case()
    target = replace(target, model_version=model_version)
    monkeypatch.setattr(layout_patch.envs, "VLLM_KV_CACHE_LAYOUT", requested)
    supported = [["LBNHC", "LBHNC", "BLNHC", "BLHNC"]]
    inputs = layout_inputs([target, draft])
    with pytest.raises(ValueError, match="valid layouts"):
        layout_patch._orig_resolve_kv_cache_layout(config, supported, inputs)
    result = layout_patch.resolve_kv_cache_layout(config, supported, inputs)
    assert result.name == expected
    assert config.cache_config.kv_cache_layout == expected
    assert target.dcp_sharded and not draft.dcp_sharded
    assert target.page_size_bytes != draft.page_size_bytes
    assert inputs == layout_inputs([target, draft])


@pytest.mark.parametrize("case", ["other_method", "unsupported_backend"])
def test_generic_layout_rejections_are_preserved(monkeypatch, case):
    config, target, draft = make_case()
    monkeypatch.setattr(layout_patch.envs, "VLLM_KV_CACHE_LAYOUT", "HND")
    supported = [["LBNHC", "LBHNC", "BLNHC", "BLHNC"]]
    if case == "other_method":
        config.speculative_config.method = "eagle3"
    else:
        supported = [["BLNHC", "BLHNC"]]
    with pytest.raises(ValueError, match="valid layouts"):
        layout_patch.resolve_kv_cache_layout(config, supported, layout_inputs([target, draft]))
    assert config.cache_config.kv_cache_layout is None
