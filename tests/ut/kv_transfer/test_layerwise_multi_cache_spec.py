# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native cache groups retain every entry of multi-cache physical layers."""

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.ut.distributed.ascend_store.cache_layout_utils import make_cache_config, make_vllm_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    LayerwiseCachePlanner,
    plan_layerwise_cache_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_kv_cache_specs,
)


@pytest.mark.parametrize(
    "suffixes",
    [
        ("attn", "indexer.k_cache", "indexer.compressor.state_cache", "compressor.state_cache", "swa_cache"),
        ("attn", "swa_cache"),
        ("attn", "indexer.k_cache"),
        ("a.indexer.k_cache", "b.indexer.k_cache"),
        ("indexer.k_cache",),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_multi_cache_entries_preserve_groups_and_do_not_alias_within_layer(suffixes, reverse):
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    names = [f"model.layers.{i}.self_attn.{suffix}" for i in range(4) for suffix in suffixes]
    if reverse:
        names.reverse()
    config = make_cache_config([(names, spec)])
    expanded = get_layerwise_kv_cache_specs(config)
    assert expanded == dict.fromkeys(names, spec)
    vllm_config = make_vllm_config(4, 1, layerwise_independent_layers=[])
    # DS-V4 has no reuse rule: the production planner must preserve its native layout.
    vllm_config.model_config.hf_text_config.model_type = "deepseek_v4"
    assert plan_layerwise_cache_reuse(config, vllm_config) is None
    # Exercise planner mechanics with an explicit rule, independently of model support.
    layout = LayerwiseCachePlanner(
        config,
        vllm_config,
        vllm_config.kv_transfer_config.kv_connector_extra_config,
        spec_reuse_rules={FullAttentionSpec: True},
    ).build()
    assert layout.layers == (0, 1, 2, 3)
    assert set(layout.cache_layer_indices) == set(names)
    assert sorted(name for tensor in layout.kv_cache_tensors for name in tensor.layers) == sorted(names)
    for tensor in layout.kv_cache_tensors:
        # Multiple entries in the same layer/group are live concurrently.
        layers = [layout.cache_layer_indices[name] for name in tensor.layers]
        assert len(layers) == len(set(layers))


def test_multi_group_entries_keep_all_save_dependencies():
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    groups = [
        ([f"model.layers.{i}.attn" for i in (0, 2, 4)], spec),
        ([f"model.layers.{i}.state" for i in (1, 3, 4)], spec),
    ]
    config = make_cache_config(groups)
    vllm_config = make_vllm_config(5, 1, layerwise_independent_layers=[])
    layout = LayerwiseCachePlanner(
        config,
        vllm_config,
        vllm_config.kv_transfer_config.kv_connector_extra_config,
        spec_reuse_rules={FullAttentionSpec: True},
    ).build()
    assert layout.prefetch_layer_map[4] == (2, 3)
