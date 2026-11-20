# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""TP replication contracts for native offloading, independent of NPU execution."""

from dataclasses import replace

import pytest
import torch

from tests.kv_cache_groups import cache_group, full_attention_spec_type
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)
from vllm_ascend.core.kv_cache_layers import group_layer_specs
from vllm_ascend.utils import vllm_version_is

# vLLM #61183 moved the planner helpers out of kv_cache_utils on main.
try:
    from vllm.v1.core.kv_cache_planning import kv_cache_groups_tp_replicas
except ModuleNotFoundError as exc:
    if exc.name != "vllm.v1.core.kv_cache_planning":
        raise
    from vllm.v1.core.kv_cache_utils import kv_cache_groups_tp_replicas


def _mla_spec():
    return AscendMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize(
    "main_layers,indexer_layers",
    [(78, 21), (61, 61)],
    ids=["glm-5.2-5.3-indexshare", "deepseek-v3.2"],
)
def test_mla_indexer_groups_publish_tp_replicas(tp_size, main_layers, indexer_layers):
    """Offloading must see one latent copy, including IndexShare's fewer indexers."""
    main = _mla_spec()
    indexer = AscendSFAIndexerCacheSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
    )
    specs = {
        **{f"main.{i}": main for i in range(main_layers)},
        **{f"indexer.{i}": indexer for i in range(indexer_layers)},
    }
    groups = [cache_group(list(specs), specs)]

    assert kv_cache_groups_tp_replicas(groups, tp_size) == tp_size
    assert kv_cache_groups_tp_replicas(groups, tp_size, dcp_size=2) == 1

    sharded = full_attention_spec_type(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        max_tp_shards=tp_size,
    )
    groups.append(cache_group(["sharded"], sharded))
    assert kv_cache_groups_tp_replicas(groups, tp_size) == 1


@pytest.mark.parametrize("spec_type", [AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec])
def test_mla_replication_metadata_survives_merge(spec_type):
    spec = spec_type(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)

    def representative(layout):
        if not vllm_version_is("0.33.0"):
            return group_layer_specs(cache_group(["a", "b"], layout))["a"]
        return spec_type.merge([layout, layout])

    merged = representative(spec)
    assert merged.max_tp_shards == 1
    unknown = replace(spec, max_tp_shards=None)
    assert representative(unknown).max_tp_shards is None
    groups = [cache_group(["unknown"], unknown)]
    assert kv_cache_groups_tp_replicas(groups, 4) == 1
