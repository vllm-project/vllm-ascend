# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tests.kv_cache_groups import full_attention_spec_type, layer_config, window_requirement
from tests.ut.kvpp_utils import make_dspark_kvpp_case
from vllm_ascend.core.kv_cache_layers import cache_layouts
from vllm_ascend.core.kv_cache_placement import create_kvpp_cache_allocation_plan
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker import worker


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("draft_block_size", [1, 2])
@pytest.mark.parametrize("enabled", [False, True])
def test_dspark_worker_returns_the_specs_used_for_kvpp_budget(monkeypatch, rank, draft_block_size, enabled):
    config, specs, drafts = make_dspark_kvpp_case()
    full_plan = create_kvpp_cache_allocation_plan(config, specs, rank)
    config.additional_config["enable_kvpp"] = enabled
    for name in drafts:
        specs[name] = window_requirement(
            block_size=draft_block_size, num_kv_heads=2, head_size=8, dtype=torch.float16, sliding_window=4
        )
    requirements = {n: layer_config(s) for n, s in specs.items()} if not vllm_version_is("0.33.0") else specs
    method = "get_kv_cache_layer_configs" if not vllm_version_is("0.33.0") else "get_kv_cache_spec"
    instance = worker.NPUWorker.__new__(worker.NPUWorker)
    instance.vllm_config = config
    instance.model_runner = SimpleNamespace(
        speculator=SimpleNamespace(draft_attn_layer_names=set(drafts)), **{method: lambda: requirements}
    )
    instance._kvpp_cache_allocation_plan = None
    monkeypatch.setattr(worker, "get_layerwise_reuse_config", lambda _, *, vllm_config: None)
    monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace(rank_in_group=rank))
    monkeypatch.setattr(worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(worker, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(
        worker, "get_ascend_config", lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    )
    returned = getattr(instance, method)()
    for name in drafts:
        assert cache_layouts(requirements)[name].block_size == draft_block_size
        assert (specs[name].manager_config.params if not vllm_version_is("0.33.0") else specs[name]).sliding_window == 4
    if not enabled:
        assert returned is requirements
        assert instance._kvpp_cache_allocation_plan is None
        return
    plan = instance._kvpp_cache_allocation_plan
    assert returned is not requirements
    assert plan.logical_cache_spec == cache_layouts(returned)
    assert plan.layer_owner_ranks == full_plan.layer_owner_ranks
    assert plan.tensor_sizes == full_plan.tensor_sizes
    assert plan.get_num_blocks(8192) == full_plan.get_num_blocks(8192)
    for name, spec in cache_layouts(returned).items():
        if name in drafts:
            assert isinstance(spec, full_attention_spec_type)
            assert spec.block_size == 2
            assert name not in plan.layer_owner_ranks
        else:
            assert spec is specs[name]
