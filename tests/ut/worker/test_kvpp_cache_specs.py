# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from tests.ut.kvpp_utils import make_dspark_kvpp_case
from vllm_ascend.core.kv_cache_placement import create_kvpp_cache_allocation_plan
from vllm_ascend.worker import worker


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("draft_block_size", [1, 2])
@pytest.mark.parametrize("enabled", [False, True])
def test_dspark_worker_returns_the_specs_used_for_kvpp_budget(monkeypatch, rank, draft_block_size, enabled):
    config, specs, drafts = make_dspark_kvpp_case()
    full_plan = create_kvpp_cache_allocation_plan(config, specs, rank)
    config.additional_config["enable_kvpp"] = enabled
    for name in drafts:
        specs[name] = SlidingWindowSpec(
            block_size=draft_block_size, num_kv_heads=2, head_size=8, dtype=torch.float16, sliding_window=4
        )
    instance = SimpleNamespace(
        vllm_config=config,
        model_runner=SimpleNamespace(
            get_kv_cache_spec=lambda: specs, drafter=SimpleNamespace(_draft_attn_layer_names=set(drafts))
        ),
        _kvpp_cache_allocation_plan=None,
    )
    monkeypatch.setattr(worker, "get_layerwise_reuse_config", lambda _: None)
    monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace(rank_in_group=rank))
    monkeypatch.setattr(worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(worker, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(
        worker, "get_ascend_config", lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    )
    returned = worker.NPUWorker.get_kv_cache_spec(instance)
    for name in drafts:
        assert isinstance(specs[name], SlidingWindowSpec)
        assert specs[name].block_size == draft_block_size
        assert specs[name].sliding_window == 4
    if not enabled:
        assert returned is specs
        assert instance._kvpp_cache_allocation_plan is None
        return
    plan = instance._kvpp_cache_allocation_plan
    assert returned is not specs
    assert plan.logical_cache_spec == returned
    assert plan.layer_owner_ranks == full_plan.layer_owner_ranks
    assert plan.tensor_sizes == full_plan.tensor_sizes
    assert plan.get_num_blocks(8192) == full_plan.get_num_blocks(8192)
    for name, spec in returned.items():
        if name in drafts:
            assert isinstance(spec, FullAttentionSpec)
            assert spec.block_size == 2
            assert name not in plan.layer_owner_ranks
        else:
            assert spec is specs[name]
