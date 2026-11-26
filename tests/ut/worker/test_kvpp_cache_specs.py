# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec

from tests.ut.kvpp_utils import make_dspark_kvpp_case
from vllm_ascend.core.kv_cache_placement import create_kvpp_cache_allocation_plan
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker import worker


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("draft_block_size", [1, 2])
@pytest.mark.parametrize("enabled", [False, True])
def test_dspark_worker_returns_the_specs_used_for_kvpp_budget(monkeypatch, rank, draft_block_size, enabled):
    is_v034 = vllm_version_is("0.34.0")
    config, specs, drafts = make_dspark_kvpp_case()
    full_plan = create_kvpp_cache_allocation_plan(config, specs, rank)
    config.additional_config["enable_kvpp"] = enabled
    if is_v034:
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        for name in drafts:
            specs[name] = SlidingWindowSpec(
                block_size=draft_block_size, num_kv_heads=2, head_size=8, dtype=torch.float16, sliding_window=4
            )
        worker_reqs = specs
    else:
        worker_reqs = {name: FullAttentionManager.layer_config(spec) for name, spec in specs.items()}
        for name in drafts:
            worker_reqs[name] = SlidingWindowManager.layer_config(
                AttentionSpec(block_size=draft_block_size, num_kv_heads=2, head_size=8, dtype=torch.float16),
                sliding_window=4,
            )

    instance = worker.NPUWorker.__new__(worker.NPUWorker)
    instance.vllm_config = config
    if is_v034:
        instance.model_runner = SimpleNamespace(
            get_kv_cache_spec=lambda: worker_reqs, speculator=SimpleNamespace(draft_attn_layer_names=set(drafts))
        )
    else:
        instance.model_runner = SimpleNamespace(
            get_kv_cache_layer_configs=lambda: worker_reqs,
            speculator=SimpleNamespace(draft_attn_layer_names=set(drafts)),
        )
    instance._kvpp_cache_allocation_plan = None
    monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace(rank_in_group=rank))
    monkeypatch.setattr(worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(worker, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(
        worker, "get_ascend_config", lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    )
    returned = instance.get_kv_cache_spec() if is_v034 else instance.get_kv_cache_layer_configs()
    for name in drafts:
        if is_v034:
            assert isinstance(worker_reqs[name], SlidingWindowSpec)
            assert worker_reqs[name].block_size == draft_block_size
            assert worker_reqs[name].sliding_window == 4
        else:
            assert worker_reqs[name].manager_cls is SlidingWindowManager
            assert worker_reqs[name].spec.block_size == draft_block_size
            assert worker_reqs[name].manager_config.params.sliding_window == 4
    if not enabled:
        assert returned is worker_reqs
        assert instance._kvpp_cache_allocation_plan is None
        return
    plan = instance._kvpp_cache_allocation_plan
    assert plan is not None
    assert returned is not worker_reqs
    expected_logical = returned if is_v034 else {name: cfg.spec for name, cfg in returned.items()}
    assert plan.logical_cache_spec == expected_logical
    assert plan.layer_owner_ranks == full_plan.layer_owner_ranks
    assert plan.tensor_sizes == full_plan.tensor_sizes
    assert plan.get_num_blocks(8192) == full_plan.get_num_blocks(8192)
    for name, entry in returned.items():
        if name in drafts:
            if is_v034:
                assert isinstance(entry, FullAttentionSpec)
                assert entry.block_size == 2
            else:
                assert entry.manager_cls is FullAttentionManager
                assert isinstance(entry.spec, AttentionSpec)
                assert entry.spec.block_size == 2
            assert name not in plan.layer_owner_ranks
        else:
            assert entry is worker_reqs[name]
