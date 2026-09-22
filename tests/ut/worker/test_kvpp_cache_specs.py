# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from tests.ut.kvpp_utils import layer_name, make_cache_config, make_dspark_kvpp_case
from vllm_ascend.core.kv_cache_placement import create_kvpp_cache_allocation_plan, get_kvpp_cache_plan
from vllm_ascend.worker import worker


@pytest.mark.parametrize("pcp,rank", [(1, 0), (1, 1), (1, 2), (2, 4)])
@pytest.mark.parametrize("draft_block_size", [1, 2])
@pytest.mark.parametrize("enabled", [False, True])
def test_dspark_worker_returns_the_specs_used_for_kvpp_budget(monkeypatch, pcp, rank, draft_block_size, enabled):
    config, specs, drafts = make_dspark_kvpp_case()
    config.parallel_config.prefill_context_parallel_size = pcp
    full_plan = create_kvpp_cache_allocation_plan(config, specs, rank, draft_layer_names=drafts)
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
    monkeypatch.setattr(worker, "get_kvpp_group", lambda: SimpleNamespace(rank_in_group=rank))
    monkeypatch.setattr(worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(
        worker, "get_ascend_config", lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    )
    instance._get_kvpp_draft_layer_names = MethodType(worker.NPUWorker._get_kvpp_draft_layer_names, instance)
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


@pytest.mark.parametrize("method", ["mtp", "dspark"])
@pytest.mark.parametrize("v2", [False, True])
def test_worker_reads_loaded_proposer_without_mutating_layers(monkeypatch, method, v2):
    config, specs, drafts = make_dspark_kvpp_case(
        draft_names=("draft.layers.9.attn", "draft.layers.103.attn", "draft.cache")
    )
    config.speculative_config.method = method
    config.use_v2_model_runner = v2
    names = set(drafts)
    runner = SimpleNamespace(
        drafter=SimpleNamespace(_draft_attn_layer_names=names),
        speculator=SimpleNamespace(draft_attn_layer_names=names),
    )
    instance = SimpleNamespace(vllm_config=config, model_runner=runner)
    monkeypatch.setattr(worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    result = worker.NPUWorker._get_kvpp_draft_layer_names(instance)
    assert result == names
    assert result is not names
    assert all(set(vars(layer)) == {"impl"} for layer in config.compilation_config.static_forward_context.values())


@pytest.mark.parametrize("v2", [False, True])
def test_missing_proposer_only_valid_on_non_draft_pp_stage(monkeypatch, v2):
    config, _, _ = make_dspark_kvpp_case()
    config.use_v2_model_runner = v2
    instance = SimpleNamespace(vllm_config=config, model_runner=SimpleNamespace(drafter=None, speculator=None))
    monkeypatch.setattr(worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    with pytest.raises(ValueError, match="loaded proposer"):
        worker.NPUWorker._get_kvpp_draft_layer_names(instance)
    monkeypatch.setattr(worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False))
    assert worker.NPUWorker._get_kvpp_draft_layer_names(instance) == set()


@pytest.mark.parametrize("via_runner", [False, True])
def test_worker_rejects_draft_cache_aliasing_target(monkeypatch, via_runner):
    config, _, drafts = make_dspark_kvpp_case()
    alias = "draft.shared.attn"
    runner = SimpleNamespace(drafter=SimpleNamespace(_draft_attn_layer_names=set(drafts) | {alias}))
    context = config.compilation_config.static_forward_context
    context[alias] = SimpleNamespace()
    if via_runner:
        runner.shared_kv_cache_layers = {alias: layer_name(9)}
    else:
        context[alias].kv_sharing_target_layer_name = layer_name(9)
    instance = SimpleNamespace(vllm_config=config, model_runner=runner)
    monkeypatch.setattr(worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    with pytest.raises(ValueError, match="shares target cache"):
        worker.NPUWorker._get_kvpp_draft_layer_names(instance)


def test_worker_passes_budgeted_plan_to_runner_and_transfer_connector(monkeypatch):
    config, specs, drafts = make_dspark_kvpp_case()
    plan = create_kvpp_cache_allocation_plan(config, specs, 1, draft_layer_names=drafts)
    transfer_init = Mock()
    runner = SimpleNamespace(initialize_kv_cache=Mock())
    instance = SimpleNamespace(
        vllm_config=config,
        model_runner=runner,
        use_v2_model_runner=False,
        _kvpp_cache_allocation_plan=plan,
        _maybe_get_memory_pool_context=lambda **_: nullcontext(),
    )
    monkeypatch.setattr(worker, "ensure_kv_transfer_initialized", transfer_init)
    engine_config = make_cache_config(specs)
    worker.NPUWorker.initialize_from_config(instance, engine_config)
    local_config = runner.initialize_kv_cache.call_args.args[0]
    assert local_config is not engine_config
    assert get_kvpp_cache_plan(local_config) is plan
    assert transfer_init.call_args.args[1] is local_config
