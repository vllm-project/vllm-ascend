# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs

from tests.ut.kvpp_utils import make_cache_config, make_dspark_kvpp_case
from vllm_ascend.core.kv_cache_placement import create_kvpp_cache_allocation_plan
from vllm_ascend.worker import worker


def make_worker(monkeypatch, *, enabled=True, last_stage=True, pp_size=1):
    config, full_specs, drafts = make_dspark_kvpp_case()
    config.additional_config["enable_kvpp"] = enabled
    raw_specs = dict(full_specs)
    for name in drafts:
        raw_specs[name] = SlidingWindowSpec(
            block_size=1, num_kv_heads=2, head_size=8, dtype=torch.float16, sliding_window=4
        )
    instance = SimpleNamespace(
        vllm_config=config,
        model_runner=SimpleNamespace(
            get_kv_cache_spec=lambda: raw_specs,
            drafter=SimpleNamespace(_draft_attn_layer_names=set(drafts)),
        ),
        _kvpp_cache_allocation_plan=None,
    )
    instance.get_kv_cache_spec = MethodType(worker.NPUWorker.get_kv_cache_spec, instance)
    monkeypatch.setattr(worker, "get_layerwise_reuse_config", lambda _: None)
    monkeypatch.setattr(worker, "get_kvpp_group", lambda: SimpleNamespace(rank_in_group=1))
    monkeypatch.setattr(
        worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=last_stage, world_size=pp_size, cpu_group="pp")
    )
    monkeypatch.setattr(
        worker, "get_ascend_config", lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    )
    return instance, raw_specs, full_specs, drafts


def group_for(specs):
    return KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs=specs))


@pytest.mark.parametrize("enabled", [False, True])
def test_get_specs_preserves_raw_draft_specs_before_budget(monkeypatch, enabled):
    instance, raw, _, drafts = make_worker(monkeypatch, enabled=enabled)
    grouping = Mock(side_effect=AssertionError("Grouping must wait until the layout is resolved"))
    monkeypatch.setattr(worker, "get_kv_cache_groups", grouping)
    assert instance.get_kv_cache_spec() is raw
    assert instance._kvpp_cache_allocation_plan is None
    assert all(isinstance(raw[name], SlidingWindowSpec) and raw[name].block_size == 1 for name in drafts)
    if not enabled:
        assert worker.NPUWorker._apply_kvpp_memory_budget(instance, 8192) == 8192
    grouping.assert_not_called()


@pytest.mark.parametrize("method", ["mtp", "dspark"])
def test_budget_uses_native_resolved_specs_without_method_specific_conversion(monkeypatch, method):
    instance, raw, resolved, drafts = make_worker(monkeypatch)
    instance.vllm_config.speculative_config.method = method
    grouping = Mock(return_value=[group_for(resolved)])
    monkeypatch.setattr(worker, "get_kv_cache_groups", grouping)
    advertised = worker.NPUWorker._apply_kvpp_memory_budget(instance, 8192)
    expected = create_kvpp_cache_allocation_plan(instance.vllm_config, resolved, 1)
    assert instance._kvpp_cache_allocation_plan == expected
    assert advertised == expected.get_num_blocks(8192) * sum(s.page_size_bytes for s in resolved.values())
    assert grouping.call_args.args[1] == raw
    assert grouping.call_args.args[1] is not raw
    assert all(isinstance(raw[name], SlidingWindowSpec) for name in drafts)


@pytest.mark.parametrize("last_stage", [False, True])
def test_pp_budget_groups_global_specs_then_projects_to_local_layers(monkeypatch, last_stage):
    instance, raw, resolved, drafts = make_worker(monkeypatch, last_stage=last_stage, pp_size=2)
    stages = [{n: s for n, s in raw.items() if n not in drafts}, {n: raw[n] for n in drafts}]
    local = stages[int(last_stage)]
    instance.model_runner.get_kv_cache_spec = lambda: local

    def gather(output, value, *, group):
        assert value is local
        assert group == "pp"
        output[:] = stages

    monkeypatch.setattr(worker.torch.distributed, "all_gather_object", gather)
    grouping = Mock(return_value=[group_for(resolved)])
    monkeypatch.setattr(worker, "get_kv_cache_groups", grouping)
    worker.NPUWorker._apply_kvpp_memory_budget(instance, 8192)
    assert grouping.call_args.args[1] == raw
    assert instance._kvpp_cache_allocation_plan.logical_cache_spec == {n: resolved[n] for n in local}


def test_native_hybrid_groups_are_rejected_without_forcing_full_allocation(monkeypatch):
    instance, raw, resolved, drafts = make_worker(monkeypatch)
    groups = [
        group_for({n: s for n, s in resolved.items() if n not in drafts}),
        KVCacheGroupSpec(list(drafts), raw[drafts[0]]),
    ]
    monkeypatch.setattr(worker, "get_kv_cache_groups", lambda *_: groups)
    with pytest.raises(ValueError, match="single native KV cache group"):
        worker.NPUWorker._apply_kvpp_memory_budget(instance, 8192)
    assert instance._kvpp_cache_allocation_plan is None
    assert all(isinstance(raw[n], SlidingWindowSpec) for n in drafts)


@pytest.mark.parametrize("matches", [False, True])
def test_final_specs_are_checked_before_allocation(monkeypatch, matches):
    instance, raw, resolved, _ = make_worker(monkeypatch)
    monkeypatch.setattr(worker, "get_kv_cache_groups", lambda *_: [group_for(resolved)])
    worker.NPUWorker._apply_kvpp_memory_budget(instance, 8192)
    transfer = Mock()
    monkeypatch.setattr(worker, "ensure_kv_transfer_initialized", transfer)
    instance.model_runner.initialize_kv_cache = Mock()
    instance._maybe_get_memory_pool_context = lambda **_: nullcontext()
    instance.use_v2_model_runner = False
    cache_config = make_cache_config(resolved if matches else raw)
    if not matches:
        with pytest.raises(ValueError, match="differ from the native groups"):
            worker.NPUWorker.initialize_from_config(instance, cache_config)
        transfer.assert_not_called()
        instance.model_runner.initialize_kv_cache.assert_not_called()
    else:
        worker.NPUWorker.initialize_from_config(instance, cache_config)
        assert transfer.call_args.args[1] is cache_config
        assert instance.model_runner.initialize_kv_cache.call_args.args[0] is cache_config
