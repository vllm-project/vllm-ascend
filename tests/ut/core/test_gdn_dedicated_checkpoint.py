# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from vllm_ascend.core.gdn_checkpoint_policy import GDNCheckpointPolicy
from vllm_ascend.core.gdn_dedicated_checkpoint import (
    DedicatedGDNCheckpoints,
    GDNCheckpointAcknowledgement,
    confirmed_writes,
)
from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.patch.platform import patch_mamba_block_aligned_split as split_patch
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager


def managers():
    pool = Mock()
    pool.get_num_free_blocks.return_value = 10
    groups = tuple(
        SimpleNamespace(
            block_pool=pool,
            block_size=2048,
            kv_cache_spec=SimpleNamespace(prefill_checkpoint_alignment=64),
            kv_cache_group_id=i,
            dedicated_plans={},
            dedicated_pages={},
        )
        for i in (1, 2)
    )
    return groups, pool


def test_plan_accounts_for_every_extra_page():
    groups, pool = managers()
    state = DedicatedGDNCheckpoints(groups)
    pool.get_num_free_blocks.return_value = 9
    assert not state.plan("req", 2048, 3456, 3500, 128, 2048, 8)
    assert all(not group.dedicated_plans for group in groups)
    pool.get_num_free_blocks.return_value = 10
    assert state.plan("req", 2048, 3456, 3500, 128, 2048, 8)
    assert all(group.dedicated_plans == {"req": (3456, 3500)} for group in groups)
    state.begin_step()
    assert all(not group.dedicated_plans for group in groups)


@pytest.mark.parametrize(
    "available_blocks,external_tokens,split_tokens,junction_enabled",
    [(10, 0, 1408, False), (3, 0, 1408, False), (10, 1, 1408, False), (10, 0, 1024, False), (10, 0, 1408, True)],
)
def test_scheduler_replaces_only_an_admitted_same_page_tail_stop(
    available_blocks, external_tokens, split_tokens, junction_enabled
):
    groups, pool = managers()
    pool.get_num_free_blocks.return_value = available_blocks
    scheduler = SimpleNamespace(
        _ascend_dedicated_checkpoints=DedicatedGDNCheckpoints(groups),
        _ascend_gdn_checkpoint_policy=GDNCheckpointPolicy(8192, True, False),
        cache_config=SimpleNamespace(block_size=2048),
        hash_block_size=128,
        mamba_shared_prefix_checkpoint=junction_enabled,
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=1,
        num_lookahead_tokens=0,
        kv_cache_manager=SimpleNamespace(
            watermark_blocks=0,
            coordinator=SimpleNamespace(single_type_managers=groups),
        ),
    )
    request = SimpleNamespace(
        request_id="req", num_tokens=3500, num_prompt_tokens=3500, num_computed_tokens=0, shared_prefix_boundary=2176
    )
    chosen = split_patch._maybe_use_dedicated_checkpoint(scheduler, request, 1452, 2048, external_tokens, split_tokens)
    eligible = available_blocks == 10 and external_tokens == 0 and split_tokens == 1408 and not junction_enabled
    assert chosen == (1452 if eligible else split_tokens)
    assert all(bool(group.dedicated_plans) == eligible for group in groups)


@pytest.mark.parametrize("failed_gate", [None, "one_group", "preempted", "stale", "aborted", "wrong_page"])
def test_inflight_pages_publish_only_after_matching_worker_writes(failed_gate):
    groups, pool = managers()
    state = DedicatedGDNCheckpoints(groups)
    pages = [SimpleNamespace(block_id=9), SimpleNamespace(block_id=13)]
    for group, page in zip(groups, pages, strict=True):
        group.dedicated_pages["req"] = (3456, page)
    req = SimpleNamespace(request_id="req", num_preemptions=0, num_stale_output_tokens=0, is_finished=lambda: False)
    requests = {"req": req}
    step = SimpleNamespace(num_scheduled_tokens={"req": 1452})
    state.dispatch(requests, step)
    writes = step.gdn_dedicated_checkpoints["req"]
    assert writes == ((1, 3456, 9), (2, 3456, 13))
    assert not any(group.dedicated_pages for group in groups)
    pool.free_blocks.assert_not_called()
    if failed_gate == "one_group":
        writes = writes[:1]
    elif failed_gate == "preempted":
        req.num_preemptions += 1
    elif failed_gate == "stale":
        req.num_stale_output_tokens = 1
    elif failed_gate == "aborted":
        requests.clear()
    elif failed_gate == "wrong_page":
        writes = writes[:1] + ((2, 3456, 99),)
    state.publish(requests, step, SimpleNamespace(req_ids=["req"], gdn_dedicated_checkpoint_writes={"req": writes}))
    assert pool.cache_partial_block.call_count == (2 if failed_gate is None else 0)
    assert [call.args[0] for call in pool.free_blocks.call_args_list] == [[pages[0]], [pages[1]]]
    assert not state.inflight


def test_forced_reset_does_not_reclaim_unfinished_checkpoint_write(monkeypatch):
    groups, _ = managers()
    pending = DedicatedGDNCheckpoints(groups)
    pending.inflight[42] = [object()]
    scheduler = SimpleNamespace(_ascend_dedicated_checkpoints=pending, _ascend_seen_mamba_tails=None)
    reset = Mock(return_value=True)
    monkeypatch.setattr(split_patch, "_original_reset_prefix_cache", reset)
    assert not split_patch._reset_prefix_cache(scheduler, reset_running_requests=True)
    reset.assert_not_called()
    pending.inflight.clear()
    assert split_patch._reset_prefix_cache(scheduler, reset_running_requests=True)
    reset.assert_called_once_with(scheduler, reset_running_requests=True)


def test_worker_requires_all_layers_in_every_group_to_write():
    writes = {"req": ((1, 3456, 9), (2, 3456, 13))}
    group_a = GDNCheckpointAcknowledgement({"req": (writes["req"][0],)}, {"layer0"})
    group_b = GDNCheckpointAcknowledgement({"req": (writes["req"][1],)}, {"layer2"})
    a, b = (SimpleNamespace(checkpoint=SimpleNamespace(dedicated_ack=ack)) for ack in (group_a, group_b))
    metadata = {"layer0": a, "layer1": a, "layer2": b}
    groups = [None, SimpleNamespace(layer_names=["layer0", "layer1"]), SimpleNamespace(layer_names=["layer2"])]
    assert confirmed_writes(writes, metadata, groups) == {}
    group_a.layers.add("layer1")
    assert confirmed_writes(writes, metadata, groups) == writes


def test_real_hybrid_cache_publishes_only_the_acknowledged_branch():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    init_none_hash(sha256)
    hasher = get_request_block_hasher(128, sha256, num_hash_lookahead_tokens=1)
    full_spec = FullAttentionSpec(block_size=2048, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    mamba_spec = MambaSpec(
        block_size=2048,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=64,
    )
    cache = KVCacheManager(
        KVCacheConfig(
            num_blocks=128,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["full"], full_spec)]
            + [KVCacheGroupSpec([f"gdn{i}"], mamba_spec) for i in range(3)],
        ),
        max_model_len=8192,
        scheduler_block_size=2048,
        hash_block_size=128,
        num_hash_lookahead_tokens=1,
    )
    all_managers = cache.coordinator.single_type_managers
    gdn_managers = tuple(all_managers[1:])
    assert all(isinstance(manager, AscendMambaManager) for manager in gdn_managers)
    state = DedicatedGDNCheckpoints(gdn_managers)
    policy = GDNCheckpointPolicy(8192, True, False)
    for manager in gdn_managers:
        manager.gdn_checkpoint_policy = policy
    scheduler = SimpleNamespace(
        _ascend_dedicated_checkpoints=state,
        _ascend_gdn_checkpoint_policy=policy,
        kv_cache_manager=cache,
        cache_config=SimpleNamespace(block_size=2048),
        hash_block_size=128,
        mamba_shared_prefix_checkpoint=False,
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=1,
        num_lookahead_tokens=1,
    )
    prefix = [11] * 2275

    def request(name, ids):
        return Request(name, ids, SamplingParams(max_tokens=32), None, block_hasher=hasher)

    seed = request("seed", prefix)
    assert cache.allocate_slots(seed, 2048) is not None
    cache.free(seed)
    cache.coordinator.new_step_starts()

    prompts = [prefix + [token] * 1225 for token in (101, 202)]
    live = {name: request(name, prompt) for name, prompt in zip(("a", "b"), prompts, strict=True)}
    step = SchedulerOutput.make_empty()
    for req in live.values():
        cached, hit, req.shared_prefix_boundary = cache.get_computed_blocks(req)
        assert hit == 2048
        count = split_patch._maybe_use_dedicated_checkpoint(scheduler, req, 1452, hit, 0, 1408)
        assert count == 1452
        assert cache.allocate_slots(req, count, hit, cached, num_lookahead_tokens=1) is not None
        step.num_scheduled_tokens[req.request_id] = count
    state.dispatch(live, step)
    transferred = deepcopy(step).gdn_dedicated_checkpoints
    ids = {page_id for writes in transferred.values() for _, _, page_id in writes}
    assert len(ids) == 6
    assert all(
        page_id not in {block.block_id for block in manager.req_to_blocks[req_id]}
        for manager in gdn_managers
        for req_id in live
        for page_id in ids
    )
    assert [cache.get_computed_blocks(request(f"before-{i}", prompt))[1] for i, prompt in enumerate(prompts)] == [
        2048,
        2048,
    ]

    output = ModelRunnerOutput(req_ids=list(live))
    # Only one branch receives an acknowledgement for the full set of writes.
    output.gdn_dedicated_checkpoint_writes = {"a": transferred["a"]}
    state.publish(live, step, deepcopy(output))
    assert all(cache.block_pool.blocks[page_id].ref_cnt == 0 for page_id in ids)
    cache.coordinator.new_step_starts()
    assert [cache.get_computed_blocks(request(f"after-{i}", prompt))[1] for i, prompt in enumerate(prompts)] == [
        3456,
        2048,
    ]
