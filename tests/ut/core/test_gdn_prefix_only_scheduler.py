# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import create_model_runner_output, create_vllm_config
from vllm_ascend.core.gdn_prefix_only_scheduler import (
    AsyncGDNPrefixOnlyScheduler,
    GDNPrefixOnlyScheduler,
    configure_gdn_prefix_only_scheduler,
)
from vllm_ascend.core.kv_cache_interface import AscendGDNPrefixOnlySpec, register_ascend_kv_cache_specs
from vllm_ascend.core.single_type_kv_cache_manager import AscendGDNPrefixOnlyManager

PAGE_SIZE = 2048
MATCH_UNIT = 128


def _make_scheduler(async_scheduling: bool) -> GDNPrefixOnlyScheduler:
    def finalize_config(**kwargs):
        kwargs["kv_transfer_config"] = None
        kwargs["model_config"].max_model_len = 8192
        kwargs["cache_config"].mamba_cache_mode = "align"
        kwargs["scheduler_config"].async_scheduling = async_scheduling
        return VllmConfig(**kwargs)

    with patch("tests.ut.kv_offload.utils.VllmConfig", new=finalize_config):
        config = create_vllm_config(block_size=PAGE_SIZE, max_num_batched_tokens=4096)
    register_all_kvcache_specs(config)
    register_ascend_kv_cache_specs()
    ascend_config = SimpleNamespace(
        gdn_internal_prefill_checkpoints=False,
        scheduler_config=SimpleNamespace(enable_balance_scheduling=False),
    )
    with patch.object(VllmConfig, "use_v2_model_runner", True):
        configure_gdn_prefix_only_scheduler(config, ascend_config)
    scheduler_cls = config.scheduler_config.get_scheduler_cls()
    assert scheduler_cls is (AsyncGDNPrefixOnlyScheduler if async_scheduling else GDNPrefixOnlyScheduler)

    state = MambaSpec(
        block_size=PAGE_SIZE,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    cache = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(block_size=PAGE_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32),
            ),
            KVCacheGroupSpec(["gdn"], AscendGDNPrefixOnlySpec.from_mamba_spec(state)),
        ],
    )
    init_none_hash(sha256)
    return scheduler_cls(config, cache, StructuredOutputManager(config), PAGE_SIZE, hash_block_size=MATCH_UNIT)


def _request(name: str, prompt: list[int]) -> Request:
    return Request(
        request_id=name,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(MATCH_UNIT, sha256),
    )


def _finish_cold_prefill(scheduler: GDNPrefixOnlyScheduler, request: Request) -> list[int]:
    scheduler.add_request(request)
    stops = []
    while request.num_computed_tokens < request.num_prompt_tokens:
        output = scheduler.schedule()
        assert output.num_scheduled_tokens[request.request_id] > 0
        stops.append(request.num_computed_tokens)
        feedback = create_model_runner_output([request])
        if request.num_computed_tokens < request.num_prompt_tokens:
            feedback.sampled_token_ids = [[]]
        scheduler.update_from_output(output, feedback)
    return stops


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    ("prefix_len", "prompt_len", "cold_stops"),
    [(2287, 3512, [2048, 2176, 2287]), (1012, 1211, [896, 1012])],
)
def test_gdn_prefix_only_preserves_shared_state_and_runs_warm_suffixes_in_one_step(
    async_scheduling, prefix_len, prompt_len, cold_stops
):
    scheduler = _make_scheduler(async_scheduling)
    cache = scheduler.kv_cache_manager
    manager = cache.coordinator.single_type_managers[1]
    assert isinstance(manager, AscendGDNPrefixOnlyManager)
    prefix = [1] * prefix_len
    assert _finish_cold_prefill(scheduler, _request("cold", prefix)) == cold_stops

    prompts = [prefix + [token] * (prompt_len - prefix_len) for token in (2, 3)]
    warm_requests = [_request(f"warm-{index}", prompt) for index, prompt in enumerate(prompts)]
    hits, hit_len, _ = cache.get_computed_blocks(warm_requests[0])
    assert hit_len == cold_stops[-2]
    source = hits.blocks[1][-1]
    for request in warm_requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {request.request_id: prompt_len - hit_len for request in warm_requests}
    cold_state_id = source.block_id
    working_ids = {cache.get_block_ids(request.request_id)[1][-1] for request in warm_requests}
    assert len(working_ids) == 2
    assert cold_state_id not in working_ids
    assert output.kv_cache_block_copies
    assert {
        copy.dst_block_id for copy in output.kv_cache_block_copies if copy.src_block_id == cold_state_id
    } == working_ids
    assert all(manager.has_local_prefix_hit(request.request_id) for request in warm_requests)
    scheduler.update_from_output(output, create_model_runner_output(warm_requests))
    assert all(not manager.has_local_prefix_hit(request.request_id) for request in warm_requests)

    replay = _request("replay", prompts[0])
    hits, replay_hit_len, _ = cache.get_computed_blocks(replay)
    assert replay_hit_len == hit_len
    assert hits.blocks[1][-1] is source


def test_gdn_prefix_only_preserves_physical_page_and_budget_splits():
    scheduler = _make_scheduler(False)
    hit = 2176
    request = _request("cross-page", [1] * 4352)
    assert scheduler._mamba_block_aligned_split(request, 4352 - hit, hit) == PAGE_SIZE * 2 - hit

    request = _request("budget-limited", [1] * 3512)
    scheduler.max_num_scheduled_tokens = 256
    assert scheduler._mamba_block_aligned_split(request, 256, hit) == 256


def test_gdn_prefix_only_reclassifies_request_after_preemption():
    scheduler = _make_scheduler(False)
    cache = scheduler.kv_cache_manager
    manager = cache.coordinator.single_type_managers[1]
    assert isinstance(manager, AscendGDNPrefixOnlyManager)
    prefix = [1] * 2287
    _finish_cold_prefill(scheduler, _request("cold", prefix))

    request = _request("warm", prefix + [2] * (3512 - len(prefix)))
    scheduler.add_request(request)
    original_budget = scheduler.max_num_scheduled_tokens
    scheduler.max_num_scheduled_tokens = 256
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == 256
    assert manager.has_local_prefix_hit(request.request_id)
    feedback = create_model_runner_output([request])
    feedback.sampled_token_ids = [[]]
    scheduler.update_from_output(output, feedback)

    remaining = request.num_prompt_tokens - request.num_computed_tokens
    scheduler.max_num_scheduled_tokens = original_budget
    assert scheduler._mamba_block_aligned_split(request, remaining) == remaining

    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.monotonic())
    assert not manager.has_local_prefix_hit(request.request_id)
    assert cache.reset_prefix_cache()
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == PAGE_SIZE
    assert not manager.has_local_prefix_hit(request.request_id)
