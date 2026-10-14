# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Admission regressions using real schedulers, KV managers and block pools.

Only the remote transport and Ascend configuration boundary are replaced. In
particular, an allocation failure must come from the real capacity check, not a
mocked ``allocate_slots`` result. These tests run on both upstream CI pins.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import SpeculativeConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.model_executor.models import ModelRegistry
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import (
    _fake_opt_model_info,
    create_model_runner_output,
    create_request,
    create_vllm_config,
)
from vllm_ascend.ascend_config import BatchJobSchedConfig
from vllm_ascend.core.batch_job_aware_scheduler import BatchJobAwareAsyncScheduler, BatchJobAwareScheduler
from vllm_ascend.core.dyntra_lb_scheduler import AsyncDyntraLBScheduler, DyntraLBScheduler
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeScheduler,
)
from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkAsyncScheduler, ProfilingChunkScheduler
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler

# Hashes must resolve the smaller SWA pages, independently of the scheduler's
# 64-token allocation unit. Equal page bytes let both groups share one pool.
_HASH_BLOCK_SIZE = 16


class _AsyncKVConnector(KVConnectorBase_V1, SupportsHMA):
    """A deterministic transport that reports remote hits without doing I/O."""

    def __init__(self, config, kv_cache_config):
        super().__init__(config, KVConnectorRole.SCHEDULER, kv_cache_config)
        self.matched_tokens: dict[str, int] = {}
        self.allocated: set[str] = set()
        self.load_async = True

    @property
    def requires_kv_delivery(self):
        return False

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        remote_end = self.matched_tokens.get(request.request_id, request.num_prompt_tokens)
        return max(remote_end - num_computed_tokens, 0), self.load_async

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            self.allocated.add(request.request_id)

    def build_connector_meta(self, scheduler_output):
        return KVConnectorMetadata()

    def request_finished_all_groups(self, request, block_ids):
        return False, None

    def start_load_kv(self, *args, **kwargs):
        raise AssertionError("A scheduler-only test must not execute the worker")

    wait_for_layer_load = save_kv_layer = wait_for_save = start_load_kv


def _make_scheduler(
    scheduler_cls,
    *,
    block_size=64,
    num_blocks=38,
    num_speculative_tokens=31,
    speculative_method="ngram",
    balance_enabled=False,
    swa_enabled=True,
    bounded_replay=False,
    replay_window=0,
    mamba_cache_mode=None,
):
    """Build all policy variants with the same real allocation configuration."""
    config = create_vllm_config(
        max_num_seqs=2,
        block_size=block_size,
        speculative_method=speculative_method
        if num_speculative_tokens and speculative_method != "draft_model"
        else None,
        num_speculative_tokens=num_speculative_tokens,
    )
    if num_speculative_tokens and speculative_method == "draft_model":
        model_info = _fake_opt_model_info()
        with patch.object(ModelRegistry, "inspect_model_cls", return_value=(model_info, model_info.architecture)):
            config.speculative_config = SpeculativeConfig(
                method="draft_model",
                model=config.model_config.model,
                num_speculative_tokens=num_speculative_tokens,
                target_model_config=config.model_config,
                target_parallel_config=config.parallel_config,
            )
    config.scheduler_config.async_scheduling = issubclass(scheduler_cls, AsyncScheduler)
    config.cache_config.num_gpu_blocks = num_blocks
    config.cache_config.swa_bounded_replay = bounded_replay
    if balance_enabled:
        config.parallel_config.data_parallel_size = 2
    full_spec = FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float16)
    groups = [KVCacheGroupSpec(["full"], full_spec)]
    if swa_enabled:
        swa_spec_cls = AscendSlidingWindowMLASpec if replay_window else SlidingWindowSpec
        swa_kwargs = {"bounded_replay": True} if replay_window else {}
        groups.append(
            KVCacheGroupSpec(
                ["swa"],
                swa_spec_cls(
                    block_size=_HASH_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                    sliding_window=replay_window or 4096,
                    page_size_padded=full_spec.page_size_bytes,
                    **swa_kwargs,
                ),
            )
        )
    if mamba_cache_mode is not None:
        config.cache_config.mamba_cache_mode = mamba_cache_mode
        groups.append(
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=128 if mamba_cache_mode == "none" else 16,
                    shapes=((1,),),
                    dtypes=(torch.float16,),
                    page_size_padded=full_spec.page_size_bytes,
                    mamba_cache_mode=mamba_cache_mode,
                    num_speculative_blocks=num_speculative_tokens,
                ),
            )
        )
    kv_cache_config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)
    connector = _AsyncKVConnector(config, kv_cache_config)
    ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_balance_scheduling=balance_enabled,
            short_request_first_config=SimpleNamespace(enabled=False, threshold=256, long_max_wait_ms=0),
            batch_job_sched_config=BatchJobSchedConfig(enabled=issubclass(scheduler_cls, BatchJobAwareScheduler)),
            profiling_chunk_config=SimpleNamespace(
                enabled=True, smooth_factor=0.8, min_chunk=16, max_fit_chunk=30, need_timing=False
            ),
        ),
        finegrained_tp_config=SimpleNamespace(oproj_tensor_parallel_size=1, mlp_tensor_parallel_size=1),
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(KVConnectorFactory, "create_connector", return_value=connector))
        for target in (
            "vllm_ascend.patch.platform.patch_balance_schedule.init_ascend_config",
            "vllm_ascend.ascend_config.init_ascend_config",
            "vllm_ascend.ascend_config.get_ascend_config",
        ):
            stack.enter_context(patch(target, return_value=ascend_config))
        return scheduler_cls(
            vllm_config=config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(config),
            block_size=block_size,
            hash_block_size=_HASH_BLOCK_SIZE,
            log_stats=False,
        )


_SCHEDULERS = [
    pytest.param(Scheduler, False, id="default"),
    pytest.param(AsyncScheduler, False, id="default-async"),
    pytest.param(BalanceScheduler, True, id="balance-enabled"),
    pytest.param(DyntraLBScheduler, False, id="dyntra"),
    pytest.param(AsyncDyntraLBScheduler, False, id="dyntra-async"),
    pytest.param(RecomputeScheduler, False, id="recompute"),
    pytest.param(AsyncRecomputeScheduler, False, id="recompute-async"),
    pytest.param(ProfilingChunkScheduler, False, id="profiling"),
    pytest.param(ProfilingChunkAsyncScheduler, False, id="profiling-async"),
    pytest.param(DyntraLBRecomputeScheduler, False, id="dyntra-recompute"),
    pytest.param(AsyncDyntraLBRecomputeScheduler, False, id="dyntra-recompute-async"),
    pytest.param(BatchJobAwareScheduler, False, id="batch-job-aware"),
    pytest.param(BatchJobAwareAsyncScheduler, False, id="batch-job-aware-async"),
]
_PREFETCH_SCHEDULERS = [
    DyntraLBScheduler,
    AsyncDyntraLBScheduler,
    DyntraLBRecomputeScheduler,
    AsyncDyntraLBRecomputeScheduler,
]


def _add_remote_requests(scheduler):
    requests = [
        create_request(
            request_id=i,
            num_tokens=blocks * scheduler.block_size,
            block_size=_HASH_BLOCK_SIZE,
            do_remote_prefill=True,
        )
        for i, blocks in ((1, 4), (2, 3))
    ]
    for request in requests:
        scheduler.add_request(request)
    return requests


def _assert_no_blocks(scheduler, request):
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
        assert not manager.req_to_blocks.get(request.request_id)


def _assert_pool_released(scheduler):
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    assert all(block.is_null or block.ref_cnt == 0 for block in pool.blocks)
    assert not scheduler._inflight_prefills
    assert not scheduler.requests


@pytest.mark.parametrize("scheduler_cls,balance_enabled", _SCHEDULERS)
@pytest.mark.parametrize(
    "num_blocks,num_speculative_tokens,num_requests,admitted,speculative_method",
    [
        (23, 31, 1, 0, "ngram"),
        (24, 31, 1, 1, "ngram"),
        (41, 31, 2, 1, "ngram"),
        (42, 31, 2, 2, "ngram"),
        (36, 0, 2, 2, "ngram"),
        (25, 31, 1, 0, "draft_model"),
        (26, 31, 1, 1, "draft_model"),
    ],
    ids=["first-short", "first-fit", "inflight-short", "inflight-fit", "no-spec", "draft-short", "draft-fit"],
)
def test_async_admission_boundary(
    scheduler_cls, balance_enabled, num_blocks, num_speculative_tokens, num_requests, admitted, speculative_method
):
    scheduler = _make_scheduler(
        scheduler_cls,
        balance_enabled=balance_enabled,
        num_blocks=num_blocks,
        num_speculative_tokens=num_speculative_tokens,
        speculative_method=speculative_method,
    )
    requests = _add_remote_requests(scheduler)
    if num_requests == 1:
        scheduler.finish_requests(requests.pop().request_id, RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    # Prompts own 20 + 15 blocks. Draft-model lookahead must be reserved,
    # not allocated during the remote load and then reserved a second time.
    margin = (5 if speculative_method == "draft_model" else 3) if num_speculative_tokens else 0
    if speculative_method == "draft_model":
        assert not scheduler.use_eagle
        assert scheduler.num_lookahead_tokens == num_speculative_tokens > 0
    assert scheduler._spec_decode_step_blocks() == margin
    assert not output.num_scheduled_tokens
    assert scheduler.connector.allocated == {r.request_id for r in requests[:admitted]}
    for request in requests[admitted:]:
        assert request.status == RequestStatus.WAITING
        _assert_no_blocks(scheduler, request)
    for request in requests[:admitted]:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert scheduler._request_remaining_blocks(request) == margin
    pool = scheduler.kv_cache_manager.block_pool
    expected_free = num_blocks - 1 - sum((20, 15)[:admitted])
    assert pool.get_num_free_blocks() == expected_free
    scheduler.finish_requests([r.request_id for r in requests], RequestStatus.FINISHED_ABORTED)
    assert pool.get_num_free_blocks() == expected_free  # Remote writers still own these blocks.
    scheduler.update_from_output(output, create_model_runner_output([], finished_recving=scheduler.connector.allocated))
    scheduler.schedule()
    _assert_pool_released(scheduler)


@pytest.mark.parametrize("scheduler_cls,balance_enabled", _SCHEDULERS)
@pytest.mark.parametrize("speculative_method,margin", [("ngram", 3), ("mtp", 5), ("draft_model", 5)])
def test_async_load_can_resume_decode(scheduler_cls, balance_enabled, speculative_method, margin):
    scheduler = _make_scheduler(scheduler_cls, balance_enabled=balance_enabled, speculative_method=speculative_method)
    request, waiting = _add_remote_requests(scheduler)
    output = scheduler.schedule()
    assert scheduler._spec_decode_step_blocks() == margin
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert waiting.status == RequestStatus.WAITING
    scheduler.update_from_output(output, create_model_runner_output([], finished_recving={request.request_id}))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] > 0
    assert not output.preempted_req_ids
    assert waiting.status == RequestStatus.WAITING
    _assert_no_blocks(scheduler, waiting)
    scheduler.update_from_output(output, create_model_runner_output([request]))
    scheduler.finish_requests([request.request_id, waiting.request_id], RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()
    _assert_pool_released(scheduler)


@pytest.mark.parametrize("scheduler_cls", _PREFETCH_SCHEDULERS)
@pytest.mark.parametrize("num_blocks,admitted", [(23, 0), (24, 1), (41, 1), (42, 2)])
def test_prefetch_admission_boundary(scheduler_cls, num_blocks, admitted):
    scheduler = _make_scheduler(scheduler_cls, num_blocks=num_blocks)
    scheduler._lb_kv_prefetch_enabled = True
    requests = _add_remote_requests(scheduler)
    if num_blocks < 25:
        scheduler.finish_requests(requests.pop().request_id, RequestStatus.FINISHED_ABORTED)
    not_ready = scheduler._run_lb_kv_prefetch()
    assert scheduler.connector.allocated == {r.request_id for r in requests[:admitted]}
    for request in requests[:admitted]:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert scheduler._request_remaining_blocks(request) == 3
    for request in requests[admitted:]:
        assert request.status == RequestStatus.WAITING
        assert request.request_id in not_ready
        _assert_no_blocks(scheduler, request)


@pytest.mark.parametrize("scheduler_cls", _PREFETCH_SCHEDULERS)
def test_prefetch_preserves_non_speculative_inflight_capacity(scheduler_cls):
    scheduler = _make_scheduler(scheduler_cls, block_size=16, swa_enabled=False, num_speculative_tokens=0, num_blocks=7)
    scheduler._lb_kv_prefetch_enabled = True
    request, waiting = _add_remote_requests(scheduler)
    scheduler.connector.matched_tokens = {request.request_id: 16, waiting.request_id: 48}
    scheduler._run_lb_kv_prefetch()
    assert scheduler._spec_decode_step_blocks() == 0
    assert scheduler._request_remaining_blocks(request) == 3
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert waiting.status == RequestStatus.WAITING
    _assert_no_blocks(scheduler, waiting)


@pytest.mark.parametrize("scheduler_cls,balance_enabled", _SCHEDULERS)
@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
def test_mamba_states_are_not_reserved_twice(scheduler_cls, balance_enabled, mamba_cache_mode):
    scheduler = _make_scheduler(
        scheduler_cls,
        balance_enabled=balance_enabled,
        block_size=16,
        swa_enabled=False,
        num_speculative_tokens=3,
        num_blocks=18,
        mamba_cache_mode=mamba_cache_mode,
    )
    requests = _add_remote_requests(scheduler)
    scheduler.schedule()
    assert scheduler._spec_decode_step_blocks() == 1
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 2
    for request in requests:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert scheduler._request_remaining_blocks(request) == 1
        blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[1].req_to_blocks[request.request_id]
        assert sum(not block.is_null for block in blocks) == 4


def test_inflight_tail_capacity_is_reused():
    scheduler = _make_scheduler(Scheduler, num_blocks=11)
    request = create_request(request_id=1, num_tokens=80, block_size=16, do_remote_prefill=True)
    scheduler.add_request(request)
    scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    # 80 -> 112 tokens fits the existing second full-attention block; only
    # SWA needs two more blocks. Adding the standalone margin would charge 3.
    assert scheduler._spec_decode_step_blocks() == 3
    assert scheduler._request_remaining_blocks(request) == 2


def test_balance_gate_precedes_kv_admission():
    scheduler = _make_scheduler(BalanceScheduler, balance_enabled=True, num_blocks=24)
    request = create_request(request_id=1, num_tokens=256, block_size=16, do_remote_prefill=True)
    scheduler.add_request(request)
    scheduler.balance_queue[0].fill_(scheduler.max_num_running_reqs)
    scheduler.schedule()
    assert request.status == RequestStatus.WAITING
    assert not scheduler.connector.allocated
    _assert_no_blocks(scheduler, request)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 23
    scheduler.balance_queue[0].zero_()
    scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 3
