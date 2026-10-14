# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Admission regressions using real schedulers, KV managers and block pools.

Only the remote transport and Ascend configuration boundary are replaced. In
particular, an allocation failure must come from the real capacity check, not a
mocked ``allocate_slots`` result. These tests run on both upstream CI pins.
"""

from contextlib import ExitStack
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import create_model_runner_output, create_request, create_vllm_config
from vllm_ascend.ascend_config import BatchJobSchedConfig
from vllm_ascend.core.batch_job_aware_scheduler import BatchJobAwareAsyncScheduler, BatchJobAwareScheduler
from vllm_ascend.core.dyntra_lb_scheduler import AsyncDyntraLBScheduler, DyntraLBScheduler
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec, register_ascend_kv_cache_specs
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeScheduler,
)
from vllm_ascend.core.scheduler import AscendAsyncScheduler, AscendScheduler
from vllm_ascend.core.scheduler_kv_reservation import spec_step_blocks
from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkAsyncScheduler, ProfilingChunkScheduler
from vllm_ascend.core.short_request_first_scheduler import (
    ShortRequestFirstAsyncScheduler,
    ShortRequestFirstRequestQueue,
)
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler
from vllm_ascend.patch.platform.patch_swa_bounded_replay_legacy import apply_legacy_replay_patch
from vllm_ascend.utils import vllm_version_is

# Hashes must resolve the smaller SWA pages, independently of the scheduler's
# 64-token allocation unit. Equal page bytes let both groups share one pool.
_HASH_BLOCK_SIZE = 16
_REPLAY_WINDOW = 32


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

    def wait_for_layer_load(self, *args, **kwargs):
        raise AssertionError("A scheduler-only test must not execute the worker")

    def save_kv_layer(self, *args, **kwargs):
        raise AssertionError("A scheduler-only test must not execute the worker")

    def wait_for_save(self):
        raise AssertionError("A scheduler-only test must not execute the worker")


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
    short_request_first_enabled=False,
):
    """Build all policy variants with the same real allocation configuration."""
    config = create_vllm_config(
        max_num_seqs=2,
        block_size=block_size,
        speculative_method=speculative_method if num_speculative_tokens else None,
        num_speculative_tokens=num_speculative_tokens,
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
            short_request_first_config=SimpleNamespace(
                enabled=short_request_first_enabled, threshold=256, long_max_wait_ms=0
            ),
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
    pytest.param(AscendScheduler, False, id="default"),
    pytest.param(AscendAsyncScheduler, False, id="default-async"),
    pytest.param(BalanceScheduler, False, id="balance-disabled"),
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


def _assert_async_load_pressure(scheduler):
    """The first remote request can run after the second was refused admission."""
    req_a, req_b = _add_remote_requests(scheduler)
    output = scheduler.schedule()
    assert not output.num_scheduled_tokens
    assert req_a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req_b.status == RequestStatus.WAITING
    _assert_no_blocks(scheduler, req_b)
    assert scheduler.connector.allocated == {req_a.request_id}

    scheduler.update_from_output(output, create_model_runner_output([], finished_recving={req_a.request_id}))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[req_a.request_id] > 0
    assert req_a in scheduler.running
    assert not output.preempted_req_ids
    assert req_b.status == RequestStatus.WAITING
    _assert_no_blocks(scheduler, req_b)

    scheduler.update_from_output(output, create_model_runner_output([req_a]))
    scheduler.finish_requests([req_a.request_id, req_b.request_id], RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()
    _assert_pool_released(scheduler)


@pytest.mark.parametrize("scheduler_cls,balance_enabled", _SCHEDULERS)
@pytest.mark.parametrize(
    "block_size,swa_enabled,num_speculative_tokens,num_blocks,expected_margin",
    [(16, False, 3, 8, 1), (16, True, 3, 17, 2), (64, True, 31, 38, 3)],
    ids=["full", "full-swa", "mixed-page-sizes"],
)
def test_async_load_preserves_decode_capacity(
    scheduler_cls, balance_enabled, block_size, swa_enabled, num_speculative_tokens, num_blocks, expected_margin
):
    scheduler = _make_scheduler(
        scheduler_cls,
        balance_enabled=balance_enabled,
        block_size=block_size,
        swa_enabled=swa_enabled,
        num_speculative_tokens=num_speculative_tokens,
        num_blocks=num_blocks,
    )
    assert spec_step_blocks(scheduler) == expected_margin
    _assert_async_load_pressure(scheduler)


@pytest.mark.parametrize("scheduler_cls,balance_enabled", _SCHEDULERS)
def test_async_load_reservation_includes_positive_lookahead(scheduler_cls, balance_enabled):
    # Balance's enabled schedule does not pad the first decode query. MTP still
    # needs draft lookahead, so this case exercises its actual decode pressure.
    scheduler = _make_scheduler(scheduler_cls, balance_enabled=balance_enabled, speculative_method="mtp")
    assert scheduler.num_lookahead_tokens == 31
    assert spec_step_blocks(scheduler) == 5  # ceil(63 / 64) + ceil(63 / 16)
    _assert_async_load_pressure(scheduler)


@pytest.mark.parametrize("scheduler_cls,balance_enabled", _SCHEDULERS)
@pytest.mark.parametrize("num_speculative_tokens,num_blocks", [(31, 42), (0, 36)], ids=["exact-fit", "no-spec"])
def test_async_admission_does_not_reserve_margin_twice(
    scheduler_cls, balance_enabled, num_speculative_tokens, num_blocks
):
    # Prompts occupy 20 + 15 blocks. The speculative margin is 3 per request;
    # with speculation disabled, the prompts fit exactly in the usable pool.
    scheduler = _make_scheduler(
        scheduler_cls,
        balance_enabled=balance_enabled,
        num_speculative_tokens=num_speculative_tokens,
        num_blocks=num_blocks,
    )
    requests = _add_remote_requests(scheduler)
    output = scheduler.schedule()
    assert not output.num_scheduled_tokens
    assert all(request.status == RequestStatus.WAITING_FOR_REMOTE_KVS for request in requests)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == num_blocks - 1 - 35
    assert sum(scheduler._request_remaining_blocks(request) for request in scheduler._inflight_prefills) == (
        6 if num_speculative_tokens else 0
    )

    # Cancellation may not free blocks that a remote writer still owns. Once
    # the completion arrives, every group must release those blocks.
    request_ids = {request.request_id for request in requests}
    scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == num_blocks - 1 - 35
    scheduler.update_from_output(output, create_model_runner_output([], finished_recving=request_ids))
    scheduler.schedule()
    _assert_pool_released(scheduler)


@pytest.mark.parametrize("scheduler_cls", _PREFETCH_SCHEDULERS)
def test_dyntra_prefetch_preserves_decode_capacity(scheduler_cls):
    scheduler = _make_scheduler(scheduler_cls)
    scheduler._lb_kv_prefetch_enabled = True
    req_a, req_b = _add_remote_requests(scheduler)
    not_ready = scheduler._run_lb_kv_prefetch()
    assert req_a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req_b.status == RequestStatus.WAITING
    assert req_b.request_id in not_ready
    assert scheduler.connector.allocated == {req_a.request_id}
    _assert_no_blocks(scheduler, req_b)
    assert sum(scheduler._request_remaining_blocks(request) for request in scheduler._inflight_prefills) == 3

    output = scheduler.schedule()
    scheduler.update_from_output(output, create_model_runner_output([], finished_recving={req_a.request_id}))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[req_a.request_id] > 0
    assert not output.preempted_req_ids
    scheduler.update_from_output(output, create_model_runner_output([req_a]))
    scheduler.finish_requests([req_a.request_id, req_b.request_id], RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()
    _assert_pool_released(scheduler)


@pytest.mark.parametrize("scheduler_cls", _PREFETCH_SCHEDULERS)
def test_dyntra_prefetch_keeps_non_speculative_inflight_reservation(scheduler_cls):
    scheduler = _make_scheduler(scheduler_cls, block_size=16, swa_enabled=False, num_speculative_tokens=0, num_blocks=8)
    scheduler._lb_kv_prefetch_enabled = True
    req_a = create_request(request_id=1, num_tokens=64, block_size=16, do_remote_prefill=True)
    req_b = create_request(request_id=2, num_tokens=80, block_size=16, do_remote_prefill=True)
    scheduler.connector.matched_tokens = {req_a.request_id: 16, req_b.request_id: 64}
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    scheduler._run_lb_kv_prefetch()
    assert spec_step_blocks(scheduler) == 0
    assert req_a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert sum(scheduler._request_remaining_blocks(request) for request in scheduler._inflight_prefills) == 3
    assert req_b.status == RequestStatus.WAITING
    _assert_no_blocks(scheduler, req_b)


@pytest.mark.parametrize("scheduler_cls,balance_enabled", _SCHEDULERS)
@pytest.mark.parametrize("num_blocks,admitted", [(23, False), (24, True)], ids=["one-short", "exact-fit"])
def test_first_async_admission_reserves_its_own_margin(scheduler_cls, balance_enabled, num_blocks, admitted):
    # There are no other in-flight requests: this tests the current admission's
    # own margin independently of _request_remaining_blocks accounting.
    scheduler = _make_scheduler(scheduler_cls, balance_enabled=balance_enabled, num_blocks=num_blocks)
    _assert_first_async_admission(scheduler, admitted)


def _assert_first_async_admission(scheduler, admitted):
    pool = scheduler.kv_cache_manager.block_pool
    request = create_request(request_id=1, num_tokens=256, block_size=16, do_remote_prefill=True)
    scheduler.add_request(request)
    scheduler.schedule()
    if admitted:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert pool.get_num_free_blocks() == 3
    else:
        assert request.status == RequestStatus.WAITING
        _assert_no_blocks(scheduler, request)
        assert not scheduler.connector.allocated
        assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1


@pytest.mark.parametrize("scheduler_cls", _PREFETCH_SCHEDULERS)
@pytest.mark.parametrize("num_blocks,admitted", [(23, False), (24, True)], ids=["one-short", "exact-fit"])
def test_first_prefetch_reserves_its_own_margin(scheduler_cls, num_blocks, admitted):
    scheduler = _make_scheduler(scheduler_cls, num_blocks=num_blocks)
    scheduler._lb_kv_prefetch_enabled = True
    request = create_request(request_id=1, num_tokens=256, block_size=16, do_remote_prefill=True)
    scheduler.add_request(request)
    not_ready = scheduler._run_lb_kv_prefetch()
    if admitted:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 3
    else:
        assert request.status == RequestStatus.WAITING
        assert request.request_id in not_ready
        _assert_no_blocks(scheduler, request)
        assert not scheduler.connector.allocated


@pytest.mark.parametrize("scheduler_cls,balance_enabled", _SCHEDULERS)
@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
def test_mamba_speculative_states_are_not_reserved_twice(scheduler_cls, balance_enabled, mamba_cache_mode):
    # Each request owns four Mamba state blocks, including three speculative
    # states. Attention requires just one extra block per decode step. Adding a
    # second Mamba margin would incorrectly refuse B in this exact-fit pool.
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
    output = scheduler.schedule()
    assert spec_step_blocks(scheduler) == 1
    assert all(request.status == RequestStatus.WAITING_FOR_REMOTE_KVS for request in requests)
    assert all(scheduler._request_remaining_blocks(request) == 1 for request in requests)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 2
    for request in requests:
        mamba_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[1].req_to_blocks[request.request_id]
        assert sum(not block.is_null for block in mamba_blocks) == 4

    request_ids = {request.request_id for request in requests}
    scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(output, create_model_runner_output([], finished_recving=request_ids))
    scheduler.schedule()
    _assert_pool_released(scheduler)


@pytest.mark.parametrize("scheduler_cls,balance_enabled", _SCHEDULERS)
def test_sync_kv_load_does_not_reserve_async_decode_margin(scheduler_cls, balance_enabled):
    # The 192-token remote prefix and 64 local tokens consume exactly 20
    # blocks. Reserving the async load's three-block margin would wrongly deny
    # this synchronous load, whose request can run and be preempted normally.
    scheduler = _make_scheduler(scheduler_cls, balance_enabled=balance_enabled, num_blocks=21)
    scheduler.connector.load_async = False
    request = create_request(request_id=1, num_tokens=256, block_size=16, do_remote_prefill=True)
    scheduler.connector.matched_tokens[request.request_id] = 192
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert spec_step_blocks(scheduler) == 3
    assert output.num_scheduled_tokens[request.request_id] == 64
    assert request.status == RequestStatus.RUNNING
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    scheduler.update_from_output(output, create_model_runner_output([request]))
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()
    _assert_pool_released(scheduler)


@pytest.mark.parametrize("scheduler_cls", [AscendScheduler, ShortRequestFirstAsyncScheduler])
@pytest.mark.parametrize("num_blocks,admitted", [(23, False), (24, True)], ids=["one-short", "exact-fit"])
def test_short_request_first_retains_current_admission_margin(scheduler_cls, num_blocks, admitted):
    scheduler = _make_scheduler(scheduler_cls, num_blocks=num_blocks, short_request_first_enabled=True)
    assert isinstance(scheduler.waiting, ShortRequestFirstRequestQueue)
    _assert_first_async_admission(scheduler, admitted)


@pytest.mark.parametrize("scheduler_cls", [AscendScheduler, ShortRequestFirstAsyncScheduler])
def test_short_request_first_preserves_decode_progress(scheduler_cls):
    scheduler = _make_scheduler(scheduler_cls, short_request_first_enabled=True)
    assert isinstance(scheduler.waiting, ShortRequestFirstRequestQueue)
    _assert_async_load_pressure(scheduler)


@pytest.fixture
def swa_replay_specs():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    apply_legacy_replay_patch(SimpleNamespace(cache_config=SimpleNamespace(swa_bounded_replay=True)))


def _make_async_load_replay_scheduler(scheduler_cls, num_blocks):
    return _make_scheduler(
        scheduler_cls,
        block_size=_HASH_BLOCK_SIZE,
        num_blocks=num_blocks,
        num_speculative_tokens=3,
        speculative_method="mtp",
        bounded_replay=True,
        replay_window=_REPLAY_WINDOW,
    )


@pytest.mark.skipif(vllm_version_is("0.30.0"), reason="The release replay backport supports local caching only")
@pytest.mark.parametrize("scheduler_cls", [AscendScheduler, AscendAsyncScheduler])
@pytest.mark.parametrize(
    "num_blocks,num_admitted",
    [(8, 0), (9, 1), (15, 1), (16, 2)],
    ids=["first-short", "first-fit", "both-short", "both-fit"],
)
def test_async_load_with_real_swa_replay_reserves_once(swa_replay_specs, scheduler_cls, num_blocks, num_admitted):
    scheduler = _make_async_load_replay_scheduler(scheduler_cls, num_blocks)
    assert scheduler.prefix_replay_tokens == _REPLAY_WINDOW
    assert scheduler.num_spec_tokens == scheduler.num_lookahead_tokens == 3
    assert spec_step_blocks(scheduler) == 2
    # The first prompt owns 4 full + 2 live SWA blocks; the second owns 3 + 2.
    # Each needs two more blocks for its first MTP step. Include the null block:
    # 9 is the first exact fit, 16 is the two-request exact fit.
    requests = _add_remote_requests(scheduler)
    output = scheduler.schedule()
    assert not output.num_scheduled_tokens
    admitted_ids = {request.request_id for request in requests[:num_admitted]}
    assert scheduler.connector.allocated == admitted_ids
    for request in requests[:num_admitted]:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    for request in requests[num_admitted:]:
        assert request.status == RequestStatus.WAITING
        _assert_no_blocks(scheduler, request)
    allocated_blocks = sum((6, 5)[:num_admitted])
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == num_blocks - 1 - allocated_blocks

    # An aborted transfer still owns its blocks until the receiver completes.
    # SWA's null prefix must not hide a leak in either group's live references.
    scheduler.finish_requests([request.request_id for request in requests], RequestStatus.FINISHED_ABORTED)
    assert pool.get_num_free_blocks() == num_blocks - 1 - allocated_blocks
    scheduler.update_from_output(output, create_model_runner_output([], finished_recving=admitted_ids))
    scheduler.schedule()
    _assert_pool_released(scheduler)


@pytest.mark.skipif(vllm_version_is("0.30.0"), reason="The release replay backport supports local caching only")
@pytest.mark.parametrize("scheduler_cls", [AscendScheduler, AscendAsyncScheduler])
def test_async_load_replays_and_preserves_resumed_payload(swa_replay_specs, scheduler_cls):
    scheduler = _make_async_load_replay_scheduler(scheduler_cls, num_blocks=9)
    assert scheduler.prefix_replay_tokens == _REPLAY_WINDOW
    request, waiting = _add_remote_requests(scheduler)
    output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert waiting.status == RequestStatus.WAITING
    manager = scheduler.kv_cache_manager
    full_prefix_blocks = manager.get_blocks(request.request_id).blocks[0][:]

    scheduler.update_from_output(output, create_model_runner_output([], finished_recving={request.request_id}))
    output = scheduler.schedule()
    assert not output.preempted_req_ids
    assert output.num_scheduled_tokens == {request.request_id: _REPLAY_WINDOW}
    data = output.scheduled_new_reqs[0]
    assert data.num_computed_tokens == data.replay_start == _REPLAY_WINDOW
    assert request.replay_start == _REPLAY_WINDOW
    full_blocks, swa_blocks = manager.get_blocks(request.request_id).blocks
    assert full_blocks[: len(full_prefix_blocks)] == full_prefix_blocks
    assert all(block.ref_cnt == 1 for block in full_prefix_blocks)
    assert all(block.is_null for block in swa_blocks[: _REPLAY_WINDOW // _HASH_BLOCK_SIZE])
    assert all(not block.is_null for block in swa_blocks[_REPLAY_WINDOW // _HASH_BLOCK_SIZE :])
    assert request.num_output_placeholders == int(issubclass(scheduler_cls, AsyncScheduler))
    # The reserved blocks suffice for the replay and MTP lookahead exactly.
    assert manager.block_pool.get_num_free_blocks() == 0
    scheduler.update_from_output(output, create_model_runner_output([request]))
    assert request.num_in_flight_tokens == request.num_output_placeholders == 0

    # Drain before preemption, then let the loaded full-attention prefix serve
    # the resumed request locally. The V1 SWA wrapper must still supply the
    # fresh replay boundary through CachedRequestData on the inherited path.
    scheduler.finish_requests(waiting.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.connector.matched_tokens[request.request_id] = 0
    scheduler.running.remove(request)
    scheduler._preempt_request(request, monotonic())
    resumed = scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert resumed.preempted_req_ids == {request.request_id}
    assert request.num_preemptions == 1
    assert resumed.num_scheduled_tokens == {request.request_id: _REPLAY_WINDOW + 1}
    assert not scheduler.use_v2_model_runner
    assert resumed.scheduled_cached_reqs.replay_start == {request.request_id: _REPLAY_WINDOW}
    assert manager.get_blocks(request.request_id).blocks[0][: len(full_prefix_blocks)] == full_prefix_blocks
    scheduler.update_from_output(resumed, create_model_runner_output([request]))
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()
    _assert_pool_released(scheduler)


@pytest.mark.skipif(not vllm_version_is("0.30.0"), reason="Main supports bounded replay with a KV connector")
@pytest.mark.parametrize("scheduler_cls", [AscendScheduler, AscendAsyncScheduler])
def test_legacy_replay_preserves_connector_rejection(swa_replay_specs, scheduler_cls):
    with pytest.raises(ValueError, match="local prefix caching only"):
        _make_async_load_replay_scheduler(scheduler_cls, num_blocks=9)
