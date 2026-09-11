# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import Request, RequestStatus

from tests.ut.core.test_dyntra_lb_scheduler import (
    create_dyntra_lb_scheduler,
    make_dyntra_test_config,
)
from tests.ut.kv_offload.utils import create_model_runner_output, create_request
from vllm_ascend.core.recompute_scheduler import (
    AsyncRecomputeScheduler,
    PreemptedRequestData,
    RecomputeReqInfo,
    RecomputeScheduler,
    RecomputeSchedulerConfig,
)
from vllm_ascend.utils import vllm_version_is


def _ratio_kwargs(ratio: int) -> dict[str, int]:
    """vLLM #51718 renamed compress_ratio to tokens_per_state on main."""
    return {"compress_ratio": ratio} if vllm_version_is("0.28.0") else {"tokens_per_state": ratio}


def _create_live_recompute_scheduler(*, async_scheduling: bool = False, max_num_seqs: int = 16):
    vllm_config = make_dyntra_test_config(max_num_seqs=max_num_seqs)
    vllm_config.scheduler_config.async_scheduling = async_scheduling
    scheduler_cls = AsyncRecomputeScheduler if async_scheduling else RecomputeScheduler
    return vllm_config, create_dyntra_lb_scheduler(vllm_config, scheduler_cls=scheduler_cls)


def _make_recompute_scheduler(*, scheduler_cls=RecomputeScheduler):
    vllm_config = make_dyntra_test_config()
    vllm_config.kv_transfer_config = None
    if scheduler_cls is AsyncRecomputeScheduler:
        vllm_config.scheduler_config.async_scheduling = True
    scheduler = create_dyntra_lb_scheduler(vllm_config, scheduler_cls=scheduler_cls)
    defaults = {
        "current_step": 0,
        "prefill_capacity_bound": False,
        "num_sampled_tokens_per_step": 1,
        "use_v2_model_runner": False,
        "needs_kv_cache_zeroing": False,
        "num_spec_tokens": 0,
        "dynamic_sd_lookup": None,
        "defer_block_free": False,
        "sched_step_seq": 0,
        "need_mamba_block_aligned_split": False,
        "use_eagle": False,
        "is_encoder_decoder": False,
        "is_encoder_only": False,
        "num_waiting_for_streaming_input": 0,
        "requires_kv_delivery": False,
        "enable_return_routed_experts": False,
        "recompute_kv_load_failures": True,
    }
    for name, value in defaults.items():
        if not hasattr(scheduler, name):
            setattr(scheduler, name, value)
    if not hasattr(scheduler, "_inflight_prefills"):
        scheduler._inflight_prefills = set()
    if not hasattr(scheduler, "prev_step_scheduled_req_ids"):
        scheduler.prev_step_scheduled_req_ids = set()
    return scheduler


def _fail_first_allocate(scheduler):
    original_allocate = scheduler.kv_cache_manager.allocate_slots
    fail_once = {"done": False}

    def allocate_slots(*args, **kwargs):
        if not fail_once["done"]:
            fail_once["done"] = True
            return None
        return original_allocate(*args, **kwargs)

    scheduler.kv_cache_manager.allocate_slots = allocate_slots


def _warmup_two_running(scheduler):
    block_size = scheduler.vllm_config.cache_config.block_size
    keep = create_request(request_id=1, block_size=block_size)
    victim = create_request(request_id=2, block_size=block_size)
    scheduler.add_request(keep)
    scheduler.add_request(victim)
    first_output = scheduler.schedule()
    scheduler.update_from_output(first_output, create_model_runner_output([keep, victim]))
    return keep, victim


def test_add_request_does_not_inject_placeholder_spec_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
    scheduler.log_stats = False
    scheduler.connector = None
    if not vllm_version_is("0.28.0"):
        # vllm main: Scheduler.add_request reads spec_decode_metrics_level.
        scheduler.spec_decode_metrics_level = "none"

    enqueued_requests = []

    def enqueue_waiting_request(self, request):
        enqueued_requests.append(request)

    scheduler._enqueue_waiting_request = MethodType(enqueue_waiting_request, scheduler)

    request = Request(
        request_id="pd-consumer-first-step",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )

    scheduler.add_request(request)

    assert enqueued_requests == [request]
    assert scheduler.requests[request.request_id] is request
    assert request.spec_token_ids == []
    assert request.num_tokens_with_spec == request.num_tokens


def test_recompute_notification_precedes_regular_output():
    scheduler_output = SimpleNamespace(
        recomputed_reqs=[
            RecomputeReqInfo(
                request_id="recomputed-request",
                output_token_ids=[],
                client_index=0,
            )
        ]
    )
    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

    RecomputeScheduler._add_recomputed_outputs(scheduler_output, outputs)
    outputs[0].append(
        EngineCoreOutput(
            request_id="regular-request",
            new_token_ids=[1],
        )
    )

    output = outputs[0][0]
    assert output.request_id == "recomputed-request"
    assert output.finish_reason == FinishReason.STOP
    assert output.stop_reason == "recomputed"
    assert outputs[0][1].request_id == "regular-request"


def test_finish_recomputed_request_uses_normal_abort_cleanup():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    request = Request(
        request_id="fallback-recomputed-request",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    request.status = RequestStatus.RUNNING

    # The fallback victim has already been popped from the running queue.
    scheduler.requests = {request.request_id: request}
    scheduler.running = []
    scheduler.waiting = MagicMock()
    scheduler.skipped_waiting = MagicMock()
    scheduler._inflight_prefills = {request}
    scheduler._connector_finished = MagicMock(return_value=(False, None))
    scheduler.encoder_cache_manager = MagicMock()
    scheduler.ec_connector = None
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._free_request_blocks = MagicMock()

    recomputed_reqs: list[RecomputeReqInfo] = []
    scheduler._finish_recomputed_request(request, recomputed_reqs)

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request not in scheduler._inflight_prefills
    assert request.request_id not in scheduler.requests
    assert request.request_id in scheduler.finished_req_ids
    scheduler._connector_finished.assert_called_once_with(request)
    scheduler.encoder_cache_manager.free.assert_called_once_with(request)
    scheduler._free_request_blocks.assert_called_once_with(request)
    assert recomputed_reqs == [
        RecomputeReqInfo(
            request_id=request.request_id,
            output_token_ids=request.output_token_ids,
            client_index=request.client_index,
        )
    ]


def test_truncate_computed_blocks_supports_legacy_short_mamba_group():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    mamba_block = MagicMock()
    attention_blocks = [MagicMock(), MagicMock()]
    blocks = SimpleNamespace(blocks=([mamba_block], attention_blocks))
    kv_cache_manager = SimpleNamespace(
        truncate_computed_blocks=MagicMock(),
        coordinator=SimpleNamespace(
            single_type_managers=[
                SimpleNamespace(block_size=4),
                SimpleNamespace(block_size=4),
            ]
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=object.__new__(MambaSpec)),
                SimpleNamespace(kv_cache_spec=MagicMock()),
            ]
        ),
        create_kv_cache_blocks=MagicMock(side_effect=lambda value: value),
    )
    scheduler.kv_cache_manager = kv_cache_manager

    truncated = scheduler._truncate_computed_blocks_for_connector(blocks, 8)

    assert truncated == ([mamba_block], attention_blocks)
    kv_cache_manager.truncate_computed_blocks.assert_not_called()

    kv_cache_manager.truncate_computed_blocks.return_value = "unaligned"
    assert scheduler._truncate_computed_blocks_for_connector(blocks, 6) == "unaligned"
    kv_cache_manager.truncate_computed_blocks.assert_called_once_with(blocks, 6)


def test_dsv4_decode_node_observes_real_dense_local_cache_hit():
    register_all_kvcache_specs(MagicMock())
    init_none_hash(sha256)
    hash_block_size = 4
    kv_cache_config = KVCacheConfig(
        num_blocks=800,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["dense_mla"],
                MLAAttentionSpec(
                    block_size=256,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.uint8,
                    **_ratio_kwargs(4),
                    model_version="deepseek_v4",
                ),
            ),
            KVCacheGroupSpec(
                ["swa_tail"],
                SlidingWindowMLASpec(
                    block_size=64,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.uint8,
                    sliding_window=128,
                    model_version="deepseek_v4",
                ),
            ),
            KVCacheGroupSpec(
                ["c4_state"],
                SlidingWindowMLASpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.uint8,
                    sliding_window=8,
                ),
            ),
            KVCacheGroupSpec(
                ["c128_state"],
                SlidingWindowMLASpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.uint8,
                    sliding_window=128,
                ),
            ),
        ],
    )
    manager = KVCacheManager(
        kv_cache_config,
        max_model_len=8192,
        scheduler_block_size=256,
        hash_block_size=hash_block_size,
        enable_caching=True,
    )
    block_hasher = get_request_block_hasher(hash_block_size, sha256)

    def make_request(request_id: str, num_tokens: int) -> Request:
        return Request(
            request_id=request_id,
            prompt_token_ids=[0] * num_tokens,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=block_hasher,
        )

    fill = make_request("fill", 1024)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(fill)
    assert num_computed == 0
    assert (
        manager.allocate_slots(
            fill,
            fill.num_tokens,
            num_new_computed_tokens=0,
            new_computed_blocks=computed_blocks,
        )
        is not None
    )
    manager.free(fill)

    non_dense_block_ids = {
        block.block_id
        for block in manager.block_pool.blocks
        if block.block_hash is not None and get_group_id(block.block_hash) in {1, 2, 3}
    }
    assert non_dense_block_ids
    manager.evict_blocks(non_dense_block_ids)

    replay = make_request("replay", 1280)
    replay.kv_transfer_params = {"do_remote_prefill": True}
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.kv_cache_manager = manager

    _, num_local, shared_prefix_boundary, hit_diverged = scheduler._get_computed_blocks_for_connector(replay)

    assert num_local == 1024
    assert shared_prefix_boundary == 0
    assert hit_diverged


def test_recompute_scheduler_config_picks_sync_and_async_class():
    vllm_config = make_dyntra_test_config()
    vllm_config.scheduler_config.async_scheduling = False
    sync_config = RecomputeSchedulerConfig.initialize_from_config(vllm_config)
    vllm_config.scheduler_config.async_scheduling = True
    async_config = RecomputeSchedulerConfig.initialize_from_config(vllm_config)

    assert sync_config.scheduler_cls == "vllm_ascend.core.recompute_scheduler.RecomputeScheduler"
    assert async_config.scheduler_cls == "vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler"


def test_default_policy_hooks_are_noops():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)

    assert scheduler._apply_load_balance_modifications() is None
    assert scheduler._can_admit_waiting_request(MagicMock()) is True


def test_build_kv_connector_meta_forwards_to_connector():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    connector = MagicMock()
    connector.build_connector_meta.return_value = "meta"

    assert scheduler._build_kv_connector_meta(connector, "output") == "meta"
    connector.build_connector_meta.assert_called_once_with("output")


def test_add_recomputed_outputs_skips_missing_list():
    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

    RecomputeScheduler._add_recomputed_outputs(SimpleNamespace(), outputs)
    RecomputeScheduler._add_recomputed_outputs(SimpleNamespace(recomputed_reqs=None), outputs)

    assert outputs == {}


def test_get_computed_blocks_falls_back_without_remote_prefill():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.get_computed_blocks.return_value = ("blocks", 4, 1)
    request = SimpleNamespace(kv_transfer_params=None)

    blocks, num_local, shared_prefix_boundary, hit_diverged = scheduler._get_computed_blocks_for_connector(request)

    assert (blocks, num_local, shared_prefix_boundary, hit_diverged) == ("blocks", 4, 1, False)


def test_get_computed_blocks_skips_disabled_prefix_lookup():
    coordinator = HybridKVCacheCoordinator.__new__(HybridKVCacheCoordinator)
    coordinator.full_attention_group_id = 0
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=coordinator,
        enable_caching=False,
        empty_kv_cache_blocks="empty",
    )
    request = SimpleNamespace(
        kv_transfer_params={"do_remote_prefill": True},
        skip_reading_prefix_cache=False,
    )

    assert scheduler._get_computed_blocks_for_connector(request) == ("empty", 0, 0, False)


def test_get_computed_blocks_falls_back_when_group_hits_match():
    coordinator = HybridKVCacheCoordinator.__new__(HybridKVCacheCoordinator)
    coordinator.full_attention_group_id = 0
    coordinator.find_longest_cache_hit_per_group = MagicMock(return_value=("computed", [8, 8]))
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=coordinator,
        enable_caching=True,
        get_computed_blocks=MagicMock(return_value=("fallback", 8, 0)),
    )
    request = SimpleNamespace(
        kv_transfer_params={"do_remote_prefill": True},
        skip_reading_prefix_cache=False,
        block_hashes=[],
        num_tokens=16,
    )

    blocks, num_local, shared_prefix_boundary, hit_diverged = scheduler._get_computed_blocks_for_connector(request)

    assert (blocks, num_local, shared_prefix_boundary, hit_diverged) == ("fallback", 8, 0, False)


def test_truncate_computed_blocks_uses_manager_when_aligned():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    blocks = SimpleNamespace(blocks=([MagicMock(), MagicMock()],))
    kv_cache_manager = SimpleNamespace(
        truncate_computed_blocks=MagicMock(return_value="truncated"),
        coordinator=SimpleNamespace(single_type_managers=[SimpleNamespace(block_size=4)]),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=MagicMock())]),
    )
    scheduler.kv_cache_manager = kv_cache_manager

    assert scheduler._truncate_computed_blocks_for_connector(blocks, 8) == "truncated"
    kv_cache_manager.truncate_computed_blocks.assert_called_once_with(blocks, 8)


def test_truncate_computed_blocks_with_mamba_clamp_caps_mamba_group():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    mamba_block = MagicMock()
    attention_blocks = [MagicMock(), MagicMock()]
    blocks = SimpleNamespace(blocks=([mamba_block], attention_blocks))
    kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            single_type_managers=[
                SimpleNamespace(block_size=4),
                SimpleNamespace(block_size=4),
            ]
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=object.__new__(MambaSpec)),
                SimpleNamespace(kv_cache_spec=MagicMock()),
            ]
        ),
        create_kv_cache_blocks=MagicMock(side_effect=lambda value: value),
    )
    scheduler.kv_cache_manager = kv_cache_manager

    truncated = scheduler._truncate_computed_blocks_with_mamba_clamp(blocks, 8)

    assert truncated == ([mamba_block], attention_blocks)


def test_update_waiting_for_remote_kv_failed_recv_caches_partial_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = MagicMock()
    scheduler.failed_recving_kv_req_ids = {"req-1"}
    scheduler.finished_recving_kv_req_ids = {"req-1"}
    scheduler.needs_kv_cache_zeroing = True
    scheduler.kv_cache_manager = MagicMock()
    request = SimpleNamespace(request_id="req-1", num_computed_tokens=4)

    scheduler._update_waiting_for_remote_kv(request)

    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 4)
    scheduler.kv_cache_manager.record_blocks_for_zeroing.assert_called_once_with("req-1", 4)
    assert scheduler.failed_recving_kv_req_ids == set()


def test_update_waiting_for_remote_kv_keeps_partial_success():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = MagicMock()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = {"req-1"}
    scheduler.kv_cache_manager = MagicMock()
    request = SimpleNamespace(request_id="req-1", num_computed_tokens=4, num_tokens=9)

    scheduler._update_waiting_for_remote_kv(request)

    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 4)
    assert request.num_computed_tokens == 4


def test_schedule_and_update_from_output_roundtrip():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    outputs = scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output([request]),
    )

    assert request.request_id in scheduler_output.num_scheduled_tokens
    assert scheduler_output.preempted_reqs == []
    assert scheduler_output.recomputed_reqs == []
    assert request.status == RequestStatus.RUNNING
    assert request.client_index in outputs


def test_schedule_paused_skips_waiting_requests():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)
    scheduler._pause_state = PauseState.PAUSED_ALL

    scheduler_output = scheduler.schedule()

    assert scheduler_output.total_num_scheduled_tokens == 0
    assert request.status == RequestStatus.WAITING
    assert request not in scheduler.running


def test_schedule_skips_waiting_when_connector_match_unknown():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)
    scheduler.connector = MagicMock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (None, False)

    scheduler_output = scheduler.schedule()

    assert request.request_id not in scheduler_output.num_scheduled_tokens
    assert request not in scheduler.running
    assert request in scheduler.skipped_waiting


def test_schedule_resumes_waiting_request_with_cached_tokens():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)
    request.num_computed_tokens = 4

    scheduler_output = scheduler.schedule()

    assert request.request_id in scheduler_output.num_scheduled_tokens
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens >= 4


def test_schedule_recompute_preemption_fallback_finishes_victim():
    _, scheduler = _create_live_recompute_scheduler()
    scheduler.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_producer=False)
    keep, victim = _warmup_two_running(scheduler)
    _fail_first_allocate(scheduler)
    scheduler.connector = MagicMock()
    scheduler.connector.update_state_before_preempt.return_value = False
    scheduler.connector.request_finished.return_value = (False, None)

    scheduler_output = scheduler.schedule()

    assert len(scheduler_output.recomputed_reqs) == 1
    assert scheduler_output.recomputed_reqs[0].request_id == victim.request_id
    assert victim.status == RequestStatus.FINISHED_ABORTED
    assert keep.request_id in scheduler_output.num_scheduled_tokens


def test_schedule_recompute_preemption_offload_records_victim():
    _, scheduler = _create_live_recompute_scheduler()
    scheduler.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_producer=False)
    keep, victim = _warmup_two_running(scheduler)
    _fail_first_allocate(scheduler)
    scheduler.connector = MagicMock()
    scheduler.connector.update_state_before_preempt.return_value = True

    scheduler_output = scheduler.schedule()

    assert scheduler_output.recomputed_reqs == []
    assert len(scheduler_output.preempted_reqs) == 1
    assert isinstance(scheduler_output.preempted_reqs[0], PreemptedRequestData)
    assert scheduler_output.preempted_reqs[0].req_id == victim.request_id
    assert victim.status == RequestStatus.PREEMPTED


def test_schedule_producer_preempts_last_running_request():
    _, scheduler = _create_live_recompute_scheduler()
    keep, victim = _warmup_two_running(scheduler)
    _fail_first_allocate(scheduler)

    scheduler_output = scheduler.schedule()

    assert scheduler_output.recomputed_reqs == []
    assert victim.status == RequestStatus.PREEMPTED
    assert keep.request_id in scheduler_output.num_scheduled_tokens
    assert victim not in scheduler.running


def test_async_recompute_scheduler_constructs():
    _, scheduler = _create_live_recompute_scheduler(async_scheduling=True)
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert isinstance(scheduler, AsyncRecomputeScheduler)
    assert request.request_id in scheduler_output.num_scheduled_tokens


def test_schedule_caps_running_request_with_long_prefill_threshold():
    _, scheduler = _create_live_recompute_scheduler()
    scheduler.max_num_scheduled_tokens = 8
    scheduler.scheduler_config.long_prefill_token_threshold = 4
    request = create_request(
        request_id=1,
        num_tokens=20,
        block_size=scheduler.vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)
    first_output = scheduler.schedule()
    scheduler.update_from_output(first_output, create_model_runner_output([request]))

    scheduler_output = scheduler.schedule()

    assert scheduler_output.num_scheduled_tokens[request.request_id] == 4


def test_schedule_skips_running_request_without_new_tokens():
    _, scheduler = _create_live_recompute_scheduler()
    keep, victim = _warmup_two_running(scheduler)
    keep.num_computed_tokens = keep.num_tokens_with_spec

    scheduler_output = scheduler.schedule()

    assert keep.request_id not in scheduler_output.num_scheduled_tokens
    assert victim.request_id in scheduler_output.num_scheduled_tokens
    assert keep in scheduler.running


def test_schedule_producer_preempts_current_request_when_no_victim_left():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)
    first_output = scheduler.schedule()
    scheduler.update_from_output(first_output, create_model_runner_output([request]))
    scheduler.kv_cache_manager.allocate_slots = lambda *args, **kwargs: None

    scheduler_output = scheduler.schedule()

    assert request.status == RequestStatus.PREEMPTED
    assert request not in scheduler.running
    assert request.request_id not in scheduler_output.num_scheduled_tokens
    assert scheduler_output.total_num_scheduled_tokens == 0


def test_schedule_skips_waiting_when_running_slots_are_full():
    _, scheduler = _create_live_recompute_scheduler(max_num_seqs=1)
    running = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    waiting = create_request(request_id=2, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(running)
    first_output = scheduler.schedule()
    scheduler.update_from_output(first_output, create_model_runner_output([running]))
    scheduler.add_request(waiting)

    scheduler_output = scheduler.schedule()

    assert waiting.status == RequestStatus.WAITING
    assert waiting not in scheduler.running
    assert waiting.request_id not in scheduler_output.num_scheduled_tokens
    assert running.request_id in scheduler_output.num_scheduled_tokens


def test_schedule_skips_waiting_request_still_blocked_on_remote_kv():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler._is_blocked_waiting_status = lambda _status: True
    scheduler._try_promote_blocked_waiting_request = lambda _request: False

    scheduler_output = scheduler.schedule()

    assert request.request_id not in scheduler_output.num_scheduled_tokens
    assert request not in scheduler.running
    assert request in scheduler.skipped_waiting


def test_schedule_skips_waiting_request_with_stale_output_tokens():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)
    request.num_stale_output_tokens = 2
    request.drop_stale_output = False

    scheduler_output = scheduler.schedule()

    assert request.request_id not in scheduler_output.num_scheduled_tokens
    assert request not in scheduler.running
    assert request in scheduler.skipped_waiting


def test_schedule_truncates_connector_partial_tail_when_external_tokens_cover_it():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(
        request_id=1,
        num_tokens=32,
        block_size=scheduler.vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)
    allocated = scheduler.kv_cache_manager.empty_kv_cache_blocks
    scheduler.connector = MagicMock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (24, False)
    scheduler._get_computed_blocks_for_connector = MagicMock(return_value=(allocated, 20, 0, False))
    scheduler._truncate_computed_blocks_for_connector = MagicMock(return_value=allocated)
    scheduler.kv_cache_manager.allocate_slots = MagicMock(return_value=allocated)
    scheduler.kv_cache_manager.get_blocks = MagicMock(return_value=allocated)

    scheduler_output = scheduler.schedule()

    scheduler._truncate_computed_blocks_for_connector.assert_called_once()
    assert request.status == RequestStatus.RUNNING
    assert request.request_id in scheduler_output.num_scheduled_tokens


def test_schedule_falls_back_from_diverged_hit_when_external_tokens_do_not_cover_tail():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(
        request_id=1,
        num_tokens=32,
        block_size=scheduler.vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)
    allocated = scheduler.kv_cache_manager.empty_kv_cache_blocks
    scheduler.connector = MagicMock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (2, False)
    scheduler._get_computed_blocks_for_connector = MagicMock(return_value=(allocated, 20, 0, True))
    scheduler.kv_cache_manager.get_computed_blocks = MagicMock(return_value=(allocated, 16, 0))
    scheduler.kv_cache_manager.allocate_slots = MagicMock(return_value=allocated)
    scheduler.kv_cache_manager.get_blocks = MagicMock(return_value=allocated)

    scheduler_output = scheduler.schedule()

    scheduler.kv_cache_manager.get_computed_blocks.assert_called_once_with(request)
    assert request.status == RequestStatus.RUNNING
    assert request.request_id in scheduler_output.num_scheduled_tokens


def test_schedule_stops_waiting_when_chunked_prefill_is_disabled():
    _, scheduler = _create_live_recompute_scheduler()
    scheduler.max_num_scheduled_tokens = 4
    scheduler.scheduler_config.enable_chunked_prefill = False
    request = create_request(
        request_id=1,
        num_tokens=20,
        block_size=scheduler.vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert request.status == RequestStatus.WAITING
    assert request not in scheduler.running
    assert request.request_id not in scheduler_output.num_scheduled_tokens


def test_schedule_resumes_preempted_waiting_request():
    _, scheduler = _create_live_recompute_scheduler()
    keep, victim = _warmup_two_running(scheduler)
    _fail_first_allocate(scheduler)
    scheduler.schedule()
    assert victim.status == RequestStatus.PREEMPTED

    scheduler_output = scheduler.schedule()

    assert victim.status == RequestStatus.RUNNING
    assert victim.request_id in scheduler_output.num_scheduled_tokens


def test_schedule_rejects_invalid_waiting_request_status():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler._is_blocked_waiting_status = lambda _status: False
    allocated = scheduler.kv_cache_manager.empty_kv_cache_blocks
    scheduler.kv_cache_manager.allocate_slots = MagicMock(return_value=allocated)
    scheduler.kv_cache_manager.get_blocks = MagicMock(return_value=allocated)

    with pytest.raises(RuntimeError, match="Invalid request status"):
        scheduler.schedule()


def test_schedule_consumer_finishes_current_request_when_no_victim_left():
    _, scheduler = _create_live_recompute_scheduler()
    scheduler.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_producer=False)
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)
    first_output = scheduler.schedule()
    scheduler.update_from_output(first_output, create_model_runner_output([request]))
    scheduler.connector = MagicMock()
    scheduler.connector.update_state_before_preempt.return_value = False
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.kv_cache_manager.allocate_slots = lambda *args, **kwargs: None

    scheduler_output = scheduler.schedule()

    assert scheduler_output.recomputed_reqs[0].request_id == request.request_id
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request.request_id not in scheduler_output.num_scheduled_tokens


def test_schedule_skips_running_request_with_async_placeholders():
    _, scheduler = _create_live_recompute_scheduler()
    keep, victim = _warmup_two_running(scheduler)
    keep.num_output_placeholders = 2
    keep.num_computed_tokens = keep.num_prompt_tokens + keep.max_tokens

    scheduler_output = scheduler.schedule()

    assert keep.request_id not in scheduler_output.num_scheduled_tokens
    assert victim.request_id in scheduler_output.num_scheduled_tokens


def test_schedule_skips_running_request_before_decode_eligible_step():
    _, scheduler = _create_live_recompute_scheduler()
    keep, victim = _warmup_two_running(scheduler)
    keep.next_decode_eligible_step = scheduler.current_step + 100

    scheduler_output = scheduler.schedule()

    assert keep.request_id not in scheduler_output.num_scheduled_tokens
    assert victim.request_id in scheduler_output.num_scheduled_tokens


def test_schedule_defers_prefills_on_throttled_step():
    _, scheduler = _create_live_recompute_scheduler()
    keep, victim = _warmup_two_running(scheduler)
    keep.num_computed_tokens = min(keep.num_computed_tokens, keep.num_prompt_tokens - 1)
    waiting = create_request(
        request_id=3,
        num_tokens=20,
        block_size=scheduler.vllm_config.cache_config.block_size,
    )
    scheduler.add_request(waiting)
    scheduler.prefill_capacity_bound = False

    scheduler_output = scheduler.schedule(throttle_prefills=True)

    assert waiting.request_id not in scheduler_output.num_scheduled_tokens
    assert victim.request_id in scheduler_output.num_scheduled_tokens


def test_schedule_priority_preempts_already_scheduled_running_request():
    _, scheduler = _create_live_recompute_scheduler()
    scheduler.policy = SchedulingPolicy.PRIORITY
    keep, victim = _warmup_two_running(scheduler)
    keep.priority = 1
    victim.priority = 0
    original_allocate = scheduler.kv_cache_manager.allocate_slots
    fail_victim_once = {"done": False}

    def allocate_slots(request, *args, **kwargs):
        if request.request_id == victim.request_id and not fail_victim_once["done"]:
            fail_victim_once["done"] = True
            return None
        return original_allocate(request, *args, **kwargs)

    scheduler.kv_cache_manager.allocate_slots = allocate_slots

    scheduler_output = scheduler.schedule()

    assert victim.request_id in scheduler_output.num_scheduled_tokens
    assert keep not in scheduler.running


def test_schedule_records_spec_tokens_for_running_request():
    _, scheduler = _create_live_recompute_scheduler()
    keep, victim = _warmup_two_running(scheduler)
    keep.spec_token_ids = [11, 12, 13, 14]

    scheduler_output = scheduler.schedule()

    assert keep.spec_token_ids == []
    assert keep.request_id in scheduler_output.num_scheduled_tokens
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens.get(keep.request_id, [])
    assert spec_tokens
    assert spec_tokens == spec_tokens[: len(spec_tokens)]


def test_schedule_loads_waiting_request_kv_async():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(
        request_id=1,
        num_tokens=32,
        block_size=scheduler.vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)
    allocated = scheduler.kv_cache_manager.empty_kv_cache_blocks
    scheduler.connector = MagicMock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (16, True)
    scheduler._get_computed_blocks_for_connector = MagicMock(return_value=(allocated, 0, 0, False))
    scheduler.kv_cache_manager.allocate_slots = MagicMock(return_value=allocated)
    scheduler.kv_cache_manager.get_blocks = MagicMock(return_value=allocated)
    scheduler.needs_kv_cache_zeroing = True
    scheduler.kv_cache_manager.get_zeroing_block_ids_in_range = MagicMock(return_value=[7, 8])
    scheduler._skip_zero_block_ids = set()
    scheduler.num_lookahead_tokens = 2

    scheduler_output = scheduler.schedule()

    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request not in scheduler.running
    assert request.request_id not in scheduler_output.num_scheduled_tokens
    inflight = scheduler._inflight_prefills
    assert request in inflight or request.request_id in inflight


def test_schedule_skips_waiting_lora_over_max_and_collects_running_loras():
    _, scheduler = _create_live_recompute_scheduler()
    keep, _victim = _warmup_two_running(scheduler)
    keep.lora_request = SimpleNamespace(lora_int_id=1)
    waiting = create_request(request_id=3, block_size=scheduler.vllm_config.cache_config.block_size)
    waiting.lora_request = SimpleNamespace(lora_int_id=2)
    scheduler.add_request(waiting)
    scheduler.lora_config = SimpleNamespace(max_loras=1)

    scheduler_output = scheduler.schedule()

    assert waiting.status == RequestStatus.WAITING
    assert waiting in scheduler.skipped_waiting
    assert waiting.request_id not in scheduler_output.num_scheduled_tokens
    assert keep.request_id in scheduler_output.num_scheduled_tokens


def test_schedule_admits_waiting_lora_request():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    request.lora_request = SimpleNamespace(lora_int_id=3)
    scheduler.add_request(request)
    scheduler.lora_config = SimpleNamespace(max_loras=4)

    scheduler_output = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert request.request_id in scheduler_output.num_scheduled_tokens


def test_schedule_aligns_mamba_tokens_and_emits_optional_output_fields():
    _, scheduler = _create_live_recompute_scheduler()
    scheduler.need_mamba_block_aligned_split = True
    scheduler._mamba_block_aligned_split = MagicMock(
        side_effect=lambda _request, num_new_tokens, *args, **kwargs: num_new_tokens
    )
    scheduler.use_v2_model_runner = True
    scheduler.dynamic_sd_lookup = {1: 2}
    scheduler.observability_config = SimpleNamespace(enable_logging_iteration_details=True)
    scheduler._make_scheduled_encoder_input_stats = MagicMock(return_value="enc-stats")
    scheduler.ec_connector = MagicMock()
    scheduler.ec_connector.build_connector_meta.return_value = "ec-meta"
    scheduler.kv_cache_manager.take_boundary_state_offloads = MagicMock(return_value={})
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert request.request_id in scheduler_output.num_scheduled_tokens
    assert scheduler_output.num_spec_tokens_to_schedule == 2
    assert scheduler_output.scheduled_encoder_input_stats == "enc-stats"
    assert scheduler_output.ec_connector_metadata == "ec-meta"
    scheduler._mamba_block_aligned_split.assert_called()
    scheduler.kv_cache_manager.take_boundary_state_offloads.assert_called_once()


def test_schedule_breaks_waiting_when_mamba_split_has_no_tokens():
    _, scheduler = _create_live_recompute_scheduler()
    scheduler.need_mamba_block_aligned_split = True
    scheduler._mamba_block_aligned_split = MagicMock(return_value=0)
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert request.status == RequestStatus.WAITING
    assert request.request_id not in scheduler_output.num_scheduled_tokens


def test_schedule_records_partial_group_prefix_stats():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(
        request_id=1,
        num_tokens=32,
        block_size=scheduler.vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)
    allocated = scheduler.kv_cache_manager.empty_kv_cache_blocks
    scheduler.connector = MagicMock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (24, False)
    scheduler._get_computed_blocks_for_connector = MagicMock(return_value=(allocated, 20, 0, True))
    scheduler._truncate_computed_blocks_for_connector = MagicMock(return_value=allocated)
    scheduler.kv_cache_manager.allocate_slots = MagicMock(return_value=allocated)
    scheduler.kv_cache_manager.get_blocks = MagicMock(return_value=allocated)
    scheduler.kv_cache_manager.record_prefix_cache_stats = None
    scheduler.kv_cache_manager.log_stats = True
    scheduler.kv_cache_manager.prefix_cache_stats = MagicMock()

    scheduler.schedule()

    scheduler.kv_cache_manager.prefix_cache_stats.record.assert_called_once()


def test_update_from_output_covers_connector_stats_events_and_finished_ids():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    scheduler.defer_block_free = True
    scheduler._drain_deferred_frees = MagicMock()
    scheduler.perf_metrics = MagicMock()
    scheduler.perf_metrics.is_enabled.return_value = True
    scheduler.perf_metrics.get_step_perf_stats_per_gpu.return_value = "perf"
    scheduler.connector = MagicMock()
    connector_stats = MagicMock()
    connector_stats.is_empty.return_value = False
    scheduler.connector.get_kv_connector_stats.return_value = connector_stats
    scheduler.connector.take_events.return_value = ["evt"]
    scheduler.kv_cache_manager.take_events = MagicMock(return_value=None)
    scheduler.kv_event_publisher = MagicMock()
    scheduler._update_from_kv_xfer_finished = MagicMock()
    scheduler.finished_req_ids_dict = {0: {"done-0"}, 99: {"done-99"}}
    model_output = create_model_runner_output([request], finished_recving={"remote"})

    outputs = scheduler.update_from_output(scheduler_output, model_output)

    scheduler._drain_deferred_frees.assert_called_once_with()
    scheduler._update_from_kv_xfer_finished.assert_called_once()
    scheduler.kv_event_publisher.publish.assert_called_once()
    assert outputs[0].finished_requests == {"done-0"}
    assert outputs[99].finished_requests == {"done-99"}
    assert scheduler.finished_req_ids_dict == {}


def test_update_from_output_skips_stale_dropped_and_missing_requests():
    _, scheduler = _create_live_recompute_scheduler()
    stale = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    missing = create_request(request_id=2, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(stale)
    scheduler.add_request(missing)
    scheduler_output = scheduler.schedule()
    stale.num_stale_output_tokens = 4096
    stale.drop_stale_output = True
    del scheduler.requests[missing.request_id]

    scheduler.update_from_output(scheduler_output, create_model_runner_output([stale, missing]))

    assert stale.status == RequestStatus.RUNNING
    assert missing.request_id not in scheduler.requests


def test_update_from_output_returns_stats_when_there_are_no_request_outputs():
    _, scheduler = _create_live_recompute_scheduler()
    scheduler.make_stats = MagicMock(return_value=SimpleNamespace())
    scheduler_output = scheduler.schedule()

    outputs = scheduler.update_from_output(scheduler_output, create_model_runner_output([]))

    assert outputs[0].scheduler_stats is not None


def test_update_from_output_removes_stopped_preempted_requests():
    _, scheduler = _create_live_recompute_scheduler()
    request = create_request(request_id=1, block_size=scheduler.vllm_config.cache_config.block_size)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    request.status = RequestStatus.PREEMPTED
    scheduler.waiting.add_request(request)

    scheduler.update_from_output(scheduler_output, create_model_runner_output([request], use_eos=True))

    assert request not in scheduler.waiting
    assert request not in scheduler.skipped_waiting


def _mock_kv_connector(*, matched=(0, False), offload=False):
    connector = MagicMock()
    connector.get_num_new_matched_tokens.return_value = matched
    connector.get_num_new_matched_tokens.side_effect = None
    connector.update_state_before_preempt.return_value = offload
    connector.request_finished.return_value = (False, None)
    connector.request_finished_all_groups.return_value = (False, None)
    connector.ensure_cache_available.return_value = True
    connector.build_connector_meta.return_value = "meta"
    connector.get_kv_connector_stats.return_value = None
    connector.take_events.return_value = []
    return connector


def test_recompute_schedule_preempt_skips_and_waiting_edges():
    _, fcfs = _create_live_recompute_scheduler()
    _warmup_two_running(fcfs)
    _fail_first_allocate(fcfs)
    fcfs.schedule()

    _, prio = _create_live_recompute_scheduler()
    keep, victim = _warmup_two_running(prio)
    prio.policy = SchedulingPolicy.PRIORITY
    keep.priority = 10
    victim.priority = 0
    _fail_first_allocate(prio)
    prio.schedule()

    _, offload = _create_live_recompute_scheduler()
    _warmup_two_running(offload)
    offload.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_producer=False)
    offload.connector = _mock_kv_connector(offload=True)
    offload.kv_cache_manager.get_block_ids = MagicMock(return_value=([0],))
    _fail_first_allocate(offload)
    offload.schedule()

    _, fallback = _create_live_recompute_scheduler()
    _warmup_two_running(fallback)
    fallback.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_producer=False)
    fallback.connector = _mock_kv_connector(offload=False)
    fallback.kv_cache_manager.get_block_ids = MagicMock(return_value=([0],))
    _fail_first_allocate(fallback)
    fallback.schedule()

    _, skips = _create_live_recompute_scheduler()
    first, second = _warmup_two_running(skips)
    first.num_output_placeholders = 2
    first.num_computed_tokens = first.num_prompt_tokens + first.max_tokens
    second.next_decode_eligible_step = skips.current_step + 8
    skips.schedule()
    first.num_output_placeholders = 0
    first.num_computed_tokens = 0
    if hasattr(first, "is_prefill_chunk"):
        first.is_prefill_chunk = True
        second.is_prefill_chunk = False
    second.next_decode_eligible_step = 0
    skips.schedule(throttle_prefills=True)

    _, extras = _create_live_recompute_scheduler()
    extras_keep, _ = _warmup_two_running(extras)
    extras.scheduler_config.long_prefill_token_threshold = 8
    extras.need_mamba_block_aligned_split = True
    extras._mamba_block_aligned_split = MagicMock(side_effect=lambda req, n, *a, **k: n)
    extras.lora_config = SimpleNamespace(max_loras=1)
    extras.ec_connector = _mock_kv_connector()
    extras._try_schedule_encoder_inputs = MagicMock(return_value=([0], 8, 99, [1]))
    extras.encoder_cache_manager.allocate = MagicMock()
    extras.kv_cache_manager.take_kv_cache_block_copies = MagicMock(return_value=([], []))
    extras_keep.lora_request = SimpleNamespace(lora_int_id=1)
    extras_keep.spec_token_ids = [9, 8, 7]
    with patch.object(Request, "has_encoder_inputs", new_callable=PropertyMock, return_value=True):
        extras.schedule()


def test_recompute_waiting_connector_lora_stale_and_output_edges():
    connector_sched = _make_recompute_scheduler()
    block_size = connector_sched.vllm_config.cache_config.block_size
    connector = MagicMock()
    matched = [(None, False), (8, True), (4, False)]

    def _matched(*_a, **_k):
        return matched.pop(0) if matched else (0, False)

    connector.get_num_new_matched_tokens.side_effect = _matched
    connector_sched.connector = connector
    connector_sched.connector_prefix_cache_stats = MagicMock()
    connector_sched._get_computed_blocks_for_connector = MagicMock(
        return_value=(connector_sched.kv_cache_manager.empty_kv_cache_blocks, 5, 0, True)
    )
    connector_sched._truncate_computed_blocks_for_connector = MagicMock(
        return_value=connector_sched.kv_cache_manager.empty_kv_cache_blocks
    )
    connector_sched.ec_connector = MagicMock()
    _ec_calls = {"n": 0}

    def _ensure(*_a, **_k):
        _ec_calls["n"] += 1
        return _ec_calls["n"] > 1

    connector_sched.ec_connector.ensure_cache_available.side_effect = _ensure
    connector_sched._build_kv_connector_meta = MagicMock(return_value="meta")
    connector_sched.needs_kv_cache_zeroing = True
    connector_sched._skip_zero_block_ids = set()
    connector_sched.kv_cache_manager.get_zeroing_block_ids_in_range = MagicMock(return_value={1})
    connector_sched.kv_cache_manager.record_prefix_cache_stats = MagicMock()
    for i in range(3):
        connector_sched.add_request(
            create_request(request_id=50 + i, num_tokens=16, max_tokens=4, block_size=block_size)
        )
    connector_sched.schedule()

    stale = _make_recompute_scheduler()
    req = create_request(request_id=60, num_tokens=16, max_tokens=4, block_size=block_size)
    stale.add_request(req)
    req.num_stale_output_tokens = 2
    req.drop_stale_output = False
    stale.schedule()

    resumed = _make_recompute_scheduler()
    resumed_req = create_request(request_id=61, num_tokens=16, max_tokens=4, block_size=block_size)
    resumed.add_request(resumed_req)
    resumed_req.status = RequestStatus.PREEMPTED
    resumed_req.num_computed_tokens = 4
    resumed.schedule()

    lora = _make_recompute_scheduler()
    lora.lora_config = SimpleNamespace(max_loras=1)
    r1 = create_request(request_id=70, num_tokens=16, max_tokens=4, block_size=block_size)
    r2 = create_request(request_id=71, num_tokens=16, max_tokens=4, block_size=block_size)
    r1.lora_request = SimpleNamespace(lora_int_id=1)
    r2.lora_request = SimpleNamespace(lora_int_id=2)
    lora.add_request(r1)
    lora.add_request(r2)
    lora.schedule()

    output_sched = _make_recompute_scheduler()
    output_sched.defer_block_free = True
    output_sched.processed_step_seq = 0
    output_sched._drain_deferred_frees = MagicMock()
    output_sched.use_v2_model_runner = True
    output_sched.dynamic_sd_lookup = {1: 2}
    output_sched.perf_metrics = MagicMock()
    output_sched.perf_metrics.is_enabled.return_value = True
    output_sched.perf_metrics.get_step_perf_stats_per_gpu.return_value = None
    output_sched.connector = _mock_kv_connector()
    output_sched.connector.get_kv_connector_stats.return_value = None
    output_sched.connector.take_events.return_value = ["evt"]
    output_sched.kv_cache_manager.take_events = MagicMock(return_value=["kv"])
    output_sched.kv_event_publisher = MagicMock()
    output_sched._build_kv_connector_meta = MagicMock(return_value="meta")
    output_sched.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    output_sched.kv_cache_manager.take_partial_tail_offloads = MagicMock(return_value=["off"])
    output_sched.kv_cache_manager.take_kv_cache_block_copies = MagicMock(return_value=(["copy"], ["cow"]))
    output_sched._free_cow_retained_blocks = MagicMock()
    output_sched.observability_config = SimpleNamespace(enable_logging_iteration_details=True)
    output_sched._make_scheduled_encoder_input_stats = MagicMock(return_value="enc-stats")
    out_req = create_request(request_id=80, num_tokens=16, max_tokens=4, block_size=block_size)
    output_sched.add_request(out_req)
    out = output_sched.schedule()
    output_sched.finished_req_ids_dict = {0: {"gone"}, 1: {"other"}}
    output_sched.grammar_compile_error_reqs = set()
    kv_out = SimpleNamespace(
        invalid_block_ids={"b"},
        kv_connector_stats=None,
        finished_recving=set(),
        finished_sending=set(),
    )
    model_out = create_model_runner_output(list(output_sched.running) or [out_req])
    model_out.kv_connector_output = kv_out
    output_sched._handle_invalid_blocks = MagicMock(return_value=set())
    output_sched._update_from_kv_xfer_finished = MagicMock()
    output_sched.update_from_output(out, model_out)
