from types import SimpleNamespace
from typing import TypeVar
from unittest.mock import patch

import pytest
import torch
from vllm.config import VllmConfig
from vllm.model_executor.models import ModelRegistry
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

import vllm_ascend.core.dyntra_lb_scheduler as dyntra_lb_scheduler_module
from tests.ut.kv_offload.utils import (
    create_model_runner_output,
    create_request,
    create_vllm_config,
)
from vllm_ascend.core.dyntra_lb_scheduler import (
    AsyncDyntraLBScheduler,
    DyntraLBPolicyMixin,
    DyntraLBScheduler,
    diagnostics_enabled,
    get_dyntra_lb_block_size,
    get_dyntra_lb_request_block_num,
    print_scheduler_summary,
)

SchedulerT = TypeVar("SchedulerT", bound=Scheduler)


def make_dyntra_test_config(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 1024,
    block_size: int = 128,
) -> VllmConfig:
    """Create a scheduler config without importing a real model class."""
    # ModelConfig normally inspects OPTForCausalLM in a subprocess. These
    # scheduler tests only need its static model capabilities.
    model_info = SimpleNamespace(
        architecture="OPTForCausalLM",
        is_text_generation_model=True,
        is_pooling_model=False,
        attn_type="decoder",
        default_seq_pooling_type=None,
        default_tok_pooling_type=None,
        score_type=None,
        supports_multimodal=False,
        supports_multimodal_raw_input_only=False,
        requires_raw_input_tokens=False,
        supports_multimodal_encoder_tp_data=False,
        supports_pp=True,
        has_inner_state=False,
        is_attention_free=False,
        is_hybrid=False,
        has_noops=False,
        supports_mamba_prefix_caching=False,
        supports_replayssm=False,
        supports_transcription=False,
        supports_transcription_only=False,
        supported_video_pruning_methods=(),
    )
    with patch.object(
        ModelRegistry,
        "inspect_model_cls",
        return_value=(model_info, model_info.architecture),
    ):
        return create_vllm_config(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size,
        )


def create_dyntra_lb_scheduler(
    vllm_config: VllmConfig,
    scheduler_cls: type[SchedulerT],
    num_blocks: int = 10000,
) -> SchedulerT:
    """Create a scheduler subclass for DyntraLB unit tests."""
    block_size = vllm_config.cache_config.block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks

    return scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        block_size=block_size,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def test_dyntra_lb_scheduler_uses_policy_mixin():
    assert issubclass(DyntraLBScheduler, DyntraLBPolicyMixin)
    assert issubclass(AsyncDyntraLBScheduler, DyntraLBScheduler)
    assert issubclass(AsyncDyntraLBScheduler, AsyncScheduler)


def _create_scheduler_with_diagnostics(enable_diagnostics: bool):
    vllm_config = make_dyntra_test_config()
    vllm_config.additional_config = {
        "scheduler_config": {
            "dyntra_lb_config": {
                "enabled": True,
                "enable_diagnostics": enable_diagnostics,
            }
        }
    }
    return create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )


def test_dyntra_lb_scheduler_diagnostics_are_disabled_by_default(monkeypatch):
    scheduler = _create_scheduler_with_diagnostics(False)
    summaries = []
    monkeypatch.setattr(
        dyntra_lb_scheduler_module,
        "print_scheduler_summary",
        lambda *args: summaries.append(args),
    )

    scheduler.schedule()

    assert summaries == []


def test_dyntra_lb_scheduler_diagnostics_can_be_enabled(monkeypatch):
    scheduler = _create_scheduler_with_diagnostics(True)
    summaries = []
    monkeypatch.setattr(
        dyntra_lb_scheduler_module,
        "print_scheduler_summary",
        lambda *args: summaries.append(args),
    )

    scheduler.schedule()

    assert len(summaries) == 1


def test_dyntra_lb_keeps_async_kv_load_in_skipped_waiting():
    vllm_config = make_dyntra_test_config(max_num_seqs=2)
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )

    remote_request = create_request(
        request_id=1,
        do_remote_prefill=True,
        block_size=vllm_config.cache_config.block_size,
    )
    ready_request = create_request(
        request_id=2,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(remote_request)
    scheduler.add_request(ready_request)

    scheduler_output = scheduler.schedule()

    assert remote_request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert remote_request in scheduler.skipped_waiting
    assert remote_request not in scheduler.waiting
    assert ready_request in scheduler.running
    assert ready_request.request_id in scheduler_output.num_scheduled_tokens


def test_dyntra_lb_blocked_statuses_are_enqueued_in_skipped_waiting():
    vllm_config = make_dyntra_test_config()
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )

    request = create_request(request_id=1)
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler._enqueue_waiting_request(request)

    assert request in scheduler.skipped_waiting
    assert request not in scheduler.waiting


def test_dyntra_lb_invalid_async_load_scans_skipped_waiting(monkeypatch):
    vllm_config = make_dyntra_test_config()
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    request = create_request(request_id=1)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.skipped_waiting.add_request(request)
    scanned_requests = []

    def _capture_requests(requests, *args, **kwargs):
        scanned_requests.append(list(requests))
        return set(), 0, []

    monkeypatch.setattr(
        scheduler,
        "_update_requests_with_invalid_blocks",
        _capture_requests,
    )

    assert scheduler._handle_invalid_blocks({1}, {}) == set()
    assert scanned_requests[0] == [request]


def test_dyntra_lb_prepare_moves_prefetched_request_to_skipped_waiting():
    vllm_config = make_dyntra_test_config()
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    request = create_request(
        request_id=1,
        do_remote_prefill=True,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)
    scheduler._lb_kv_prefetch_enabled = True

    candidates = scheduler.prepare_dyntra_lb_step()

    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request in scheduler.skipped_waiting
    assert request not in scheduler.waiting
    assert request not in candidates
    assert request in scheduler._inflight_prefills


def test_dyntra_lb_priority_merges_waiting_queues_in_schedule_order():
    vllm_config = make_dyntra_test_config()
    vllm_config.scheduler_config.policy = "priority"
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    highest_priority = create_request(request_id=1)
    middle_priority = create_request(request_id=2)
    lowest_priority = create_request(request_id=3)
    highest_priority.priority = 0
    middle_priority.priority = 1
    lowest_priority.priority = 2
    highest_priority.arrival_time = 3.0
    middle_priority.arrival_time = 2.0
    lowest_priority.arrival_time = 1.0
    scheduler.waiting.add_request(highest_priority)
    scheduler.waiting.add_request(lowest_priority)
    scheduler.skipped_waiting.add_request(middle_priority)

    ordered = scheduler._waiting_requests_in_schedule_order()

    assert ordered == [highest_priority, middle_priority, lowest_priority]


def test_dyntra_lb_priority_preempts_lowest_priority_request():
    block_size = 16
    vllm_config = make_dyntra_test_config(
        max_num_seqs=3,
        max_num_batched_tokens=200,
        block_size=block_size,
    )
    vllm_config.kv_transfer_config = None
    vllm_config.scheduler_config.policy = "priority"
    vllm_config.scheduler_config.watermark = 0.0
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
        num_blocks=6,
    )
    low_priority = create_request(
        request_id=1,
        num_tokens=2 * block_size,
        block_size=block_size,
    )
    low_priority.priority = 5
    low_priority.arrival_time = 1.0
    scheduler.add_request(low_priority)
    first_output = scheduler.schedule()
    scheduler.update_from_output(
        first_output,
        create_model_runner_output([low_priority]),
    )

    high_priority = create_request(
        request_id=2,
        num_tokens=2 * block_size,
        block_size=block_size,
    )
    high_priority.priority = 0
    high_priority.arrival_time = 2.0
    scheduler.add_request(high_priority)
    second_output = scheduler.schedule()
    scheduler.update_from_output(
        second_output,
        create_model_runner_output([low_priority, high_priority]),
    )

    scheduler.schedule()

    assert low_priority.status == RequestStatus.PREEMPTED
    assert low_priority in scheduler.waiting
    assert high_priority in scheduler.running


def test_dyntra_lb_preemption_drops_stale_output_when_kv_delivery_requires_it(
    monkeypatch,
):
    vllm_config = make_dyntra_test_config()
    vllm_config.kv_transfer_config = None
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    request = create_request(request_id=1)
    scheduler.add_request(request)
    first_output = scheduler.schedule()
    scheduler.update_from_output(
        first_output,
        create_model_runner_output([request]),
    )

    # Emulate the stale-output fields provided by the pinned vLLM main.
    request.num_stale_output_tokens = 0
    request.drop_stale_output = False
    scheduler.requires_kv_delivery = True
    preempt_calls = []

    def preempt_request(request, timestamp, *, drop_stale_output=False):
        preempt_calls.append((request, timestamp, drop_stale_output))
        request.status = RequestStatus.PREEMPTED
        scheduler.waiting.prepend_request(request)

    monkeypatch.setattr(scheduler, "_preempt_request", preempt_request)
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "allocate_slots",
        lambda *args, **kwargs: None,
    )

    scheduler.schedule()

    assert len(preempt_calls) == 1
    assert preempt_calls[0][0] is request
    assert preempt_calls[0][2] is True


def test_dyntra_lb_does_not_resume_deliverable_stale_output():
    vllm_config = make_dyntra_test_config()
    vllm_config.kv_transfer_config = None
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    request = create_request(request_id=1)
    scheduler.add_request(request)
    request.num_stale_output_tokens = 1
    request.drop_stale_output = False

    blocked_output = scheduler.schedule()

    assert request in scheduler.skipped_waiting
    assert request not in scheduler.running
    assert request.request_id not in blocked_output.num_scheduled_tokens

    request.num_stale_output_tokens = 0
    resumed_output = scheduler.schedule()

    assert request in scheduler.running
    assert request.request_id in resumed_output.num_scheduled_tokens


def test_dyntra_lb_v026_waits_for_paused_in_flight_output():
    scheduler = SimpleNamespace(_lb_paused_req_ids={"paused"})
    paused_request = SimpleNamespace(
        request_id="paused",
        num_in_flight_tokens=1,
    )
    normally_preempted_request = SimpleNamespace(
        request_id="other",
        num_in_flight_tokens=1,
    )

    assert DyntraLBPolicyMixin._has_pending_deliverable_output(
        scheduler,
        paused_request,
    )
    assert not DyntraLBPolicyMixin._has_pending_deliverable_output(
        scheduler,
        normally_preempted_request,
    )


def test_dyntra_lb_reconciles_connector_hit_with_local_partial_tail(monkeypatch):
    block_size = 16
    vllm_config = make_dyntra_test_config(block_size=block_size)
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    request = create_request(
        request_id=1,
        num_tokens=block_size,
        block_size=block_size,
    )
    scheduler.add_request(request)
    empty_blocks = scheduler.kv_cache_manager.empty_kv_cache_blocks
    connector_local_token_counts = []
    truncate_calls = []

    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "get_computed_blocks_for_connector",
        lambda request: (empty_blocks, 5, 0, False),
        raising=False,
    )

    def truncate_computed_blocks(blocks, num_tokens):
        truncate_calls.append((blocks, num_tokens))
        return empty_blocks

    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "truncate_computed_blocks",
        truncate_computed_blocks,
        raising=False,
    )

    def get_num_new_matched_tokens(request, num_local_tokens):
        connector_local_token_counts.append(num_local_tokens)
        return 8, True

    monkeypatch.setattr(
        scheduler.connector,
        "get_num_new_matched_tokens",
        get_num_new_matched_tokens,
    )

    scheduler.schedule()

    assert connector_local_token_counts == [0]
    assert truncate_calls == [(empty_blocks, 0)]
    assert request.num_computed_tokens == 8
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_dyntra_lb_v026_uses_release_connector_lookup(monkeypatch):
    block_size = 16
    vllm_config = make_dyntra_test_config(block_size=block_size)
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    request = create_request(
        request_id=1,
        num_tokens=block_size,
        block_size=block_size,
    )
    scheduler.add_request(request)
    empty_blocks = scheduler.kv_cache_manager.empty_kv_cache_blocks
    connector_local_token_counts = []

    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "get_computed_blocks_for_connector",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "truncate_computed_blocks",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "record_prefix_cache_stats",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "get_computed_blocks",
        lambda request: (empty_blocks, 5, 0),
    )

    def get_num_new_matched_tokens(request, num_local_tokens):
        connector_local_token_counts.append(num_local_tokens)
        return 8, True

    monkeypatch.setattr(
        scheduler.connector,
        "get_num_new_matched_tokens",
        get_num_new_matched_tokens,
    )

    scheduler.schedule()

    assert connector_local_token_counts == [5]
    assert request.num_computed_tokens == 13
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_dyntra_lb_forwards_partial_tail_and_encoder_cache_metadata(monkeypatch):
    vllm_config = make_dyntra_test_config()
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    partial_tail_offloads = [object()]
    encoder_cache_metadata = object()

    class RecordingSchedulerOutput(SimpleNamespace):
        __dataclass_fields__ = {
            "partial_tail_offloads": object(),
            "ec_manager_metadata": object(),
        }

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    monkeypatch.setattr(
        dyntra_lb_scheduler_module,
        "SchedulerOutput",
        RecordingSchedulerOutput,
    )
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "take_partial_tail_offloads",
        lambda: partial_tail_offloads,
        raising=False,
    )
    monkeypatch.setattr(
        scheduler.encoder_cache_manager,
        "get_manager_metadata",
        lambda: encoder_cache_metadata,
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "_build_kv_connector_meta",
        lambda connector, scheduler_output: None,
    )

    scheduler_output = scheduler.schedule()

    assert scheduler_output.partial_tail_offloads is partial_tail_offloads
    assert scheduler_output.ec_manager_metadata is encoder_cache_metadata


def test_dyntra_lb_v026_omits_unsupported_scheduler_output_fields(monkeypatch):
    vllm_config = make_dyntra_test_config()
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )

    class V026SchedulerOutput(SimpleNamespace):
        __dataclass_fields__ = {}

        def __init__(self, **kwargs):
            assert "partial_tail_offloads" not in kwargs
            assert "ec_manager_metadata" not in kwargs
            super().__init__(**kwargs)

    def unexpected_call():
        raise AssertionError("unsupported v0.26 extension was called")

    monkeypatch.setattr(
        dyntra_lb_scheduler_module,
        "SchedulerOutput",
        V026SchedulerOutput,
    )
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "take_partial_tail_offloads",
        unexpected_call,
        raising=False,
    )
    monkeypatch.setattr(
        scheduler.encoder_cache_manager,
        "get_manager_metadata",
        unexpected_call,
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "_build_kv_connector_meta",
        lambda connector, scheduler_output: None,
    )

    scheduler_output = scheduler.schedule()

    assert not hasattr(scheduler_output, "partial_tail_offloads")
    assert not hasattr(scheduler_output, "ec_manager_metadata")


def test_dyntra_lb_refreshes_blocked_waiting_requests(monkeypatch):
    vllm_config = make_dyntra_test_config()
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    blocked_request = create_request(request_id=1)
    blocked_request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.skipped_waiting.add_request(blocked_request)
    promoted = []
    monkeypatch.setattr(
        scheduler,
        "_try_promote_blocked_waiting_request",
        lambda request: promoted.append(request),
    )

    scheduler._refresh_blocked_waiting_requests()

    assert promoted == [blocked_request]


def test_dyntra_lb_prefetch_tracks_connector_and_capacity_failures(monkeypatch):
    vllm_config = make_dyntra_test_config(max_num_seqs=1)
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    requests = [create_request(request_id=request_id) for request_id in range(1, 5)]
    for request in requests:
        scheduler.add_request(request)
    scheduler._lb_kv_prefetch_enabled = True
    connector_results = iter(((None, False), (0, False)))
    monkeypatch.setattr(
        scheduler.connector,
        "get_num_new_matched_tokens",
        lambda *args: next(connector_results),
    )

    not_ready_req_ids = scheduler._run_lb_kv_prefetch()

    assert not_ready_req_ids == {
        requests[0].request_id,
        requests[2].request_id,
        requests[3].request_id,
    }


def test_dyntra_lb_prefetch_tracks_allocation_failure(monkeypatch):
    vllm_config = make_dyntra_test_config()
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    request = create_request(request_id=1)
    scheduler.add_request(request)
    scheduler._lb_kv_prefetch_enabled = True
    monkeypatch.setattr(
        scheduler.connector,
        "get_num_new_matched_tokens",
        lambda *args: (8, True),
    )
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "allocate_slots",
        lambda *args, **kwargs: None,
    )

    not_ready_req_ids = scheduler._run_lb_kv_prefetch()

    assert not_ready_req_ids == {request.request_id}
    assert request in scheduler.waiting


def test_dyntra_lb_in_blk_builds_admission_mask():
    vllm_config = make_dyntra_test_config(
        max_num_seqs=2,
        block_size=8,
    )
    vllm_config.kv_transfer_config = None
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    short_request = create_request(request_id=1, num_tokens=9, block_size=8)
    long_request = create_request(request_id=2, num_tokens=17, block_size=8)
    scheduler.add_request(short_request)
    scheduler.add_request(long_request)
    scheduler.prepare_dyntra_lb_step()
    scheduler.modifications = {
        "out_blk": [],
        "in_blk": [3],
        "freeze": False,
    }

    scheduler_output = scheduler.schedule()

    assert long_request in scheduler.running
    assert long_request.request_id in scheduler_output.num_scheduled_tokens
    assert short_request in scheduler.skipped_waiting
    assert short_request.request_id not in scheduler_output.num_scheduled_tokens


def test_dyntra_lb_empty_in_blk_blocks_all_admission():
    vllm_config = make_dyntra_test_config()
    vllm_config.kv_transfer_config = None
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    request = create_request(request_id=1)
    scheduler.add_request(request)
    scheduler.prepare_dyntra_lb_step()
    scheduler.modifications = {
        "out_blk": [],
        "in_blk": [],
        "freeze": False,
    }

    scheduler_output = scheduler.schedule()

    assert not scheduler.running
    assert request in scheduler.skipped_waiting
    assert request.request_id not in scheduler_output.num_scheduled_tokens


def test_dyntra_lb_same_block_count_uses_candidate_order():
    vllm_config = make_dyntra_test_config(max_num_seqs=2)
    vllm_config.kv_transfer_config = None
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    first_request = create_request(request_id=1)
    second_request = create_request(request_id=2)
    scheduler.add_request(first_request)
    scheduler.add_request(second_request)
    candidates = scheduler.prepare_dyntra_lb_step()
    block_num = (len(first_request.all_token_ids) + scheduler.block_size - 1) // scheduler.block_size
    scheduler.modifications = {
        "out_blk": [],
        "in_blk": [block_num],
        "freeze": False,
    }

    scheduler.schedule()

    assert candidates == [first_request, second_request]
    assert first_request in scheduler.running
    assert second_request in scheduler.skipped_waiting


def test_dyntra_lb_out_request_is_not_readmitted_in_same_step():
    vllm_config = make_dyntra_test_config()
    vllm_config.kv_transfer_config = None
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=DyntraLBScheduler,
    )
    request = create_request(request_id=1)
    scheduler.add_request(request)
    first_output = scheduler.schedule()
    scheduler.update_from_output(
        first_output,
        create_model_runner_output([request]),
    )
    scheduler.prepare_dyntra_lb_step()
    block_num = (len(request.all_token_ids) + scheduler.block_size - 1) // scheduler.block_size
    scheduler.modifications = {
        "out_blk": [block_num],
        "in_blk": [block_num],
        "freeze": False,
    }

    scheduler.schedule()

    assert request.status == RequestStatus.PREEMPTED
    assert request.request_id in scheduler._lb_paused_req_ids
    assert request in scheduler.skipped_waiting
    assert request not in scheduler.running


def _create_dyntra_lb_scheduler(async_scheduling: bool):
    vllm_config = make_dyntra_test_config(max_num_seqs=2)
    vllm_config.kv_transfer_config = None
    vllm_config.scheduler_config.async_scheduling = async_scheduling
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=AsyncDyntraLBScheduler if async_scheduling else DyntraLBScheduler,
    )
    return vllm_config, scheduler


def test_dyntra_lb_async_finished_lb_paused_request_is_removed_from_skipped_waiting():
    vllm_config, scheduler = _create_dyntra_lb_scheduler(async_scheduling=True)
    request = create_request(
        request_id=1,
        max_tokens=1,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)

    # Schedule one token but deliberately delay its output.
    first_output = scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert request.num_output_placeholders == 1

    # Pause the running request through the DyntraLB out_blk path.
    scheduler.prepare_dyntra_lb_step()
    block_num = (len(request.all_token_ids) + scheduler.block_size - 1) // scheduler.block_size
    scheduler.modifications = {
        "out_blk": [block_num],
        "in_blk": [],
        "freeze": False,
    }

    scheduler.schedule()

    assert request.status == RequestStatus.PREEMPTED
    assert request in scheduler.skipped_waiting
    assert request not in scheduler.waiting

    # The delayed token reaches max_tokens=1 and finishes the paused request.
    scheduler.update_from_output(
        first_output,
        create_model_runner_output([request]),
    )

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request not in scheduler.waiting
    assert request not in scheduler.skipped_waiting

    # Dynamic LB may become inactive on the following step. A stale finished
    # request must not be picked up again.
    scheduler.modifications = None
    scheduler.schedule()


def test_dyntra_lb_async_schedules_running_request_in_consecutive_steps():
    vllm_config, scheduler = _create_dyntra_lb_scheduler(async_scheduling=True)
    request = create_request(
        request_id=1,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)

    first_output = scheduler.schedule()

    assert request.request_id in first_output.num_scheduled_tokens
    assert request.num_output_placeholders == 1

    second_output = scheduler.schedule()

    assert request.request_id in second_output.num_scheduled_tokens
    assert request.num_output_placeholders == 2

    scheduler.update_from_output(
        first_output,
        create_model_runner_output([request]),
    )
    assert request.num_output_placeholders == 1

    scheduler.update_from_output(
        second_output,
        create_model_runner_output([request]),
    )
    assert request.num_output_placeholders == 0
    assert request.num_output_tokens == 2


def test_dyntra_lb_async_v2_sets_pp_decode_cadence():
    vllm_config = make_dyntra_test_config(max_num_seqs=2)
    vllm_config.scheduler_config.async_scheduling = True
    vllm_config.parallel_config.pipeline_parallel_size = 2
    scheduler = create_dyntra_lb_scheduler(
        vllm_config,
        scheduler_cls=AsyncDyntraLBScheduler,
    )
    scheduler.use_v2_model_runner = True
    request = create_request(
        request_id=1,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert request.request_id in scheduler_output.num_scheduled_tokens
    assert scheduler.pp_size == 2
    assert request.next_decode_eligible_step == scheduler.current_step + scheduler.pp_size


def test_dyntra_lb_sync_skips_until_previous_output_arrives():
    vllm_config, scheduler = _create_dyntra_lb_scheduler(async_scheduling=False)
    request = create_request(
        request_id=1,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)

    first_output = scheduler.schedule()
    second_output = scheduler.schedule()

    assert request.request_id in first_output.num_scheduled_tokens
    assert request.request_id not in second_output.num_scheduled_tokens
    assert request.num_output_placeholders == 0


def test_dyntra_lb_async_reclaims_placeholder_for_lb_paused_request():
    vllm_config, scheduler = _create_dyntra_lb_scheduler(async_scheduling=True)
    request = create_request(
        request_id=1,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    num_preemptions = request.num_preemptions

    scheduler.running.remove(request)
    scheduler._lb_pause_request(request, 0.0)
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output([request]),
    )

    assert request.status == RequestStatus.PREEMPTED
    assert request.request_id in scheduler._lb_paused_req_ids
    assert request in scheduler.waiting
    assert request.num_preemptions == num_preemptions
    assert request.num_output_placeholders == 0
    assert request.num_output_tokens == 1


def test_dyntra_lb_paused_request_resumes_as_cached():
    vllm_config, scheduler = _create_dyntra_lb_scheduler(async_scheduling=False)
    request = create_request(
        request_id=1,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)
    first_output = scheduler.schedule()
    scheduler.update_from_output(
        first_output,
        create_model_runner_output([request]),
    )
    scheduler.running.remove(request)
    num_computed_tokens = request.num_computed_tokens
    num_preemptions = request.num_preemptions
    block_ids = scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids()

    scheduler._lb_pause_request(request, 0.0)

    assert request.status == RequestStatus.PREEMPTED
    assert request.request_id in scheduler._lb_paused_req_ids
    assert request.num_computed_tokens == num_computed_tokens
    assert request.num_preemptions == num_preemptions
    assert scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids() == block_ids
    assert request.request_id not in scheduler.reset_preempted_req_ids

    resumed_output = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert request.request_id not in scheduler._lb_paused_req_ids
    assert request.request_id in resumed_output.scheduled_cached_reqs.resumed_req_ids
    assert not resumed_output.scheduled_new_reqs


def test_dyntra_lb_paused_request_keeps_watermark_enabled(monkeypatch):
    vllm_config, scheduler = _create_dyntra_lb_scheduler(async_scheduling=False)
    running_request = create_request(
        request_id=1,
        block_size=vllm_config.cache_config.block_size,
    )
    paused_request = create_request(
        request_id=2,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(running_request)
    scheduler.add_request(paused_request)
    first_output = scheduler.schedule()
    scheduler.update_from_output(
        first_output,
        create_model_runner_output([running_request, paused_request]),
    )
    scheduler.running.remove(paused_request)
    scheduler._lb_pause_request(paused_request, 0.0)

    original_allocate_slots = scheduler.kv_cache_manager.allocate_slots
    has_scheduled_reqs_values = []

    def allocate_slots(request, *args, **kwargs):
        if request is paused_request:
            has_scheduled_reqs_values.append(kwargs["has_scheduled_reqs"])
        return original_allocate_slots(request, *args, **kwargs)

    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "allocate_slots",
        allocate_slots,
    )

    scheduler.schedule()

    assert has_scheduled_reqs_values == [True]


def test_dyntra_lb_pause_emits_diagnostics(monkeypatch):
    messages = []
    monkeypatch.setattr(
        dyntra_lb_scheduler_module.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    scheduler = _create_scheduler_with_diagnostics(True)
    request = create_request(request_id=1)
    scheduler.add_request(request)
    scheduler.schedule()
    scheduler.running.remove(request)

    scheduler._lb_pause_request(request, 0.0)

    assert f"DYNTRA_LB_PAUSE request_id={request.request_id}" in messages
    assert request in scheduler.waiting


def _diagnostic_request(
    request_id: str,
    status: RequestStatus,
    num_tokens: int,
):
    return SimpleNamespace(
        request_id=request_id,
        status=status,
        all_token_ids=list(range(num_tokens)),
        num_prompt_tokens=num_tokens - 1,
        num_computed_tokens=num_tokens - 2,
    )


def test_print_scheduler_summary_includes_waiting_queues(monkeypatch):
    messages = []
    monkeypatch.setattr(
        dyntra_lb_scheduler_module.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    scheduler = SimpleNamespace(
        running=[
            _diagnostic_request(
                "running",
                RequestStatus.RUNNING,
                17,
            )
        ],
        waiting=[
            _diagnostic_request(
                "waiting",
                RequestStatus.WAITING,
                9,
            )
        ],
        skipped_waiting=[
            _diagnostic_request(
                "remote",
                RequestStatus.WAITING_FOR_REMOTE_KVS,
                5,
            )
        ],
        block_size=8,
        cache_config=SimpleNamespace(block_size=8),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={
            "request-1": 1,
            "request-2": 1,
        }
    )

    print_scheduler_summary(scheduler, scheduler_output)

    output = "\n".join(messages)
    assert "schedule() | scheduler req num: [1, 2, 1, 1, 0, 0]" in output
    assert "blk num [3, 3, 2]" in output
    assert "block size [scheduler=8, dyntra_lb=8]" in output
    assert "running request |" not in output
    assert "waiting request |" not in output


def test_print_scheduler_summary_distinguishes_lb_pause_from_preemption(monkeypatch):
    messages = []
    monkeypatch.setattr(
        dyntra_lb_scheduler_module.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    scheduler = SimpleNamespace(
        running=[],
        waiting=[
            _diagnostic_request(
                "waiting",
                RequestStatus.WAITING,
                9,
            ),
            _diagnostic_request(
                "lb-paused",
                RequestStatus.PREEMPTED,
                17,
            ),
            _diagnostic_request(
                "preempted",
                RequestStatus.PREEMPTED,
                25,
            ),
        ],
        skipped_waiting=[],
        block_size=8,
        cache_config=SimpleNamespace(block_size=8),
        _lb_paused_req_ids={"lb-paused"},
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={})

    print_scheduler_summary(scheduler, scheduler_output)

    output = "\n".join(messages)
    assert "schedule() | scheduler req num: [0, 3, 2, 0, 0, 1]" in output
    assert "blk num [0, 9, 5]" in output


def test_print_scheduler_summary_uses_effective_attention_block_size(monkeypatch):
    messages = []
    monkeypatch.setattr(
        dyntra_lb_scheduler_module.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    scheduler = SimpleNamespace(
        running=[
            _diagnostic_request(
                "running",
                RequestStatus.RUNNING,
                90438,
            )
        ],
        waiting=[],
        skipped_waiting=[],
        block_size=14080000,
        cache_config=SimpleNamespace(block_size=2048),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={
            "running": 2,
        }
    )

    print_scheduler_summary(scheduler, scheduler_output)

    output = "\n".join(messages)
    assert "blk num [45, 0, 0]" in output
    assert "block size [scheduler=14080000, dyntra_lb=2048]" in output


def test_diagnostics_enabled_reads_nested_dyntra_lb_config():
    vllm_config = SimpleNamespace(
        additional_config={
            "scheduler_config": {
                "dyntra_lb_config": {
                    "enabled": False,
                    "enable_diagnostics": True,
                }
            }
        }
    )

    assert diagnostics_enabled(vllm_config) is True


def test_diagnostics_enabled_is_false_for_missing_or_invalid_config():
    assert diagnostics_enabled(SimpleNamespace(additional_config=None)) is False
    assert diagnostics_enabled(SimpleNamespace(additional_config={})) is False
    assert diagnostics_enabled(SimpleNamespace(additional_config={"scheduler_config": "bad"})) is False
    assert (
        diagnostics_enabled(SimpleNamespace(additional_config={"scheduler_config": {"dyntra_lb_config": []}})) is False
    )
    assert (
        diagnostics_enabled(
            SimpleNamespace(
                additional_config={
                    "scheduler_config": {
                        "dyntra_lb_config": {
                            "enable_diagnostics": "true",
                        }
                    }
                }
            )
        )
        is False
    )


def test_get_dyntra_lb_block_helpers_reject_invalid_and_round_up():
    valid = SimpleNamespace(cache_config=SimpleNamespace(block_size=8))
    assert get_dyntra_lb_block_size(valid) == 8
    assert get_dyntra_lb_request_block_num(valid, SimpleNamespace(all_token_ids=list(range(9)))) == 2
    for block_size in (0, -1, None, "16"):
        with pytest.raises(RuntimeError, match="Invalid DyntraLB block size"):
            get_dyntra_lb_block_size(SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size)))


def test_print_scheduler_summary_counts_structured_output_waiting(monkeypatch):
    grammar_status = getattr(RequestStatus, "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR", None)
    if grammar_status is None:
        pytest.skip("WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR is not available")
    messages = []
    monkeypatch.setattr(
        dyntra_lb_scheduler_module.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    print_scheduler_summary(
        SimpleNamespace(
            running=[],
            waiting=[],
            skipped_waiting=[_diagnostic_request("fsm", grammar_status, 9)],
            block_size=8,
            cache_config=SimpleNamespace(block_size=8),
        ),
        SimpleNamespace(num_scheduled_tokens={}),
    )
    assert "schedule() | scheduler req num: [0, 1, 0, 0, 1, 0]" in "\n".join(messages)


def _sync_scheduler(**config_kwargs):
    vllm_config = make_dyntra_test_config(**config_kwargs)
    vllm_config.kv_transfer_config = None
    return create_dyntra_lb_scheduler(vllm_config, scheduler_cls=DyntraLBScheduler)


def test_dyntra_lb_waiting_order_fcfs_and_priority_remainder():
    scheduler = _sync_scheduler()
    skipped = create_request(request_id=1)
    waiting = create_request(request_id=2)
    scheduler.skipped_waiting.add_request(skipped)
    scheduler.waiting.add_request(waiting)
    assert scheduler._waiting_requests_in_schedule_order() == [skipped, waiting]

    vllm_config = make_dyntra_test_config()
    vllm_config.scheduler_config.policy = "priority"
    priority_scheduler = create_dyntra_lb_scheduler(vllm_config, scheduler_cls=DyntraLBScheduler)
    highest = create_request(request_id=1)
    middle = create_request(request_id=2)
    lowest = create_request(request_id=3)
    highest.priority, middle.priority, lowest.priority = 0, 1, 2
    highest.arrival_time, middle.arrival_time, lowest.arrival_time = 3.0, 2.0, 1.0
    priority_scheduler.waiting.add_request(highest)
    priority_scheduler.skipped_waiting.add_request(middle)
    priority_scheduler.skipped_waiting.add_request(lowest)
    assert priority_scheduler._waiting_requests_in_schedule_order() == [highest, middle, lowest]


def test_dyntra_lb_prefetch_skips_ineligible_requests(monkeypatch):
    scheduler = create_dyntra_lb_scheduler(make_dyntra_test_config(), scheduler_cls=DyntraLBScheduler)
    scheduler._lb_kv_prefetch_enabled = True
    scheduler.connector = None
    assert scheduler._run_lb_kv_prefetch() == set()

    scheduler = create_dyntra_lb_scheduler(make_dyntra_test_config(), scheduler_cls=DyntraLBScheduler)
    scheduler._lb_kv_prefetch_enabled = True
    blocked = create_request(request_id=1)
    blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    computed = create_request(request_id=2)
    computed.status = RequestStatus.PREEMPTED
    computed.num_computed_tokens = 4
    idle = create_request(request_id=3)
    scheduler.skipped_waiting.add_request(blocked)
    scheduler.waiting.add_request(computed)
    scheduler.add_request(idle)
    monkeypatch.setattr(scheduler.connector, "get_num_new_matched_tokens", lambda *args: (0, False))
    assert scheduler._run_lb_kv_prefetch() == set()
    assert idle.status == RequestStatus.WAITING
    assert idle in scheduler.waiting


def test_dyntra_lb_freeze_and_newly_added_out_blk():
    scheduler = _sync_scheduler(max_num_seqs=2)
    first = create_request(request_id=1)
    second = create_request(request_id=2)
    scheduler.add_request(first)
    scheduler.add_request(second)
    output = scheduler.schedule()
    scheduler.update_from_output(output, create_model_runner_output([first, second]))
    second.lb_newly_added = True
    block_num = (len(second.all_token_ids) + scheduler.block_size - 1) // scheduler.block_size
    scheduler.prepare_dyntra_lb_step()
    scheduler.modifications = {"out_blk": [block_num], "in_blk": [block_num], "freeze": True}
    scheduler.schedule()
    assert second.status == RequestStatus.PREEMPTED
    assert second.request_id in scheduler._lb_paused_req_ids
    assert first in scheduler.running
    assert scheduler.lb_freeze is True
    assert scheduler._lb_admit_req_ids == set()

    scheduler.lb_freeze = True
    scheduler.modifications = None
    scheduler._apply_load_balance_modifications()
    assert scheduler.lb_freeze is False
    assert scheduler._lb_admit_req_ids is None
    assert scheduler._can_admit_waiting_request(first) is True


def test_dyntra_lb_connector_lookup_helpers(monkeypatch):
    scheduler = create_dyntra_lb_scheduler(make_dyntra_test_config(), scheduler_cls=DyntraLBScheduler)
    request = create_request(request_id=1)
    empty_blocks = scheduler.kv_cache_manager.empty_kv_cache_blocks
    recorded = []

    class FakeHybrid:
        def find_longest_cache_hit_per_group(self, block_hashes, max_len):
            return ([object()], (3, 5))

    monkeypatch.setattr(dyntra_lb_scheduler_module, "HybridKVCacheCoordinator", FakeHybrid)
    monkeypatch.setattr(scheduler.kv_cache_manager, "get_computed_blocks_for_connector", None, raising=False)
    monkeypatch.setattr(scheduler.kv_cache_manager, "truncate_computed_blocks", None, raising=False)
    monkeypatch.setattr(scheduler.kv_cache_manager, "coordinator", FakeHybrid())
    monkeypatch.setattr(scheduler.kv_cache_manager, "create_kv_cache_blocks", lambda computed: empty_blocks)
    monkeypatch.setattr(scheduler, "has_mamba_layers", True)
    scheduler.kv_cache_manager.log_stats = True
    scheduler.kv_cache_manager.prefix_cache_stats = SimpleNamespace(record=lambda **kwargs: recorded.append(kwargs))

    blocks, num_local, boundary, diverged, supports_tail = scheduler._get_connector_computed_blocks(request)
    assert (blocks, num_local, boundary, diverged, supports_tail) == (empty_blocks, 5, 0, False, False)
    assert recorded[0]["num_hits"] == 5

    scheduler.kv_cache_manager.log_stats = False
    assert scheduler._get_connector_computed_blocks(request)[1] == 5

    seen = []
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "record_prefix_cache_stats",
        lambda req, num_tokens: seen.append((req, num_tokens)),
        raising=False,
    )
    scheduler._record_prefix_cache_stats(request, 7)
    assert seen == [(request, 7)]


def test_dyntra_lb_lifecycle_hooks_clear_paused_state(monkeypatch):
    scheduler = _sync_scheduler()
    stale = SimpleNamespace(request_id="stale", num_stale_output_tokens=1)
    legacy = SimpleNamespace(request_id="legacy")
    scheduler._lb_paused_req_ids = {"stale", "legacy", "paused", "free"}
    preempt_calls = []
    monkeypatch.setattr(
        Scheduler,
        "_preempt_request",
        lambda self, request, timestamp, *args, **kwargs: preempt_calls.append((request.request_id, args, kwargs)),
    )
    scheduler._preempt_request(stale, 1.0, drop_stale_output=True)
    scheduler._preempt_request(legacy, 2.0)
    assert preempt_calls == [("stale", (), {"drop_stale_output": True}), ("legacy", (), {})]
    assert scheduler._lb_paused_req_ids == {"paused", "free"}

    paused = create_request(request_id=1)
    paused.status = RequestStatus.PREEMPTED
    scheduler.skipped_waiting.add_request(paused)
    monkeypatch.setattr(Scheduler, "_handle_stopped_request", lambda self, request: True)
    assert scheduler._handle_stopped_request(paused) is True
    assert paused not in scheduler.skipped_waiting

    running = create_request(request_id=2)
    running.status = RequestStatus.RUNNING
    monkeypatch.setattr(Scheduler, "_handle_stopped_request", lambda self, request: False)
    assert scheduler._handle_stopped_request(running) is False

    free_req = create_request(request_id=3)
    scheduler._lb_paused_req_ids.add(free_req.request_id)
    monkeypatch.setattr(
        Scheduler,
        "_free_request",
        lambda self, request, delay_free_blocks=False: {"ok": delay_free_blocks},
    )
    assert scheduler._free_request(free_req, delay_free_blocks=True) == {"ok": True}
    assert free_req.request_id not in scheduler._lb_paused_req_ids


def test_dyntra_lb_schedule_paused_all_blocks_waiting():
    scheduler = _sync_scheduler()
    waiting = create_request(request_id=1)
    scheduler.add_request(waiting)
    scheduler._pause_state = PauseState.PAUSED_ALL
    paused_output = scheduler.schedule()
    assert paused_output.num_scheduled_tokens == {}
    assert waiting not in scheduler.running
    assert waiting in scheduler.waiting


def test_dyntra_lb_throttled_step_defers_prefill_chunks():
    scheduler = _sync_scheduler(max_num_seqs=2)
    decode = create_request(request_id=1)
    scheduler.add_request(decode)
    first = scheduler.schedule()
    scheduler.update_from_output(first, create_model_runner_output([decode]))
    prefill = create_request(request_id=2)
    scheduler.add_request(prefill)
    scheduler.schedule()
    scheduler.prefill_capacity_bound = False
    throttled = scheduler.schedule(throttle_prefills=True)
    assert prefill.request_id not in throttled.num_scheduled_tokens


def test_dyntra_lb_caps_waiting_request_by_long_prefill_threshold():
    vllm_config = make_dyntra_test_config()
    vllm_config.kv_transfer_config = None
    vllm_config.scheduler_config.long_prefill_token_threshold = 4
    scheduler = create_dyntra_lb_scheduler(vllm_config, scheduler_cls=DyntraLBScheduler)
    request = create_request(request_id=1, num_tokens=20, block_size=scheduler.block_size)
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == 4


def test_dyntra_lb_waiting_mamba_split_limits_new_tokens(monkeypatch):
    scheduler = _sync_scheduler()
    monkeypatch.setattr(scheduler, "need_mamba_block_aligned_split", True)
    monkeypatch.setattr(scheduler, "_mamba_block_aligned_split", lambda *args, **kwargs: 5)
    request = create_request(request_id=1, num_tokens=12, block_size=scheduler.block_size)
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == 5


def test_dyntra_lb_schedules_waiting_request_with_computed_tokens():
    scheduler = _sync_scheduler()
    request = create_request(request_id=1, num_tokens=10, block_size=scheduler.block_size)
    scheduler.add_request(request)
    request.num_computed_tokens = 4
    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    assert request in scheduler.running


def test_dyntra_lb_keeps_unready_remote_kv_request_in_skipped_waiting(monkeypatch):
    scheduler = _sync_scheduler()
    request = create_request(request_id=1)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.skipped_waiting.add_request(request)
    monkeypatch.setattr(scheduler, "_try_promote_blocked_waiting_request", lambda request: False)
    output = scheduler.schedule()
    assert request in scheduler.skipped_waiting
    assert request.request_id not in output.num_scheduled_tokens


def test_dyntra_lb_skips_waiting_request_when_max_loras_reached():
    scheduler = _sync_scheduler(max_num_seqs=2)
    scheduler.lora_config = SimpleNamespace(max_loras=1)
    first = create_request(request_id=1)
    second = create_request(request_id=2)
    first.lora_request = SimpleNamespace(lora_int_id=1)
    second.lora_request = SimpleNamespace(lora_int_id=2)
    scheduler.add_request(first)
    scheduler.add_request(second)
    output = scheduler.schedule()
    assert first.request_id in output.num_scheduled_tokens
    assert second in scheduler.skipped_waiting


def test_dyntra_lb_waiting_stops_when_kv_slots_unavailable(monkeypatch):
    scheduler = _sync_scheduler()
    request = create_request(request_id=1)
    scheduler.add_request(request)
    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", lambda *args, **kwargs: None)
    output = scheduler.schedule()
    assert request not in scheduler.running
    assert request.request_id not in output.num_scheduled_tokens


def test_dyntra_lb_schedules_encoder_inputs_for_running_request(monkeypatch):
    scheduler = _sync_scheduler()
    request = create_request(request_id=1)
    scheduler.add_request(request)
    first = scheduler.schedule()
    scheduler.update_from_output(first, create_model_runner_output([request]))
    monkeypatch.setattr(type(request), "has_encoder_inputs", True)
    monkeypatch.setattr(
        scheduler,
        "_try_schedule_encoder_inputs",
        lambda *args, **kwargs: ([0], 1, 0, [1]),
    )
    allocated = []
    monkeypatch.setattr(
        scheduler.encoder_cache_manager,
        "allocate",
        lambda req, idx: allocated.append(idx),
    )
    scheduler.ec_connector = SimpleNamespace(
        update_state_after_alloc=lambda *args: None,
        build_connector_meta=lambda scheduler_output: None,
    )
    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    assert allocated == [0, 1]


def test_dyntra_lb_schedule_connector_partial_tail_keep_local(monkeypatch):
    block_size = 16
    vllm_config = make_dyntra_test_config(block_size=block_size)
    scheduler = create_dyntra_lb_scheduler(vllm_config, scheduler_cls=DyntraLBScheduler)
    request = create_request(request_id=1, num_tokens=block_size, block_size=block_size)
    scheduler.add_request(request)
    empty_blocks = scheduler.kv_cache_manager.empty_kv_cache_blocks
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "get_computed_blocks_for_connector",
        lambda request: (empty_blocks, 5, 0, False),
        raising=False,
    )
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "truncate_computed_blocks",
        lambda blocks, num_tokens: empty_blocks,
        raising=False,
    )
    monkeypatch.setattr(scheduler.connector, "get_num_new_matched_tokens", lambda request, num_local: (3, True))
    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    assert request.status == RequestStatus.RUNNING
    assert output.num_scheduled_tokens[request.request_id] == block_size - 5


def test_dyntra_lb_schedule_reconciles_diverged_hybrid_hit(monkeypatch):
    block_size = 16
    vllm_config = make_dyntra_test_config(block_size=block_size)
    scheduler = create_dyntra_lb_scheduler(vllm_config, scheduler_cls=DyntraLBScheduler)
    request = create_request(request_id=1, num_tokens=block_size, block_size=block_size)
    scheduler.add_request(request)
    empty_blocks = scheduler.kv_cache_manager.empty_kv_cache_blocks
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "get_computed_blocks_for_connector",
        lambda request: (empty_blocks, 8, 0, True),
        raising=False,
    )
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "truncate_computed_blocks",
        lambda blocks, num_tokens: empty_blocks,
        raising=False,
    )
    monkeypatch.setattr(scheduler.kv_cache_manager, "get_computed_blocks", lambda request: (empty_blocks, 4, 0))
    monkeypatch.setattr(scheduler.connector, "get_num_new_matched_tokens", lambda request, num_local: (0, False))
    scheduler.schedule()
    assert request.shared_prefix_boundary == 0
    assert request.status == RequestStatus.RUNNING


def test_dyntra_lb_async_kv_load_skips_zeroing_blocks(monkeypatch):
    vllm_config = make_dyntra_test_config()
    scheduler = create_dyntra_lb_scheduler(vllm_config, scheduler_cls=DyntraLBScheduler)
    request = create_request(
        request_id=1,
        do_remote_prefill=True,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)
    monkeypatch.setattr(scheduler, "needs_kv_cache_zeroing", True)
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "get_zeroing_block_ids_in_range",
        lambda request_id, start, end: {7, 8},
        raising=False,
    )
    scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert {7, 8}.issubset(scheduler._skip_zero_block_ids)


def test_dyntra_lb_schedule_optional_output_fields(monkeypatch):
    scheduler = _sync_scheduler()
    request = create_request(request_id=1)
    scheduler.add_request(request)
    scheduler.dynamic_sd_lookup = {1: 3}
    scheduler.defer_block_free = True
    scheduler.sched_step_seq = 4
    copies = [object()]
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "take_kv_cache_block_copies",
        lambda: (copies, ["cow"]),
    )
    freed = []
    monkeypatch.setattr(scheduler, "_free_cow_retained_blocks", lambda blocks, seq: freed.append((blocks, seq)))
    scheduler.ec_connector = SimpleNamespace(
        build_connector_meta=lambda scheduler_output: "ec-meta",
        ensure_cache_available=lambda *args, **kwargs: True,
    )
    object.__setattr__(scheduler.observability_config, "enable_logging_iteration_details", True)
    monkeypatch.setattr(scheduler, "_make_scheduled_encoder_input_stats", lambda inputs: "enc-stats")

    output = scheduler.schedule()
    assert output.kv_cache_block_copies is copies
    assert freed == [(["cow"], 5)]
    assert output.ec_connector_metadata == "ec-meta"
    assert output.num_spec_tokens_to_schedule == 3
    assert output.scheduled_encoder_input_stats == "enc-stats"
    assert scheduler.sched_step_seq == 5


def test_dyntra_lb_schedule_running_spec_and_fcfs_preempt(monkeypatch):
    scheduler = _sync_scheduler()
    request = create_request(request_id=1)
    scheduler.add_request(request)
    first = scheduler.schedule()
    scheduler.update_from_output(first, create_model_runner_output([request]))
    request.spec_token_ids = [9, 8, 7]
    spec_output = scheduler.schedule()
    assert request.request_id in spec_output.scheduled_spec_decode_tokens
    assert spec_output.scheduled_spec_decode_tokens[request.request_id]
    assert request.spec_token_ids == []

    scheduler = _sync_scheduler()
    victim = create_request(request_id=1)
    scheduler.add_request(victim)
    first = scheduler.schedule()
    scheduler.update_from_output(first, create_model_runner_output([victim]))
    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", lambda *args, **kwargs: None)
    scheduler.schedule()
    assert victim.status == RequestStatus.PREEMPTED
    assert victim in scheduler.waiting


def test_dyntra_lb_async_skips_ineligible_running_requests():
    vllm_config, scheduler = _create_dyntra_lb_scheduler(async_scheduling=True)
    finished = create_request(
        request_id=1,
        max_tokens=16,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(finished)
    first = scheduler.schedule()
    finished.num_output_placeholders = 1
    finished.num_computed_tokens = finished.num_prompt_tokens + finished.max_tokens
    second = scheduler.schedule()
    assert finished.request_id in first.num_scheduled_tokens
    assert finished.request_id not in second.num_scheduled_tokens

    vllm_config = make_dyntra_test_config(max_num_seqs=2)
    vllm_config.scheduler_config.async_scheduling = True
    vllm_config.parallel_config.pipeline_parallel_size = 2
    pp_scheduler = create_dyntra_lb_scheduler(vllm_config, scheduler_cls=AsyncDyntraLBScheduler)
    pp_scheduler.use_v2_model_runner = True
    request = create_request(request_id=1, block_size=vllm_config.cache_config.block_size)
    pp_scheduler.add_request(request)
    first = pp_scheduler.schedule()
    second = pp_scheduler.schedule()
    assert request.request_id in first.num_scheduled_tokens
    assert request.request_id not in second.num_scheduled_tokens


def test_dyntra_lb_connector_ext_tokens_none_skips_waiting(monkeypatch):
    vllm_config = make_dyntra_test_config()
    scheduler = create_dyntra_lb_scheduler(vllm_config, scheduler_cls=DyntraLBScheduler)
    request = create_request(request_id=1, block_size=vllm_config.cache_config.block_size)
    scheduler.add_request(request)
    monkeypatch.setattr(scheduler.connector, "get_num_new_matched_tokens", lambda *args: (None, False))
    output = scheduler.schedule()
    assert request in scheduler.skipped_waiting
    assert request.request_id not in output.num_scheduled_tokens


def test_dyntra_lb_mamba_split_zero_skips_running_request(monkeypatch):
    scheduler = _sync_scheduler()
    request = create_request(request_id=1)
    scheduler.add_request(request)
    first = scheduler.schedule()
    scheduler.update_from_output(first, create_model_runner_output([request]))
    monkeypatch.setattr(scheduler, "need_mamba_block_aligned_split", True)
    monkeypatch.setattr(scheduler, "_mamba_block_aligned_split", lambda *args, **kwargs: 0)
    output = scheduler.schedule()
    assert request.request_id not in output.num_scheduled_tokens
    assert request in scheduler.running
