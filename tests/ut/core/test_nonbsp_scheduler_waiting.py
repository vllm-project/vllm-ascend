from vllm.v1.request import RequestStatus

import vllm_ascend.core.nonbsp_scheduler as nonbsp_scheduler_module
from tests.ut.kv_offload.utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)
from vllm_ascend.core.nonbsp_scheduler import NonBSPScheduler


def _create_scheduler_with_diagnostics(enable_diagnostics: bool):
    vllm_config = create_vllm_config()
    vllm_config.additional_config = {
        "scheduler_config": {
            "nonbsp_config": {
                "enabled": True,
                "enable_diagnostics": enable_diagnostics,
            }
        }
    }
    return create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
    )


def test_nonbsp_scheduler_diagnostics_are_disabled_by_default(monkeypatch):
    scheduler = _create_scheduler_with_diagnostics(False)
    summaries = []
    monkeypatch.setattr(
        nonbsp_scheduler_module,
        "print_scheduler_summary",
        lambda *args: summaries.append(args),
    )

    scheduler.schedule()

    assert summaries == []


def test_nonbsp_scheduler_diagnostics_can_be_enabled(monkeypatch):
    scheduler = _create_scheduler_with_diagnostics(True)
    summaries = []
    monkeypatch.setattr(
        nonbsp_scheduler_module,
        "print_scheduler_summary",
        lambda *args: summaries.append(args),
    )

    scheduler.schedule()

    assert len(summaries) == 1


def test_nonbsp_keeps_async_kv_load_in_skipped_waiting():
    vllm_config = create_vllm_config(max_num_seqs=2)
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
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


def test_nonbsp_blocked_statuses_are_enqueued_in_skipped_waiting():
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
    )

    request = create_request(request_id=1)
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler._enqueue_waiting_request(request)

    assert request in scheduler.skipped_waiting
    assert request not in scheduler.waiting


def test_nonbsp_invalid_async_load_scans_skipped_waiting(monkeypatch):
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
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


def test_nonbsp_prepare_moves_prefetched_request_to_skipped_waiting():
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
    )
    request = create_request(
        request_id=1,
        do_remote_prefill=True,
        block_size=vllm_config.cache_config.block_size,
    )
    scheduler.add_request(request)
    scheduler._lb_kv_prefetch_enabled = True

    candidates = scheduler.prepare_nonbsp_step()

    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request in scheduler.skipped_waiting
    assert request not in scheduler.waiting
    assert request not in candidates
    assert request in scheduler._inflight_prefills


def test_nonbsp_in_blk_builds_admission_mask():
    vllm_config = create_vllm_config(
        max_num_seqs=2,
        block_size=8,
    )
    vllm_config.kv_transfer_config = None
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
    )
    short_request = create_request(request_id=1, num_tokens=9, block_size=8)
    long_request = create_request(request_id=2, num_tokens=17, block_size=8)
    scheduler.add_request(short_request)
    scheduler.add_request(long_request)
    scheduler.prepare_nonbsp_step()
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


def test_nonbsp_empty_in_blk_blocks_all_admission():
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config = None
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
    )
    request = create_request(request_id=1)
    scheduler.add_request(request)
    scheduler.prepare_nonbsp_step()
    scheduler.modifications = {
        "out_blk": [],
        "in_blk": [],
        "freeze": False,
    }

    scheduler_output = scheduler.schedule()

    assert not scheduler.running
    assert request in scheduler.skipped_waiting
    assert request.request_id not in scheduler_output.num_scheduled_tokens


def test_nonbsp_same_block_count_uses_candidate_order():
    vllm_config = create_vllm_config(max_num_seqs=2)
    vllm_config.kv_transfer_config = None
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
    )
    first_request = create_request(request_id=1)
    second_request = create_request(request_id=2)
    scheduler.add_request(first_request)
    scheduler.add_request(second_request)
    candidates = scheduler.prepare_nonbsp_step()
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


def test_nonbsp_out_request_is_not_readmitted_in_same_step():
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config = None
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
    )
    request = create_request(request_id=1)
    scheduler.add_request(request)
    first_output = scheduler.schedule()
    scheduler.update_from_output(
        first_output,
        create_model_runner_output([request]),
    )
    scheduler.prepare_nonbsp_step()
    block_num = (len(request.all_token_ids) + scheduler.block_size - 1) // scheduler.block_size
    scheduler.modifications = {
        "out_blk": [block_num],
        "in_blk": [block_num],
        "freeze": False,
    }

    scheduler.schedule()

    assert request.status == RequestStatus.PREEMPTED
    assert scheduler.is_lb_paused(request)
    assert request in scheduler.skipped_waiting
    assert request not in scheduler.running


def _create_nonbsp_scheduler(async_scheduling: bool):
    vllm_config = create_vllm_config(max_num_seqs=2)
    vllm_config.kv_transfer_config = None
    vllm_config.scheduler_config.async_scheduling = async_scheduling
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
    )
    return vllm_config, scheduler


def test_nonbsp_async_finished_lb_paused_request_is_removed_from_skipped_waiting():
    vllm_config, scheduler = _create_nonbsp_scheduler(async_scheduling=True)
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

    # Pause the running request through the NonBSP out_blk path.
    scheduler.prepare_nonbsp_step()
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


def test_nonbsp_async_schedules_running_request_in_consecutive_steps():
    vllm_config, scheduler = _create_nonbsp_scheduler(async_scheduling=True)
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


def test_nonbsp_sync_skips_until_previous_output_arrives():
    vllm_config, scheduler = _create_nonbsp_scheduler(async_scheduling=False)
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


def test_nonbsp_async_reclaims_placeholder_for_lb_paused_request():
    vllm_config, scheduler = _create_nonbsp_scheduler(async_scheduling=True)
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
    assert scheduler.is_lb_paused(request)
    assert request in scheduler.waiting
    assert request.num_preemptions == num_preemptions
    assert request.num_output_placeholders == 0
    assert request.num_output_tokens == 1


def test_nonbsp_lb_paused_request_resumes_as_cached():
    vllm_config, scheduler = _create_nonbsp_scheduler(async_scheduling=False)
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
    assert scheduler.is_lb_paused(request)
    assert request.num_computed_tokens == num_computed_tokens
    assert request.num_preemptions == num_preemptions
    assert scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids() == block_ids
    assert request.request_id not in scheduler.reset_preempted_req_ids

    resumed_output = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert not scheduler.is_lb_paused(request)
    assert request.request_id in resumed_output.scheduled_cached_reqs.resumed_req_ids
    assert not resumed_output.scheduled_new_reqs


def test_nonbsp_lb_paused_request_keeps_watermark_enabled(monkeypatch):
    vllm_config, scheduler = _create_nonbsp_scheduler(async_scheduling=False)
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
