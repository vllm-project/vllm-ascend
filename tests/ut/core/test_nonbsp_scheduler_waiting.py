# This patch must run before importing RequestStatus because NonBSP extends the
# enum at import time.
import vllm_ascend.patch.platform.patch_nonbsp_request_status  # noqa: F401, I001
from vllm.v1.request import RequestStatus

from tests.ut.kv_offload.utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)
from vllm_ascend.core.nonbsp_scheduler import NonBSPScheduler


def test_nonbsp_uses_single_persistent_waiting_queue():
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
    assert remote_request in scheduler.waiting
    assert ready_request in scheduler.running
    assert ready_request.request_id in scheduler_output.num_scheduled_tokens
    assert not scheduler.skipped_waiting


def test_nonbsp_blocked_statuses_are_enqueued_in_waiting():
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
    )

    request = create_request(request_id=1)
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler._enqueue_waiting_request(request)

    assert request in scheduler.waiting
    assert not scheduler.skipped_waiting


def test_nonbsp_invalid_async_load_scans_persistent_waiting_queue(monkeypatch):
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
    )
    request = create_request(request_id=1)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.waiting.add_request(request)
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


def _create_nonbsp_scheduler(async_scheduling: bool):
    vllm_config = create_vllm_config(max_num_seqs=2)
    vllm_config.kv_transfer_config = None
    vllm_config.scheduler_config.async_scheduling = async_scheduling
    scheduler = create_scheduler(
        vllm_config,
        scheduler_cls=NonBSPScheduler,
    )
    return vllm_config, scheduler


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

    scheduler.running.remove(request)
    scheduler._lb_pause_request(request, 0.0)
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output([request]),
    )

    assert request.status == RequestStatus.LB_PAUSED
    assert request in scheduler.waiting
    assert request.num_output_placeholders == 0
    assert request.num_output_tokens == 1
