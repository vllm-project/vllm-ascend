# This patch must run before importing RequestStatus because NonBSP extends the
# enum at import time.
import vllm_ascend.patch.platform.patch_nonbsp_request_status  # noqa: F401, I001
from vllm.v1.request import RequestStatus

from tests.ut.kv_offload.utils import (
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
