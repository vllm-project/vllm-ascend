from unittest.mock import MagicMock

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request, RequestStatus

import vllm_ascend.patch.platform.patch_async_prefill_recompute  # noqa: F401


def _make_request(*, do_remote_decode: bool, status: RequestStatus) -> Request:
    request = Request(
        request_id="request-id",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "kv_transfer_params": {
                    "do_remote_decode": do_remote_decode,
                    "do_remote_prefill": not do_remote_decode,
                }
            },
        ),
        pooling_params=None,
    )
    request.status = status
    request.num_output_placeholders = 1
    return request


def test_prefiller_drops_terminal_output_after_async_preemption():
    scheduler = AsyncScheduler.__new__(AsyncScheduler)
    request = _make_request(
        do_remote_decode=True,
        status=RequestStatus.PREEMPTED,
    )
    request.num_computed_tokens = 0

    new_token_ids, stopped = scheduler._update_request_with_output(request, [42])

    assert new_token_ids == []
    assert not stopped
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    assert list(request.output_token_ids) == []
    assert request.num_output_placeholders == 0


def test_prefiller_keeps_terminal_output_without_preemption():
    scheduler = AsyncScheduler.__new__(AsyncScheduler)
    scheduler.max_model_len = 16
    scheduler.kv_cache_manager = MagicMock()
    request = _make_request(
        do_remote_decode=True,
        status=RequestStatus.RUNNING,
    )

    new_token_ids, stopped = scheduler._update_request_with_output(request, [42])

    assert new_token_ids == [42]
    assert stopped
    assert list(request.output_token_ids) == [42]
    assert request.num_output_placeholders == 0
    scheduler.kv_cache_manager.cache_blocks.assert_called_once()


def test_decoder_keeps_terminal_output_after_async_preemption():
    scheduler = AsyncScheduler.__new__(AsyncScheduler)
    scheduler.max_model_len = 16
    scheduler.kv_cache_manager = MagicMock()
    request = _make_request(
        do_remote_decode=False,
        status=RequestStatus.PREEMPTED,
    )

    new_token_ids, stopped = scheduler._update_request_with_output(request, [42])

    assert new_token_ids == [42]
    assert stopped
    assert list(request.output_token_ids) == [42]
    assert request.num_output_placeholders == 0
