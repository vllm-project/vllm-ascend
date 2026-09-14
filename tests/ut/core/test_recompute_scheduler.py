# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler


def test_pd_consumer_first_step_injects_placeholder_spec_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
    scheduler.is_kv_producer = False
    scheduler.is_hybrid_model = False
    scheduler.is_mtp_kv_consumer = True
    scheduler.num_spec_tokens = 1
    scheduler.max_model_len = 1024
    scheduler.log_stats = False
    scheduler.connector = None

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
    assert request.spec_token_ids == [PLACEHOLDER_TOKEN_ID]
    assert request.num_tokens_with_spec == request.num_tokens + 1


def test_hybrid_kv_load_failure_matches_group_zero_and_retries_request():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    failed_request = SimpleNamespace(request_id="failed", client_index=2)
    unaffected_request = SimpleNamespace(request_id="unaffected", client_index=3)
    scheduler.running = [unaffected_request]
    scheduler.skipped_waiting = [failed_request]
    scheduler.requests = {
        failed_request.request_id: failed_request,
        unaffected_request.request_id: unaffected_request,
    }
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.get_block_ids.side_effect = lambda request_id: {
        "failed": ([10, 11], [90]),
        "unaffected": ([20], [11]),
    }[request_id]
    scheduler.finish_requests = MagicMock()

    failed_req_ids = scheduler._get_hybrid_kv_load_failed_request_ids({11})
    outputs = defaultdict(list)
    scheduler._retry_hybrid_kv_load_failures(failed_req_ids, outputs)

    assert failed_req_ids == {"failed"}
    scheduler.finish_requests.assert_called_once_with({"failed"}, RequestStatus.FINISHED_STOPPED)
    assert len(outputs[2]) == 1
    assert outputs[2][0].request_id == "failed"
    assert outputs[2][0].stop_reason == "recomputed"
    assert outputs[2][0].new_token_ids == []
    assert 3 not in outputs
