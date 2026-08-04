# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
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


def test_update_from_output_settles_finished_request_in_flight_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    request = SimpleNamespace(
        num_in_flight_tokens=1,
        is_finished=lambda: True,
    )
    scheduler.requests = {"request": request}
    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.enable_return_routed_experts = False
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.finished_req_ids_dict = {}
    scheduler.make_stats = MagicMock(return_value=None)

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request": 1},
        recomputed_reqs=None,
    )
    model_runner_output = SimpleNamespace(
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        routed_experts=None,
    )

    assert scheduler.update_from_output(scheduler_output, model_runner_output) == {}
    assert request.num_in_flight_tokens == 0
