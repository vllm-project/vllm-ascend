# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType
from unittest.mock import MagicMock, patch

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from vllm_ascend.core import recompute_scheduler as recompute_scheduler_module
from vllm_ascend.core.recompute_scheduler import (
    RecomputeReqInfo,
    RecomputeScheduler,
)


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


def test_recompute_trigger_warning_is_emitted_once():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.requests = {}
    scheduler.enable_return_routed_experts = False
    scheduler.running = []
    scheduler.waiting = MagicMock()
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_event_publisher = MagicMock()
    scheduler.finished_req_ids_dict = {}
    scheduler.make_stats = MagicMock(return_value=None)

    scheduler_output = MagicMock()
    scheduler_output.num_scheduled_tokens = {}
    scheduler_output.recomputed_reqs = [
        RecomputeReqInfo(
            request_id="request-1",
            output_token_ids=[],
            client_index=0,
        )
    ]

    model_runner_output = MagicMock()
    model_runner_output.sampled_token_ids = []
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.routed_experts = None

    with patch.object(recompute_scheduler_module.logger, "warning") as mock_warning:
        outputs = scheduler.update_from_output(scheduler_output, model_runner_output)

    mock_warning.assert_called_once_with(
        "[RecomputeScheduler] Recompute triggered for request %s.",
        "request-1",
    )
    recompute_output = outputs[0].outputs[0]
    assert recompute_output.request_id == "request-1"
    assert recompute_output.stop_reason == "recomputed"
