# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import inspect
import textwrap
from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.core.recompute_scheduler import (
    RecomputeReqInfo,
    RecomputeScheduler,
)


def test_add_request_does_not_inject_placeholder_spec_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
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


def test_finished_request_propagates_all_transfer_params():
    """Guard the vLLM _free_request and EngineCoreOutput protocol."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(RecomputeScheduler.update_from_output)))

    free_request_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_free_request"
    ]
    assert len(free_request_assignments) == 1
    target = free_request_assignments[0].targets[0]
    assert isinstance(target, ast.Tuple)
    assert [ast.unparse(element) for element in target.elts] == [
        "kv_transfer_params",
        "ec_transfer_params",
    ]

    output_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "EngineCoreOutput"
        and any(keyword.arg == "kv_transfer_params" for keyword in node.keywords)
    ]
    assert len(output_calls) == 1
    output_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in output_calls[0].keywords}
    assert output_keywords["kv_transfer_params"] == "kv_transfer_params"
    assert output_keywords["ec_transfer_params"] == "ec_transfer_params"
