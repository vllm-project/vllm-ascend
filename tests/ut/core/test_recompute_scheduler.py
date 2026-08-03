# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import inspect
import textwrap
from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import PauseState
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


def test_schedule_uses_current_mamba_split_contract():
    source = textwrap.dedent(inspect.getsource(RecomputeScheduler.schedule))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_mamba_block_aligned_split"
    ]

    assert calls
    assert max(len(call.args) for call in calls) == 4


def test_failed_remote_kv_load_rezeroes_unwritten_blocks():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = MagicMock()
    scheduler.needs_kv_cache_zeroing = True
    scheduler.kv_cache_manager = MagicMock()
    scheduler.failed_recving_kv_req_ids = {"request"}
    scheduler.finished_recving_kv_req_ids = {"request"}
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=32,
        num_tokens=64,
    )

    scheduler._update_waiting_for_remote_kv(request)

    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 32)
    scheduler.kv_cache_manager.record_blocks_for_zeroing.assert_called_once_with(request.request_id, 32)


def test_schedule_forwards_cow_copies_and_filters_async_loaded_blocks():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.current_step = 0
    scheduler.max_num_scheduled_tokens = 1
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.num_spec_tokens = 0
    scheduler.prefill_capacity_bound = False
    scheduler._pause_state = PauseState.PAUSED_ALL
    scheduler.running = []
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.lora_config = None
    scheduler.max_num_running_reqs = 1
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler.use_v2_model_runner = False
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler._make_cached_request_data = MagicMock(return_value=[])
    scheduler.dynamic_sd_lookup = None
    scheduler.reset_preempted_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = MagicMock()
    scheduler.encoder_cache_manager.get_freed_mm_hashes.return_value = []
    scheduler.needs_kv_cache_zeroing = True
    scheduler._skip_zero_block_ids = {11}
    scheduler.sched_step_seq = 4
    scheduler.defer_block_free = False
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler._update_after_schedule = MagicMock()
    scheduler._free_cow_retained_blocks = MagicMock()

    block_copy = SimpleNamespace(src_block_id=1, dst_block_id=2)
    retained_blocks = [SimpleNamespace(block_id=1), SimpleNamespace(block_id=2)]
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.take_kv_cache_block_copies.return_value = (
        [block_copy],
        retained_blocks,
    )
    scheduler.kv_cache_manager.take_new_block_ids.return_value = [10, 11, 12]

    output = scheduler.schedule()

    assert output.kv_cache_block_copies == [block_copy]
    assert output.new_block_ids_to_zero == [10, 12]
    scheduler._free_cow_retained_blocks.assert_called_once_with(retained_blocks, 5)
