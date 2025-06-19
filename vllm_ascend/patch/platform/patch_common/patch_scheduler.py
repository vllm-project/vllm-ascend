#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
import time
from collections import deque

from vllm.distributed.kv_events import KVEventBatch
from vllm.logger import init_logger
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


def ascend_update_waiting_for_remote_kv(self, request) -> bool:
    """
    P/D: check if the request_id is finished_recving.

    The finished_recving_kv_req_ids list is populated
    on the previous steps()'s update_from_output based
    on the worker side connector.

    When the kv transfer is ready, we cache the blocks
    and the request state will be moved back to WAITING from
    WAITING_FOR_REMOTE_KV.
    """
    if request.request_id not in self.finished_recving_kv_req_ids:
        return False
    assert len(self.kv_cache_config.kv_cache_groups
               ) == 1, "KV connector only supports one KV cache group now"
    # Now that the blocks are ready, actually cache them.
    # In order to make decode node always do the decode step, we transfer every block as long as it contains the
    # data computed by prefill node.
    num_computed_tokens = request.num_tokens
    if num_computed_tokens == request.num_tokens:
        num_computed_tokens -= 1
    self.kv_cache_manager.cache_blocks(
        request,
        num_computed_tokens,
    )

    # Update the request state for scheduling.
    request.num_computed_tokens = num_computed_tokens

    # Return that we are ready.
    self.finished_recving_kv_req_ids.remove(request.request_id)
    return True


def ascend_schedule(self) -> SchedulerOutput:
    # NOTE(woosuk) on the scheduling algorithm:
    # There's no "decoding phase" nor "prefill phase" in the scheduler.
    # Each request just has the num_computed_tokens and
    # num_tokens_with_spec. num_tokens_with_spec =
    # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
    # At each step, the scheduler tries to assign tokens to the requests
    # so that each request's num_computed_tokens can catch up its
    # num_tokens_with_spec. This is general enough to cover
    # chunked prefills, prefix caching, speculative decoding,
    # and the "jump decoding" optimization in the future.

    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    # NOTE: structured_output_request_ids maps
    # a request's (request that uses structured output)
    # request_id to the running request index.
    # This will helps us determine to slice the grammar bitmask
    # and only applies valid mask for requests that
    # uses structured decoding.
    structured_output_request_ids: dict[str, int] = {}

    req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens
    # Encoder-related.
    scheduled_encoder_inputs: dict[str, list[int]] = {}
    encoder_budget = self.max_num_encoder_input_tokens
    # Spec decode-related.
    scheduled_spec_decode_tokens: dict[str, list[int]] = {}

    # For logging.
    scheduled_timestamp = time.monotonic()

    # First, schedule the RUNNING requests.
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]

        num_new_tokens = (request.num_tokens_with_spec -
                          request.num_computed_tokens)
        if (0 < self.scheduler_config.long_prefill_token_threshold <
                num_new_tokens):
            num_new_tokens = (
                self.scheduler_config.long_prefill_token_threshold)
        num_new_tokens = min(num_new_tokens, token_budget)

        # Make sure the input position does not exceed the max model len.
        # This is necessary when using spec decoding.
        num_new_tokens = min(num_new_tokens,
                             self.max_model_len - request.num_computed_tokens)

        # Schedule encoder inputs.
        encoder_inputs_to_schedule = None
        new_encoder_budget = encoder_budget
        if request.has_encoder_inputs:
            (encoder_inputs_to_schedule, num_new_tokens,
             new_encoder_budget) = self._try_schedule_encoder_inputs(
                 request, request.num_computed_tokens, num_new_tokens,
                 encoder_budget)

        if num_new_tokens == 0:
            # The request cannot be scheduled because one of the following
            # reasons:
            # 1. No new tokens to schedule. This may happen when PP>1 and
            #    we have already scheduled all prompt tokens but they are
            #    not finished yet.
            # 2. The encoder budget is exhausted.
            # 3. The encoder cache is exhausted.
            # NOTE(woosuk): Here, by doing `continue` instead of `break`,
            # we do not strictly follow the FCFS scheduling policy and
            # allow the lower-priority requests to be scheduled.
            req_index += 1
            continue

        num_draft_tokens = max(
            num_new_tokens + request.num_computed_tokens - request.num_tokens,
            0)

        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_draft_tokens=num_draft_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens)
            if new_blocks is None:
                # The request cannot be scheduled.
                # Preempt the lowest-priority request.
                preempted_req = self.running.pop()
                self.kv_cache_manager.free(preempted_req)
                preempted_req.status = RequestStatus.PREEMPTED
                preempted_req.num_computed_tokens = 0
                if self.log_stats:
                    preempted_req.record_event(EngineCoreEventType.PREEMPTED,
                                               scheduled_timestamp)

                self.waiting.appendleft(preempted_req)
                preempted_reqs.append(preempted_req)
                if preempted_req == request:
                    # No more request to preempt.
                    can_schedule = False
                    break
            else:
                # The request can be scheduled.
                can_schedule = True
                break
        if not can_schedule:
            break
        assert new_blocks is not None

        # Schedule the request.
        scheduled_running_reqs.append(request)
        if request.use_structured_output:
            # PERF: in case of chunked prefill,
            # request might not include any new tokens.
            # Therefore, we might introduce some additional
            # cycle to fill in the bitmask, which could be a big no-op.
            structured_output_request_ids[request.request_id] = req_index
        req_to_new_block_ids[request.request_id] = (new_blocks.get_block_ids())
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1

        # Speculative decode related.
        if request.spec_token_ids:
            num_scheduled_spec_tokens = (num_new_tokens +
                                         request.num_computed_tokens -
                                         request.num_tokens)
            if num_scheduled_spec_tokens > 0:
                # Trim spec_token_ids list to num_scheduled_spec_tokens.
                del request.spec_token_ids[num_scheduled_spec_tokens:]
                scheduled_spec_decode_tokens[request.request_id] = (
                    request.spec_token_ids)

        # Encoder-related.
        if encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request.request_id] = (
                encoder_inputs_to_schedule)
            # Allocate the encoder cache.
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
            encoder_budget = new_encoder_budget

    # Record the LoRAs in scheduled_running_reqs
    scheduled_loras: set[int] = set()
    if self.lora_config:
        scheduled_loras = set(
            req.lora_request.lora_int_id for req in scheduled_running_reqs
            if req.lora_request and req.lora_request.lora_int_id > 0)
        assert len(scheduled_loras) <= self.lora_config.max_loras

    # Use a temporary deque to collect requests that need to be skipped
    # and put back at the head of the waiting queue later
    skipped_waiting_requests: deque[Request] = deque()

    # Next, schedule the WAITING requests.
    if not preempted_reqs:
        while self.waiting and token_budget > 0:
            if len(self.running) == self.max_num_running_reqs:
                break

            request = self.waiting[0]

            # KVTransfer: skip request if still waiting for remote kvs.
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                is_ready = self._update_waiting_for_remote_kv(request)
                if is_ready:
                    request.status = RequestStatus.WAITING
                else:
                    logger.debug(
                        "%s is still in WAITING_FOR_REMOTE_KVS state.",
                        request.request_id)
                    self.waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    continue

            # Skip request if the structured output request is still waiting
            # for FSM compilation.
            if request.status == RequestStatus.WAITING_FOR_FSM:
                structured_output_req = request.structured_output_request
                if structured_output_req and structured_output_req.grammar:
                    request.status = RequestStatus.WAITING
                else:
                    self.waiting.popleft()
                    skipped_waiting_requests.appendleft(request)
                    continue

            # Check that adding the request still respects the max_loras
            # constraint.
            if self.lora_config and request.lora_request and (
                    len(scheduled_loras) == self.lora_config.max_loras and
                    request.lora_request.lora_int_id not in scheduled_loras):
                # Scheduling would exceed max_loras, skip.
                self.waiting.popleft()
                skipped_waiting_requests.appendleft(request)
                continue

            num_external_computed_tokens = 0
            load_kv_async = False

            # Get already-cached tokens.
            if request.num_computed_tokens == 0:
                # Get locally-cached tokens.
                new_computed_blocks, num_new_local_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(
                        request)

                # Get externally-cached tokens if using a KVConnector.
                if self.connector is not None:
                    num_external_computed_tokens, load_kv_async = (
                        self.connector.get_num_new_matched_tokens(
                            request, num_new_local_computed_tokens))

                # Total computed tokens (local + external).
                num_computed_tokens = (num_new_local_computed_tokens +
                                       num_external_computed_tokens)
            # KVTransfer: WAITING reqs have num_computed_tokens > 0
            # after async KV recvs are completed.
            else:
                new_computed_blocks = (
                    self.kv_cache_manager.create_empty_block_list())
                num_new_local_computed_tokens = 0
                num_computed_tokens = request.num_computed_tokens

            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget

            # KVTransfer: loading remote KV, do not allocate for new work.
            if load_kv_async:
                assert num_external_computed_tokens > 0
                num_new_tokens = 0
            # Number of tokens to be scheduled.
            else:
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                if self.connector is not None and not self.cache_config.enable_prefix_caching \
                    and num_new_tokens > token_budget:
                    break
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens,
                     new_encoder_budget) = self._try_schedule_encoder_inputs(
                         request, num_computed_tokens, num_new_tokens,
                         encoder_budget)
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break

            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens + num_external_computed_tokens,
                num_new_local_computed_tokens,
                new_computed_blocks,
                num_lookahead_tokens=self.num_lookahead_tokens,
                delay_cache_blocks=load_kv_async,
            )
            if new_blocks is None:
                # The request cannot be scheduled.
                break

            # KVTransfer: the connector uses this info to determine
            # if a load is needed. Note that
            # This information is used to determine if a load is
            # needed for this request.
            if self.connector is not None:
                self.connector.update_state_after_alloc(
                    request,
                    new_computed_blocks + new_blocks,
                    num_external_computed_tokens,
                )

            self.waiting.popleft()
            if load_kv_async:
                # If loading async, allocate memory and put request
                # into the WAITING_FOR_REMOTE_KV state.
                skipped_waiting_requests.appendleft(request)
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                continue

            if request.use_structured_output:
                structured_output_request_ids[request.request_id] = req_index
            req_index += 1
            self.running.append(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED,
                                     scheduled_timestamp)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)
            req_to_new_block_ids[request.request_id] = (
                self.kv_cache_manager.get_block_ids(request.request_id))
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            # Count the number of prefix cached tokens.
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = num_computed_tokens
            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

    # Put back any skipped requests at the head of the waiting queue
    if skipped_waiting_requests:
        self.waiting.extendleft(skipped_waiting_requests)

    # Check if the scheduling constraints are satisfied.
    total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
    assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
    assert token_budget >= 0
    assert len(self.running) <= self.max_num_running_reqs
    # Since some requests in the RUNNING queue may not be scheduled in
    # this step, the total number of scheduled requests can be smaller than
    # len(self.running).
    assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
            len(scheduled_running_reqs) <= len(self.running))

    # Get the longest common prefix among all requests in the running queue.
    # This can be potentially used for cascade attention.
    num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
    if self.running:
        any_request = self.running[0]
        num_common_prefix_blocks = (
            self.kv_cache_manager.get_num_common_prefix_blocks(
                any_request, len(self.running)))

    grammar_bitmask = self.structured_output_manager.grammar_bitmask(
        self.requests,
        structured_output_request_ids,
        scheduled_spec_decode_tokens,
    )
    # Construct the scheduler output.
    new_reqs_data = [
        NewRequestData.from_request(req, req_to_new_block_ids[req.request_id])
        for req in scheduled_new_reqs
    ]
    resumed_reqs_data = [
        self._make_cached_request_data(
            req,
            num_scheduled_tokens[req.request_id],
            len(scheduled_spec_decode_tokens.get(req.request_id, ())),
            req_to_new_block_ids[req.request_id],
            resumed_from_preemption=True,
        ) for req in scheduled_resumed_reqs
    ]
    running_reqs_data = [
        self._make_cached_request_data(
            req,
            num_scheduled_tokens[req.request_id],
            len(scheduled_spec_decode_tokens.get(req.request_id, ())),
            req_to_new_block_ids[req.request_id],
            resumed_from_preemption=False,
        ) for req in scheduled_running_reqs
    ]
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=new_reqs_data,
        scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
        num_common_prefix_blocks=num_common_prefix_blocks,
        # finished_req_ids is an existing state in the scheduler,
        # instead of being newly scheduled in this step.
        # It contains the request IDs that are finished in between
        # the previous and the current steps.
        finished_req_ids=self.finished_req_ids,
        free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
        structured_output_request_ids=structured_output_request_ids,
        grammar_bitmask=grammar_bitmask,
    )

    # NOTE(Kuntai): this function is designed for multiple purposes:
    # 1. Plan the KV cache store
    # 2. Wrap up all the KV cache load / save ops into an opaque object
    # 3. Clear the internal states of the connector
    if self.connector is not None:
        meta = self.connector.build_connector_meta(scheduler_output)
        scheduler_output.kv_connector_metadata = meta

    events = self.kv_cache_manager.take_events()
    if events:
        batch = KVEventBatch(ts=time.time(), events=events)
        self.kv_event_publisher.publish(batch)

    # Advance the number of computed tokens for the request AFTER
    # the request is scheduled.
    # 1. The scheduler_output of the current step has to include the
    #    original number of scheduled tokens to determine input IDs.
    # 2. Advance the number of computed tokens here allowing us to
    #    schedule the prefill request again immediately in the next
    #    scheduling step.
    # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
    #    computed tokens will be adjusted in update_from_output.
    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        self.requests[req_id].num_computed_tokens += num_scheduled_token

    self.finished_req_ids = set()
    return scheduler_output


Scheduler._update_waiting_for_remote_kv = ascend_update_waiting_for_remote_kv
Scheduler.schedule = ascend_schedule
