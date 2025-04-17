#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
from collections import deque

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AscendScheduler(Scheduler):
    """This Scheduler extends vllm's original v1 scheduler
    with prefill-first scheduling strategy."""

    def schedule(self) -> SchedulerOutput:
        if self.scheduler_config.chunked_prefill_enabled:
            return super().schedule()
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # Record scheduled LoRA requests.
        scheduled_loras: set[int] = set()

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: deque[Request] = deque()

        # Schedule prefill requests first.
        while self.waiting and token_budget > 0:
            if len(scheduled_new_reqs) == self.max_num_running_reqs:
                break

            request = self.waiting[0]

            def skip_cur_request():
                self.waiting.popleft()
                skipped_waiting_requests.appendleft(request)

            # Check that adding the request still respects the max_loras
            # constraint.
            if (self.lora_config and request.lora_request and
                (len(scheduled_loras) == self.lora_config.max_loras
                 and request.lora_request.lora_int_id not in scheduled_loras)):
                # Scheduling would exceed max_loras, skip.
                skip_cur_request()
                continue

            prompt_limit = self._get_prompt_limit(request)
            # Get already-cached tokens.
            computed_blocks, num_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request))
            num_new_tokens = request.num_prompt_tokens - num_computed_tokens
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            max_tokens_in_kvcache = (self.kv_cache_config.num_blocks *
                                     self.block_size)
            prompt_limit = min(prompt_limit, max_tokens_in_kvcache)

            # Finish request that exceeds prompt_limit or kv cache size.
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d",
                    num_new_tokens,
                    prompt_limit,
                )
                request.status = RequestStatus.FINISHED_IGNORED
                self.finished_req_ids.add(request.request_id)  # type: ignore
                self.waiting.popleft()
                continue

            if num_new_tokens > token_budget:
                # Scheduling would exceed token_budget, skip.
                skip_cur_request()
                continue

            assert num_new_tokens > 0
            watermark = getattr(self.scheduler_config, "watermark", 0.01)
            if not self._check_watermark_for_prefill(
                    request, num_new_tokens, computed_blocks, watermark):
                # Scheduling would exceed watermark, skip.
                skip_cur_request()
                continue

            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, computed_blocks)
            if new_blocks is None:
                # The request cannot be scheduled.
                break

            self.waiting.popleft()
            self.running.append(request)
            self.scheduled_req_ids.add(request.request_id)
            # Check request status.
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in computed_blocks + new_blocks
            ]
            # Update request info.
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.extendleft(skipped_waiting_requests)

        # If no prefill requests are scheduled,
        # Schedule decode requests next.
        if len(self.scheduled_req_ids) == 0:
            req_index = 0
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]
                if request.request_id in self.scheduled_req_ids:
                    # This request has already been scheduled.
                    req_index += 1
                    continue

                num_new_tokens = (request.num_tokens_with_spec -
                                  request.num_computed_tokens)
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens == 1

                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request, num_new_tokens)
                    if new_blocks is None:
                        # The request cannot be scheduled.
                        # Preempt the lowest-priority request.
                        preempted_req = self.running.pop()
                        self.kv_cache_manager.free(preempted_req)
                        preempted_req.status = RequestStatus.PREEMPTED
                        preempted_req.num_computed_tokens = 0
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
                self.scheduled_req_ids.add(request.request_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in new_blocks
                ]
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

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
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
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,  # type: ignore
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

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

        self.finished_req_ids = set()  # type: ignore
        return scheduler_output

    def _check_watermark_for_prefill(self,
                                     request,
                                     num_new_tokens,
                                     computed_blocks,
                                     watermark=0.01):
        computed_blocks = computed_blocks or []
        watermark_blocks = self.kv_cache_config.num_blocks * watermark
        num_computed_tokens = (request.num_computed_tokens +
                               len(computed_blocks) * self.block_size)
        num_required_blocks = cdiv(num_new_tokens + num_computed_tokens,
                                   self.block_size)
        req_blocks = self.kv_cache_manager.req_to_blocks[request.request_id]
        num_new_blocks = (num_required_blocks - len(req_blocks) -
                          len(computed_blocks))
        num_evictable_computed_blocks = sum(1 for blk in computed_blocks
                                            if blk.ref_cnt == 0)
        # If number of free blocks is less than water mark after allocating, don't allocate.
        if (self.kv_cache_manager.block_pool.get_num_free_blocks() -
                num_evictable_computed_blocks -
                num_new_blocks) < watermark_blocks:
            return False
        return True

    def _get_prompt_limit(self, request: Request) -> int:
        if (self.scheduler_config.chunked_prefill_enabled
                and not self.scheduler_config.is_multi_step):
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(
                self.scheduler_config.max_model_len,
                self.scheduler_config.max_num_batched_tokens,
            )

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if request.lora_request and request.lora_request.long_lora_max_len:
            assert prompt_limit <= request.lora_request.long_lora_max_len
            return request.lora_request.long_lora_max_len
        else:
            return prompt_limit
