from typing import cast

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sampling_params import SamplingType
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.utils import vllm_version_is


# The current version of `GPUModelRunner._update_states` is v0.13.0.
# The patch part is noted by "====== patch part ======".
def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
    """Update the cached states and the persistent batch with the scheduler
    output.

    The updated states are used by the `_prepare_inputs` function to create
    the input GPU tensors for the model.

    The SamplingMetadata is updated and copied to the GPU if there is a
    new/resumed/paused/finished request in the batch.
    """
    # Remove finished requests from the cached states.
    for req_id in scheduler_output.finished_req_ids:
        self.requests.pop(req_id, None)
        self.num_prompt_logprobs.pop(req_id, None)
    # Remove the finished requests from the persistent batch.
    # NOTE(woosuk): There could be an edge case where finished_req_ids and
    # scheduled_req_ids overlap. This happens when a request is aborted and
    # then resubmitted with the same ID. In this case, we treat them as two
    # distinct requests - clearing the cached states for the first request
    # and handling the second as a new request.
    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)

    # Free the cached encoder outputs.
    for mm_hash in scheduler_output.free_encoder_mm_hashes:
        self.encoder_cache.pop(mm_hash, None)

    # Remove the unscheduled requests from the persistent batch.
    # NOTE(woosuk): The unscheduled requests are either preempted requests
    # or running requests that are not scheduled in this step. We remove
    # them from the persistent batch but keep their cached states since
    # they will be scheduled again sometime in the future.
    scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
    cached_req_ids = self.input_batch.req_id_to_index.keys()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
    # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
    # apart from the forced-preemption case in reset_prefix_cache. And in
    # that case we include the resumed_req_ids in the unscheduled set so
    # that they get cleared from the persistent batch before being re-scheduled
    # in the normal resumed request path.
    unscheduled_req_ids = cached_req_ids - (scheduled_req_ids -
                                            resumed_req_ids)
    # NOTE(woosuk): The persistent batch optimization assumes that
    # consecutive batches contain mostly the same requests. If batches
    # have low request overlap (e.g., alternating between two distinct
    # sets of requests), this optimization becomes very inefficient.
    for req_id in unscheduled_req_ids:
        self.input_batch.remove_request(req_id)

    reqs_to_add: list[CachedRequestState] = []
    # Add new requests to the cached states.
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        sampling_params = new_req_data.sampling_params
        pooling_params = new_req_data.pooling_params

        if (sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        if self.is_pooling_model:
            assert pooling_params is not None
            task = pooling_params.task
            assert task is not None, "You did not set `task` in the API"

            model = cast(VllmModelForPooling, self.get_model())
            to_update = model.pooler.get_pooling_updates(task)
            to_update.apply(pooling_params)

        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            prompt_embeds=new_req_data.prompt_embeds,
            mm_features=new_req_data.mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )
        self.requests[req_id] = req_state

        if sampling_params and sampling_params.prompt_logprobs is not None:
            self.num_prompt_logprobs[req_id] = (
                self.input_batch.vocab_size if sampling_params.prompt_logprobs
                == -1 else sampling_params.prompt_logprobs)

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            self._init_xdrope_positions(req_state)

        reqs_to_add.append(req_state)

    # Update the states of the running/resumed requests.
    is_last_rank = get_pp_group().is_last_rank
    req_data = scheduler_output.scheduled_cached_reqs

    # Wait until valid_sampled_tokens_count is copied to cpu,
    # then use it to update actual num_computed_tokens of each request.
    valid_sampled_token_count = self._get_valid_sampled_token_count()

    for i, req_id in enumerate(req_data.req_ids):
        req_state = self.requests[req_id]
        num_computed_tokens = req_data.num_computed_tokens[i]
        new_block_ids = req_data.new_block_ids[i]
        resumed_from_preemption = req_id in req_data.resumed_req_ids
        num_output_tokens = req_data.num_output_tokens[i]
        req_index = self.input_batch.req_id_to_index.get(req_id)

        # prev_num_draft_len is used in async scheduling mode with
        # spec decode. it indicates if need to update num_computed_tokens
        # of the request. for example:
        # fist step: num_computed_tokens = 0, spec_tokens = [],
        # prev_num_draft_len = 0.
        # second step: num_computed_tokens = 100(prompt length),
        # spec_tokens = [a,b], prev_num_draft_len = 0.
        # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
        # prev_num_draft_len = 2.
        # num_computed_tokens in first step and second step doesn't contain
        # the spec tokens length, but in third step it contains the
        # spec tokens length. we only need to update num_computed_tokens
        # when prev_num_draft_len > 0.
        if req_state.prev_num_draft_len:
            if req_index is None:
                req_state.prev_num_draft_len = 0
            else:
                assert self.input_batch.prev_req_id_to_index is not None
                prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                num_accepted = valid_sampled_token_count[prev_req_index] - 1
                num_rejected = req_state.prev_num_draft_len - num_accepted
                num_computed_tokens -= num_rejected
                req_state.output_token_ids.extend([-1] * num_accepted)

        # Update the cached states.
        req_state.num_computed_tokens = num_computed_tokens

        if not is_last_rank:
            # When using PP, the scheduler sends the sampled tokens back,
            # because there's no direct communication between the first-
            # stage worker and the last-stage worker.
            new_token_ids = req_data.new_token_ids[i]
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec tokens.
            num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    new_token_ids[-num_new_tokens:])
        elif num_output_tokens < len(req_state.output_token_ids):
            # Some output tokens were discarded due to a sync-KV-load
            # failure. Align the cached state.
            del req_state.output_token_ids[num_output_tokens:]
            if req_index is not None:
                end_idx = (self.input_batch.num_prompt_tokens[req_index] +
                           num_output_tokens)
                self.input_batch.num_tokens[req_index] = end_idx
                self.input_batch.num_tokens_no_spec[req_index] = end_idx

        # Update the block IDs.
        if not resumed_from_preemption:
            if new_block_ids is not None:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids,
                                              new_block_ids):
                    block_ids.extend(new_ids)
        else:
            assert req_index is None
            assert new_block_ids is not None
            # The request is resumed from preemption.
            # Replace the existing block IDs with the new ones.
            req_state.block_ids = new_block_ids

        if req_index is None:
            # The request is not in the persistent batch.
            # The request was either preempted and resumed later, or was not
            # scheduled in the previous step and needs to be added again.

            if self.use_async_scheduling and num_output_tokens > 0:
                # We must recover the output token ids for resumed requests in the
                # async scheduling case, so that correct input_ids are obtained.
                resumed_token_ids = req_data.all_token_ids[req_id]
                req_state.output_token_ids = resumed_token_ids[
                    -num_output_tokens:]

            reqs_to_add.append(req_state)
            continue

        # Update the persistent batch.
        self.input_batch.num_computed_tokens_cpu[
            req_index] = num_computed_tokens
        if new_block_ids is not None:
            self.input_batch.block_table.append_row(new_block_ids, req_index)

        # For the last rank, we don't need to update the token_ids_cpu
        # because the sampled tokens are already cached.
        if not is_last_rank:
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index, start_token_index:end_token_index] = new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            self.input_batch.num_tokens[req_index] = end_token_index

        # Add spec_token_ids to token_ids_cpu.
        spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
            req_id, [])
        num_spec_tokens = len(spec_token_ids)
        # For async scheduling, token_ids_cpu assigned from
        # spec_token_ids are placeholders and will be overwritten in
        # _prepare_input_ids.
        if num_spec_tokens:
            start_index = self.input_batch.num_tokens_no_spec[req_index]
            end_token_index = start_index + num_spec_tokens
            self.input_batch.token_ids_cpu[
                req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec tokens.
            self.input_batch.num_tokens[req_index] += num_spec_tokens

        # When speculative decoding is used with structured output,
        # the scheduler can drop draft tokens that do not
        # conform to the schema. This can result in
        # scheduler_output.scheduled_spec_decode_tokens being empty,
        # even when speculative decoding is enabled.
        self.input_batch.spec_token_ids[req_index].clear()
        self.input_batch.spec_token_ids[req_index].extend(spec_token_ids)

        # there are no draft tokens with async scheduling,
        # we clear the spec_decoding info in scheduler_output and
        # use normal sampling but rejection_sampling.
        if self.use_async_scheduling:
            req_state.prev_num_draft_len = num_spec_tokens
            if num_spec_tokens and self._draft_token_ids is None:
                scheduler_output.total_num_scheduled_tokens -= num_spec_tokens
                scheduler_output.num_scheduled_tokens[
                    req_id] -= num_spec_tokens
                scheduler_output.scheduled_spec_decode_tokens.pop(req_id, None)
    # Add the new or resumed requests to the persistent batch.
    # The smaller empty indices are filled first.
    for request in reqs_to_add:
        self.input_batch.add_request(request)

        # ========================= patch part =============================
        # Update the request state with the number of draft tokens for async
        # scheduling. This tracks token generation progress and maintains
        # request state. NOTE: The spec tokens are placeholders and not
        # added to token_ids_cpu.
        if self.is_kv_consumer and self.speculative_config and \
            self.speculative_config.method == "mtp" and self.use_async_scheduling:
            req_state = self.requests[request.req_id]
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                request.req_id, [])
            req_state.prev_num_draft_len = len(spec_token_ids)
        # ==================================================================

    # Condense the batched states if there are gaps left by removed requests
    self.input_batch.condense()
    # Allow attention backend to reorder the batch, potentially
    self._may_reorder_batch(scheduler_output)
    # Refresh batch metadata with any pending updates.
    self.input_batch.refresh_metadata()


if vllm_version_is('0.13.0'):
    GPUModelRunner._update_states = _update_states
else:
    logger.warning(
        "vllm_version is not v0.13.0, patch GPUModelRunner._update_states failed!"
    )