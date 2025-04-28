#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import time
from typing import Dict, Optional

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.sequence import SequenceGroup, SequenceGroupBase, SequenceStatus


@classmethod  # type: ignore
def from_seq_group(
    cls, seq_group: SequenceGroup, use_cache: bool,
    seq_id_to_seq_group: Dict[str, SequenceGroupBase]
) -> Optional["RequestOutput"]:
    finished = seq_group.is_finished()

    if seq_group.request_id in seq_id_to_seq_group:
        group: SequenceGroupBase = seq_id_to_seq_group[seq_group.request_id]
        assembled_seq_group = group.maybe_assemble_group(seq_group)
        if finished:
            group.finish_seq(seq_group)
        if assembled_seq_group is None:
            return None

        # clear finished seq in seq_id_to_seq_group
        if len(group.to_be_finished) == 0:
            for sub_request_id in list(group.seq_id_to_index.keys()):
                if sub_request_id in seq_id_to_seq_group:
                    del seq_id_to_seq_group[sub_request_id]
        return cls.from_seq_group(assembled_seq_group, use_cache,
                                  seq_id_to_seq_group)

    sampling_params = seq_group.sampling_params
    if sampling_params is None:
        raise ValueError(
            "Sampling parameters are missing for a CompletionRequest.")

    if sampling_params.output_kind == RequestOutputKind.FINAL_ONLY and (
            not finished):
        return None

    # Init cache (if needed)
    if use_cache and seq_group.cached_request_output is None:
        seq_group.cached_request_output = RequestOutput(  # type: ignore
            request_id="",
            prompt=None,
            prompt_token_ids=[],
            prompt_logprobs=None,
            outputs=[],
            finished=False)

    top_n_seqs = seq_group.get_seqs()

    # Create the outputs.
    # NOTE: We need omit logprobs here explicitly because the sequence
    # always has the logprobs of the sampled tokens even if the
    # logprobs are not requested.
    include_logprobs = sampling_params.logprobs is not None
    text_buffer_length = sampling_params.output_text_buffer_length
    delta = sampling_params.output_kind == RequestOutputKind.DELTA

    outputs = []
    include_prompt = True
    # num_cached_tokens should be the same for all the sequences
    num_cached_tokens = None
    for i, seq in enumerate(top_n_seqs):
        output_text = seq.get_output_text_to_return(text_buffer_length, delta)

        output_token_ids = seq.get_output_token_ids_to_return(delta)
        num_output_tokens = 1 if isinstance(output_token_ids,
                                            int) else len(output_token_ids)
        num_cached_tokens = seq.data.get_num_cached_tokens()

        output_logprobs = seq.output_logprobs if include_logprobs else None

        if delta:
            # Slice logprobs delta if applicable
            if output_logprobs:
                output_logprobs = output_logprobs[-num_output_tokens:]
            # Don't include prompt if this is after the first output
            # containing decode token ids
            if include_prompt and seq.get_output_len() > num_output_tokens:
                include_prompt = False

        if use_cache:
            # Get cached output object
            cached_outputs = seq_group.cached_request_output.outputs  # type: ignore
            if i >= len(cached_outputs):
                cached_outputs.append(
                    CompletionOutput(index=i,
                                     text="",
                                     token_ids=[],
                                     cumulative_logprob=None,
                                     logprobs=None,
                                     finish_reason=None,
                                     stop_reason=None))
            output = cached_outputs[i]

            # Init cached output object
            assert output.index == i
            output.text = output_text

            if isinstance(output_token_ids, int):
                output.token_ids.clear()
                output.token_ids.append(output_token_ids)
            else:
                output.token_ids = output_token_ids

            output.cumulative_logprob = seq.get_cumulative_logprob() \
                if include_logprobs else None
            output.logprobs = output_logprobs
            output.finish_reason = SequenceStatus.get_finished_reason(
                seq.status)
            output.stop_reason = seq.stop_reason

        else:
            output = CompletionOutput(
                top_n_seqs.index(seq), output_text, [output_token_ids]
                if isinstance(output_token_ids, int) else output_token_ids,
                seq.get_cumulative_logprob() if include_logprobs else None,
                output_logprobs,
                SequenceStatus.get_finished_reason(seq.status),
                seq.stop_reason)

        outputs.append(output)

    # Every sequence in the sequence group should have the same prompt.
    if include_prompt:
        prompt = seq_group.prompt
        prompt_token_ids = seq_group.prompt_token_ids
        encoder_prompt = seq_group.encoder_prompt
        encoder_prompt_token_ids = seq_group.encoder_prompt_token_ids
        prompt_logprobs = seq_group.prompt_logprobs
    else:
        prompt = None
        prompt_token_ids = None
        encoder_prompt = None
        encoder_prompt_token_ids = None
        prompt_logprobs = None
    finished_time = time.time() if finished else None
    seq_group.set_finished_time(finished_time)

    init_kwargs = {
        "request_id": seq_group.request_id,
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "prompt_logprobs": prompt_logprobs,
        "outputs": outputs,
        "finished": finished,
        "metrics": seq_group.metrics,
        "lora_request": seq_group.lora_request,
        "encoder_prompt": encoder_prompt,
        "encoder_prompt_token_ids": encoder_prompt_token_ids,
        "num_cached_tokens": num_cached_tokens,
        "multi_modal_placeholders": seq_group.multi_modal_placeholders
    }

    if use_cache:
        request_output = seq_group.cached_request_output
        request_output.__init__(**init_kwargs)  # type: ignore
    else:
        request_output = cls(**init_kwargs)  # type: ignore

    return request_output


# Add code to clear finished seq in seq_id_to_seq_group
RequestOutput.from_seq_group = from_seq_group
