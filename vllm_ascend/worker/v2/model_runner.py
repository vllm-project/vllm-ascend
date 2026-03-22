# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

import functools

import numpy as np
import torch
import vllm
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import set_weight_prefetch_method
from vllm_ascend.worker.v2.aclgraph_utils import AclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.sample.sampler import AscendSampler
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.eagle import AscendEagleSpeculator
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import block_table_wrapper, model_states_wrapper, torch_cuda_wrapper


class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with torch_cuda_wrapper(), block_table_wrapper(), model_states_wrapper():
            super().__init__(vllm_config, device)

        # because we will override these attribute, delete these attribute to
        # make sure it's collected by python gc immediately.
        del self.cudagraph_manager
        del self.req_states
        del self.input_buffers
        del self.sampler
        del self.speculator

        # NPU specific initializations can be added below.
        self.cudagraph_manager: AclGraphManager = AclGraphManager(
            self.vllm_config,
            self.use_aux_hidden_state_outputs,
            self.device,
            self,
        )

        # we define AscendEagleSpeculator in vllm_ascend.worker.v2.spec_decode.eagle
        # init_speculator will return AscendEagleSpeculator when eagle is used.
        # so here we just call init_speculator to reinitialize speculator.
        self.speculator: AscendEagleSpeculator | None = None
        if self.speculative_config is not None:
            self.speculator = init_speculator(self.vllm_config, self.device)

        # AscendRequestState has extra `num_computed_tokens_cpu` attribute.
        # so reinitialize req_states here.
        self.req_states: AscendRequestState = AscendRequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        # so reinitialize input_buffers here.
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )
        # we need to adjust triton operators in sampler,
        # so reinitialize sampler here.
        self.sampler: AscendSampler = AscendSampler(
            max_num_reqs=self.max_num_reqs,
            vocab_size=self.vocab_size,
            device=self.device,
            req_states=self.req_states,
            logprobs_mode=self.model_config.logprobs_mode,
            num_speculative_tokens=self.num_speculative_steps + 1,
        )

        # we need to copy num_computed_tokens back to cpu to help
        # update actual seq_lens_cpu. gpu attention backend doesn't need these
        # attributes, cause their attention backends doesn't use seq_lens_cpu.
        # and seq_lens_cpu is deprecated in gpu_model_runner_v2.
        self.num_computed_tokens_event = torch.npu.Event()
        self.num_computed_tokens_stream = torch.npu.Stream()
        self.num_computed_tokens_cpu = torch.empty(
            self.max_num_reqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        # set this just the same as model runner v1, or it will raise error.
        set_weight_prefetch_method(self.ascend_config.weight_prefetch_config)

        # we need to update full graph params in run_fullgraph,
        # so create a stream to update full graph params.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()

        # we need to use return value of `get_cudagraph_and_dp_padding`
        # to set forward_context in `run_fullgraph`.
        # so we can inherit `execute_model` method.
        self.cudagraph_and_dp_padding: tuple[int, torch.Tensor | None, int] | None = None

        # we need to use input_batch to set forward_context in run_fullgraph.
        # so we can inherit `execute_model` method.
        self.input_batch: AscendInputBatch | None = None

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        """Override GPUModelRunner.execute_model for Ascend NPUs by there reasons:
        1. when run fullgraph, we need to use ret value of `get_cudagraph_and_dp_padding`
        to set forward_context in `run_fullgraph`.
        """

        # use closure to store return value of get_cudagraph_and_dp_padding in model runner.
        def wrapper(func):
            @functools.wraps(func)
            def inner(*args, **kwargs):
                self.cudagraph_and_dp_padding = func(*args, **kwargs)
                return self.cudagraph_and_dp_padding

            return inner

        if self.cudagraph_and_dp_padding is None:
            vllm.v1.worker.gpu.model_runner.get_cudagraph_and_dp_padding = wrapper(
                vllm.v1.worker.gpu.model_runner.get_cudagraph_and_dp_padding
            )

        return super().execute_model(
            scheduler_output,
            intermediate_tensors,
            dummy_run,
            skip_attn_for_dummy_run,
        )

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_tokens_after_padding: int,
    ) -> AscendInputBatch:
        """Override GPUModelRunner.prepare_inputs for Ascend NPUs.
        npu attention backends need seq_lens_cpu to work.
        so we need to prepare seq_lens_cpu here.
        """
        num_tokens = scheduler_output.total_num_scheduled_tokens
        assert num_tokens > 0
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        # Decode first, then prefill.
        # batch_idx -> req_id
        req_ids = sorted(num_tokens_per_req, key=num_tokens_per_req.get)  # type: ignore

        self._update_seq_lens_cpu(scheduler_output, req_ids)

        numtoks_iter = map(num_tokens_per_req.get, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)
        num_valid_tokens = num_scheduled_tokens
        if scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = np.array(
                [
                    num_tokens - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                    for num_tokens, i in zip(num_scheduled_tokens, req_ids)
                ],
                dtype=np.int32,
            )
        attn_state = build_attn_state(
            self.vllm_config,
            self.input_buffers.seq_lens_np,
            num_reqs,
            num_scheduled_tokens,
            num_valid_tokens,
        )
        idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        idx_mapping_cpu = torch.from_numpy(idx_mapping_np)
        idx_mapping = async_copy_to_gpu(idx_mapping_cpu, device=self.device)

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        else:
            num_draft_tokens = np.array(
                [len(draft_tokens.get(req_id, ())) for req_id in req_ids],
                dtype=np.int32,
            )
            total_num_draft_tokens = int(num_draft_tokens.sum())
            total_num_logits = num_reqs + total_num_draft_tokens

            num_logits = num_draft_tokens + 1
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

            max_expand_len = self.num_speculative_steps + 1
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, max_expand_len
            )

        # Get query_start_loc.
        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        query_start_loc_np = np.empty(self.max_num_reqs + 2, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens

        # This is only required for vllm-ascend.
        query_start_loc_np, num_reqs_padded = self._pad_query_start_loc_for_fia(
            num_tokens_padded=num_tokens_after_padding,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            query_start_loc_np=query_start_loc_np,
            max_query_len=max(scheduler_output.num_scheduled_tokens.values()),
        )
        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)

        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]

        # Get prefill tokens if any.
        if self.req_states.any_prefills(idx_mapping_np):
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs]

        # Pad for full CUDA graph mode.
        self.input_buffers.seq_lens_np[num_reqs_padded:] = 0

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
        )

        input_ids = self.input_buffers.input_ids[:num_tokens_after_padding]
        positions = self.input_buffers.positions[:num_tokens_after_padding]

        self.input_batch = AscendInputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            dcp_local_seq_lens=None,  # TODO(Ronald1995): support cp.
            input_ids=input_ids,
            positions=positions,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            # extra attributes for ascend npus.
            seq_lens_np=self.input_buffers.seq_lens_np,
            attn_state=attn_state,
        )
        return self.input_batch

    def postprocess(
        self,
        input_batch,
        sampled_tokens,
        num_sampled,
        num_rejected,
    ):
        """Override GPUModelRunner.postprocess for Ascend NPUs.
        npu attention backends need seq_lens_cpu to work.
        so we need to copy num_computed_tokens back to cpu here.
        """
        super().postprocess(
            input_batch,
            sampled_tokens,
            num_sampled,
            num_rejected,
        )
        # npu attention backend still need to use seq_lens_cpu,
        # we need to copy num_computed_tokens back to cpu.
        default_stream = torch.cuda.current_stream()
        assert self.num_computed_tokens_stream is not None
        assert self.num_computed_tokens_cpu is not None
        with torch.npu.stream(self.num_computed_tokens_stream):
            self.num_computed_tokens_stream.wait_stream(default_stream)
            self.num_computed_tokens_cpu.copy_(
                self.req_states.num_computed_tokens.gpu,
                non_blocking=True,
            )
            self.num_computed_tokens_event.record()

    def _update_seq_lens_cpu(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: list[str],
    ):
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        # wait for num_computed_tokens copy to cpu stream to finish.
        self.num_computed_tokens_event.synchronize()
        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            req_index = self.req_states.req_id_to_index[req_id]
            # num_computed_tokens_cpu has reverted by num_rejected_tokens already.
            # in super postprocess method.
            self.req_states.num_computed_tokens_cpu[req_index] = self.num_computed_tokens_cpu[req_index]

        # update seq_lens_cpu
        for i, req_id in enumerate(req_ids):  # type: ignore
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens = self.req_states.num_computed_tokens_cpu[req_index]
            self.input_buffers.seq_lens_cpu[i] = num_computed_tokens + num_scheduled_tokens[req_id]

    def eplb_warmup(self):
        # TODO(Ronald1995): just define the method in case calling error in
        # worker, implement it in the future.
        pass

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_tokens: int,
        num_reqs: int,
        query_start_loc_np: np.ndarray,
        max_query_len: int,
    ) -> tuple[np.ndarray, int]:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        assert self.cudagraph_and_dp_padding is not None
        _num_tokens_after_padding, _num_tokens_across_dp, synced_cudagraph_mode = self.cudagraph_and_dp_padding
        cudagraph_runtime_mode = CUDAGraphMode(synced_cudagraph_mode)
        if cudagraph_runtime_mode != CUDAGraphMode.FULL:
            return query_start_loc_np, num_reqs
        uniform_decode_query_len = self.cudagraph_manager.uniform_decode_query_len
        is_uniform_decode = self.cudagraph_manager.is_uniform_decode(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            max_query_len=max_query_len,
        )
        if is_uniform_decode:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            num_reqs_padded = num_tokens_padded // uniform_decode_query_len

            last_loc = query_start_loc_np[num_reqs]
            query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = (
                np.arange(1, num_reqs_padded + 1 - num_reqs) * uniform_decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            num_reqs_padded = min(num_tokens_padded, self.max_num_reqs)

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            query_start_loc_np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        return query_start_loc_np, num_reqs_padded
