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

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import (InputBatch,
                                            combine_sampled_and_draft_tokens,
                                            prepare_pos_seq_lens,
                                            prepare_prefill_inputs)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2.aclgraph_utils import AclGraphManager
from vllm_ascend.worker.v2.attn_utils import (build_attn_metadata,
                                              build_attn_state)
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.sample.sampler import AscendSampler
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.eagle import AscendEagleSpeculator
from vllm_ascend.worker.v2.states import AscendRequestState, uva_wrapper
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

logger = init_logger(__name__)


class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with (torch_cuda_wrapper(), uva_wrapper()):
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
            vllm_config,
            device,
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
            pin_memory=self.pin_memory,
        )
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        # so reinitialize input_buffers here.
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            inputs_embeds_size=self.inputs_embeds_size,
            vocab_size=self.vocab_size,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )
        # we need to adjust triton operators in sampler,
        # so reinitialize sampler here.
        self.sampler: AscendSampler = AscendSampler(
            logprobs_mode=self.model_config.logprobs_mode, )

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
            pin_memory=self.pin_memory,
        )

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_tokens_after_padding: int,
    ) -> InputBatch:
        """Override GPUModelRunner.prepare_inputs for Ascend NPUs.
        npu attention backends need seq_lens_cpu to work.
        so we need to prepare seq_lens_cpu here.
        """
        num_tokens = scheduler_output.total_num_scheduled_tokens
        assert num_tokens > 0
        num_reqs = len(scheduler_output.num_scheduled_tokens)

        # Decode first, then prefill.
        # batch_idx -> req_id
        req_ids = sorted(
            scheduler_output.num_scheduled_tokens.keys(),
            key=lambda k: scheduler_output.num_scheduled_tokens[k],
        )

        self._update_seq_lens_cpu(scheduler_output, req_ids)

        num_scheduled_tokens = np.array(
            [scheduler_output.num_scheduled_tokens[i] for i in req_ids],
            dtype=np.int32)
        num_valid_tokens = num_scheduled_tokens
        if scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = np.array(
                [
                    num_tokens - len(
                        scheduler_output.scheduled_spec_decode_tokens.get(
                            i, []))
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

        idx_mapping_list = [
            self.req_states.req_id_to_index[req_id] for req_id in req_ids
        ]
        idx_mapping = self.input_buffers.idx_mapping
        idx_mapping.np[:num_reqs] = idx_mapping_list
        idx_mapping_np = idx_mapping.np[:num_reqs]
        idx_mapping_cpu = idx_mapping.cpu[:num_reqs]
        idx_mapping_npu = idx_mapping.copy_to_gpu(num_reqs)

        # Get the number of draft tokens for each request.
        if not scheduler_output.scheduled_spec_decode_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits = torch.arange(num_reqs + 1,
                                         device=self.device,
                                         dtype=torch.int32)
        else:
            draft_tokens = scheduler_output.scheduled_spec_decode_tokens
            num_draft_tokens = np.array(
                [
                    len(draft_tokens[req_id]) if req_id in draft_tokens else 0
                    for req_id in req_ids
                ],
                dtype=np.int32,
            )
            total_num_draft_tokens = int(num_draft_tokens.sum())
            total_num_logits = num_reqs + total_num_draft_tokens

            np.cumsum(
                num_draft_tokens + 1,
                out=self.input_buffers.cu_num_logits.np[1:num_reqs + 1],
            )
            cu_num_logits = self.input_buffers.cu_num_logits.copy_to_gpu(
                num_reqs + 1)

        # Block tables: num_kv_cache_groups x [num_reqs, max_num_blocks]
        block_tables = self.block_tables.gather_block_tables(idx_mapping_npu)

        # Get query_start_loc.
        np.cumsum(
            num_scheduled_tokens,
            out=self.input_buffers.query_start_loc.np[1:num_reqs + 1],
        )
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        self.input_buffers.query_start_loc.np[num_reqs + 1:] = num_tokens
        self.input_buffers.query_start_loc.copy_to_gpu()
        query_start_loc_gpu = self.input_buffers.query_start_loc.gpu[:
                                                                     num_reqs +
                                                                     1]
        query_start_loc_cpu = self.input_buffers.query_start_loc.cpu[:
                                                                     num_reqs +
                                                                     1]
        query_start_loc_np = self.input_buffers.query_start_loc.np[:num_reqs +
                                                                   1]

        # Get prefill tokens.
        prepare_prefill_inputs(
            self.input_buffers.input_ids,
            self.req_states.next_prefill_tokens,
            idx_mapping_npu,
            query_start_loc_gpu,
            # use prefill_token_ids.copy_to_gpu() because npu doesn't
            # support uva buffer.
            self.req_states.prefill_token_ids.copy_to_gpu(),
            self.req_states.prefill_len.gpu,
            self.req_states.num_computed_tokens,
        )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping_npu,
            query_start_loc_gpu,
            self.req_states.num_computed_tokens,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs]

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping_npu,
            self.req_states.last_sampled_tokens,
            query_start_loc_gpu,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
        )

        # Compute slot mappings: [num_kv_cache_groups, num_tokens]
        slot_mappings = self.block_tables.compute_slot_mappings(
            query_start_loc_gpu, self.input_buffers.positions[:num_tokens])

        # Layer name -> attention metadata.
        # TODO(Ronald1995): try to add a new method `build_attn_metadata` in
        # vllm gpu_model_runner_v2, maybe we don't overwrite `prepare_inputs`
        # method like this.
        attn_metadata = build_attn_metadata(
            attn_metadata_builders=self.attn_metadata_builders,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=self.input_buffers.seq_lens,
            seq_lens_np=self.input_buffers.seq_lens_np,
            num_computed_tokens_cpu=self.req_states.
            num_computed_tokens_cpu[idx_mapping_cpu],
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
            attn_state=attn_state,
        )

        input_ids = self.input_buffers.input_ids[:num_tokens_after_padding]
        positions = self.input_buffers.positions[:num_tokens_after_padding]
        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            idx_mapping=idx_mapping_npu,
            idx_mapping_np=idx_mapping_np,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            query_start_loc=query_start_loc_gpu,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_np=self.input_buffers.seq_lens_np,
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
        )

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
                self.req_states.num_computed_tokens,
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
            self.req_states.num_computed_tokens_cpu[
                req_index] = self.num_computed_tokens_cpu[req_index]

        # update seq_lens_cpu
        for i, req_id in enumerate(req_ids):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens = self.req_states.num_computed_tokens_cpu[
                req_index]
            self.input_buffers.seq_lens_cpu[
                i] = num_computed_tokens + num_scheduled_tokens[req_id]

    def eplb_warmup(self):
        # TODO(Ronald1995): just define the method in case calling error in
        # worker, implement it in the future.
        pass
