#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from vllm-project/vllm/vllm/worker/gpu_input_batch.py
#

from typing import Optional, cast

import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.utils import copy_slice
from vllm.v1.worker.gpu_input_batch import InputBatch


class NpuInputBatch(InputBatch):

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        logitsprocs: Optional[LogitsProcessors] = None,
        is_pooling_model: bool = False,
        is_spec_decode: bool = False,
    ):
        """
        TODO: The NPUInputBatch is currently inherit to the InputBatch.
        we temporarily retain this class for Interface compatibility and will remove it later.
        """
        super().__init__(
            max_num_reqs,
            max_model_len,
            max_num_batched_tokens,
            device,
            pin_memory,
            vocab_size,
            block_sizes,
            logitsprocs,
            is_pooling_model,
            is_spec_decode,
        )

    def refresh_sampling_metadata(self):
        self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:
        num_reqs = self.num_reqs
        if not self.all_greedy:
            temperature = copy_slice(self.temperature_cpu_tensor,
                                     self.temperature, num_reqs)
        else:
            temperature = None
        if not self.no_top_p:
            copy_slice(self.top_p_cpu_tensor, self.top_p, num_reqs)
        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)
        if not self.no_min_p:
            copy_slice(self.min_p_cpu_tensor, self.min_p, num_reqs)

        if not self.no_penalties:
            # Since syncing these tensors is expensive only copy them
            # if necessary i.e. if there are requests which require
            # penalties to be applied during sampling.
            copy_slice(self.frequency_penalties_cpu_tensor,
                       self.frequency_penalties, num_reqs)
            copy_slice(self.presence_penalties_cpu_tensor,
                       self.presence_penalties, num_reqs)
            copy_slice(self.repetition_penalties_cpu_tensor,
                       self.repetition_penalties, num_reqs)

        needs_prompt_token_ids = (
            not self.no_penalties
            or self.logits_processing_needs_token_ids[:num_reqs].any())
        if needs_prompt_token_ids:
            # The prompt tokens are used only for applying penalties or
            # step pooling during the sampling/pooling process.
            # Hence copy these tensors only when there are requests which
            # need penalties/step_pooler to be applied.
            prompt_token_ids = self._make_prompt_token_ids_tensor()
        else:
            prompt_token_ids = None

        allowed_token_ids_mask: Optional[torch.Tensor] = None
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            copy_slice(self.allowed_token_ids_mask_cpu_tensor,
                       self.allowed_token_ids_mask, num_reqs)
            allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p[:num_reqs],
            top_k=None if self.no_top_k else self.top_k[:num_reqs],
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs],
            presence_penalties=self.presence_penalties[:num_reqs],
            repetition_penalties=self.repetition_penalties[:num_reqs],
            output_token_ids=cast(list[list[int]], self.req_output_token_ids),
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
            logitsprocs=self.logitsprocs,
        )

    def _make_prompt_token_ids_tensor(self) -> torch.Tensor:
        max_prompt_len = self.num_prompt_tokens[:self.num_reqs].max()
        prompt_token_ids_cpu_tensor = torch.empty(
            (self.num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[:self.
                                                 num_reqs, :max_prompt_len]
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        for i in range(self.num_reqs):
            prompt_token_ids[i, self.num_prompt_tokens[i]:] = self.vocab_size
        return prompt_token_ids_cpu_tensor.to(device=self.device,
                                              non_blocking=True)
