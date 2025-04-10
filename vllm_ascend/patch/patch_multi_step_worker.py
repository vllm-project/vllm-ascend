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

from typing import List, Set, Tuple

import torch
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.multi_step_worker import MultiStepWorker

from vllm_ascend.worker.draft_model_runner import TP1DraftModelRunner


def sampler_output(
    self,
    execute_model_req: ExecuteModelRequest,
    sample_len: int,
    seq_ids_with_bonus_token_in_last_step: Set[int],
) -> Tuple[List[SamplerOutput], bool]:
    """Run the model forward pass sample_len times. Returns the list of
    sampler output, one per model forward pass, along with indicator of
    whether torch tensor in sampler output need to be transposed in latter
    sampler_output_to_torch logic.

    For multi step worker, this indicator shall be True.
    """
    self._raise_if_unsupported(execute_model_req)
    # Expand the batch for sequences with a bonus token.
    # Perform a forward pass on the expanded batch and filter the
    # response to retain only the original sequences' responses.
    expanded_request, indices_of_seq_with_bonus_tokens =\
        self._expand_execute_model_request(
            execute_model_req, seq_ids_with_bonus_token_in_last_step)

    # Run model sample_len times.
    model_outputs: List[SamplerOutput] = []

    # TODO: supports_gpu_multi_step is False in ASCEND
    if isinstance(self.model_runner, TP1DraftModelRunner) and \
        self.model_runner.supports_gpu_multi_step(expanded_request):
        # Here we run the draft_model_runner with multi-step prepare
        # on the GPU directly
        expanded_request.num_steps = sample_len
        self.model_runner.set_indices_of_seq_with_bonus_tokens(
            indices_of_seq_with_bonus_tokens)
        model_outputs = self.execute_model(execute_model_req=expanded_request)
    else:
        # Here we run multi-step directly, with every step prepared
        # on the CPU.
        # TODO: Remove this branch once DraftModelRunner supports TP>1
        # and other restrictions that are part of DraftModelRunner's
        # supports_gpu_multi_step(..)
        for _ in range(sample_len):
            model_output: List[SamplerOutput] = self.worker.execute_model(
                execute_model_req=expanded_request)
            assert (len(model_output) == 1
                    ), "composing multistep workers not supported"
            model_output = model_output[0]

            self._append_new_tokens(model_output,
                                    expanded_request.seq_group_metadata_list,
                                    indices_of_seq_with_bonus_tokens)
            model_outputs.append(model_output)

    # move indices to device to avoid stream sync
    indices_of_seq_with_bonus_tokens = torch.tensor(
        indices_of_seq_with_bonus_tokens, device=self.device)
    filtered_model_outputs = self._filter_model_output(
        model_outputs, indices_of_seq_with_bonus_tokens)
    return filtered_model_outputs, True


MultiStepWorker.sampler_output = torch.inference_mode()(sampler_output)
