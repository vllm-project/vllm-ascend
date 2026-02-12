# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/states.py
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

import torch
from vllm.v1.worker.gpu.states import RequestState


class AscendRequestState(RequestState):
    """Request state for Ascend NPUs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
    ):
        super().__init__(
            max_num_reqs,
            max_model_len,
            max_num_batched_tokens,
            num_speculative_steps,
            vocab_size,
            device,
        )
        # vllm gpu_model_runner_v2 deprecate the seqs_lens_cpu attribute,
        # because they think most attention backends do not need it.
        # However, Ascend attention backend muse uses seqs_lens_cpu,
        # so we keep num_computed_tokens_cpu here, seq_lens_cpu need to be
        # calculated by num_computed_tokens_cpu + decode_token_per_req outside.
        self.num_computed_tokens_cpu: torch.Tensor = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device="cpu",
        )

    def add_request(
        self,
        req_id,
        prompt_len,
        prefill_token_ids,
        num_computed_tokens,
    ):
        super().add_request(
            req_id,
            prompt_len,
            prefill_token_ids,
            num_computed_tokens,
        )
        req_idx = self.req_id_to_index[req_id]
        self.num_computed_tokens_cpu[req_idx] = num_computed_tokens
