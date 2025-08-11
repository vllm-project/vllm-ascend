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
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

from typing import Optional

import torch
from vllm.config import VllmConfig

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class NPUTorchairModelRunner(NPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

    def _get_forward_metadata_across_dp_and_pad(
            self, num_tokens: int, with_prefill: bool, enable_dbo: bool
    ) -> tuple[int, Optional[torch.Tensor], bool, bool]:
        if self.dp_size == 1:
            if not with_prefill:
                maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                    num_tokens)
                return maybe_padded_num_tokens, None, with_prefill, enable_dbo
            return num_tokens, None, with_prefill, enable_dbo

        num_tokens_across_dp, with_prefill, enable_dbo = self._get_forward_metadata_across_dp(
            num_tokens, with_prefill, enable_dbo)

        if not with_prefill:
            max_num_token = num_tokens_across_dp.max().item()
            maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                max_num_token)
            num_tokens_across_dp = torch.full((self.dp_size, ),
                                              maybe_padded_num_tokens,
                                              dtype=torch.int32,
                                              device="cpu")
        else:
            maybe_padded_num_tokens = num_tokens

        return maybe_padded_num_tokens, num_tokens_across_dp, with_prefill, enable_dbo
