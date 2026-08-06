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

import torch
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather

from vllm_ascend.utils import enable_sp, is_moe_model

FLASHCOMM_DENSE_TOKEN_THRESHOLD = 1000


def _flashcomm_enabled(vllm_config: VllmConfig, num_tokens: int) -> bool:
    return enable_sp(vllm_config) and (is_moe_model(vllm_config) or num_tokens > FLASHCOMM_DENSE_TOKEN_THRESHOLD)


def _all_gather_hidden_states(hidden_states: torch.Tensor, num_tokens: int):
    hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
    return hidden_states[:num_tokens]


def _all_gather_hidden_states_and_aux(hidden_states, num_tokens: int):
    if isinstance(hidden_states, tuple):
        return (
            _all_gather_hidden_states(hidden_states[0], num_tokens),
            [_all_gather_hidden_states(aux_hidden_state, num_tokens) for aux_hidden_state in hidden_states[1]],
        )
    return _all_gather_hidden_states(hidden_states, num_tokens)
