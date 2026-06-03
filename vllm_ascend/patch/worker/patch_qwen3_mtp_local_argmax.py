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
"""Enable vocab-parallel local argmax for Qwen3 MTP draft sampling."""

import torch
from torch import nn
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP


def _qwen_mtp_get_top_tokens(
    self: nn.Module,
    hidden_states: torch.Tensor,
    spec_step_idx: int = 0,
) -> torch.Tensor:
    del spec_step_idx
    return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)


if not hasattr(Qwen3_5MTP, "get_top_tokens"):
    Qwen3_5MTP.get_top_tokens = _qwen_mtp_get_top_tokens

if not hasattr(Qwen3NextMTP, "get_top_tokens"):
    Qwen3NextMTP.get_top_tokens = _qwen_mtp_get_top_tokens
