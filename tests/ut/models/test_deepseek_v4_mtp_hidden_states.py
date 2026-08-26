# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import torch.nn as nn

from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4ForCausalLM


def _make_target(mtp_hidden_buffer: torch.Tensor | None) -> AscendDeepseekV4ForCausalLM:
    target = AscendDeepseekV4ForCausalLM.__new__(AscendDeepseekV4ForCausalLM)
    nn.Module.__init__(target)
    target.model = nn.Module()
    target.model._mtp_hidden_buffer = mtp_hidden_buffer
    target._register_mtp_target_hidden_states_hook()
    return target


def test_mtp_target_hidden_states_hook_is_hidden_without_buffer() -> None:
    target = _make_target(None)

    assert not hasattr(target, "get_mtp_target_hidden_states")


def test_mtp_target_hidden_states_hook_returns_allocated_buffer() -> None:
    hidden_buffer = torch.empty(4, 8)
    target = _make_target(hidden_buffer)

    assert hasattr(target, "get_mtp_target_hidden_states")
    assert target.get_mtp_target_hidden_states() is hidden_buffer
