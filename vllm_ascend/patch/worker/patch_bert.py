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

import torch
from torch import nn

from vllm.model_executor.models import bert

# aclgraph does not support shift operator for now
# TODO: revert me when aclgraph supports shift operator
TOKEN_TYPE_MULTIPLIER = 1 << 30
TOKEN_MASK = TOKEN_TYPE_MULTIPLIER - 1


def _encode_token_type_ids(input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> None:
    # input_ids can be padded to the right
    input_ids[: token_type_ids.shape[0]].bitwise_or_(token_type_ids * TOKEN_TYPE_MULTIPLIER)


def _decode_token_type_ids(input_ids: torch.Tensor) -> torch.Tensor:
    token_type_ids = input_ids // TOKEN_TYPE_MULTIPLIER

    input_ids.bitwise_and_(TOKEN_MASK)

    return token_type_ids

class AscendBertOutput(nn.Module):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = torch.ops._C_ascend.add_layer_norm(
            hidden_states, input_tensor, self.LayerNorm.weight, self.LayerNorm.bias, self.LayerNorm.eps
        )[0]
        return hidden_states


class AscendBertSelfOutput(nn.Module):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = torch.ops._C_ascend.add_layer_norm(
            hidden_states, input_tensor, self.LayerNorm.weight, self.LayerNorm.bias, self.LayerNorm.eps
        )[0]
        return hidden_states

class AscendBertIntermediate(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.intermediate_act_fn, nn.GELU):
            hidden_states = torch.ops._C_ascend.matmul_gelu(hidden_states, self.dense.weight, self.dense.bias)
        else:
            hidden_states, _ = self.dense(hidden_states)
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

bert._encode_token_type_ids = _encode_token_type_ids
bert._decode_token_type_ids = _decode_token_type_ids

bert.BertOutput.forward = AscendBertOutput.forward
bert.BertSelfOutput.forward = AscendBertSelfOutput.forward
bert.BertIntermediate.forward = AscendBertIntermediate.forward

