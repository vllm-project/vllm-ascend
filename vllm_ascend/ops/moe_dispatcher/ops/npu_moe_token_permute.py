# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import torch



__all__ = ["npu_moe_token_permute"]

from vllm_ascend.ops.moe_dispatcher.ops.op_builder.npu_moe_token_permute_builder import MoeTokenPermuteOpBuilder

moe_token_permute_op_builder = MoeTokenPermuteOpBuilder()


def npu_moe_token_permute(
        tokens: torch.Tensor,
        indices: torch.Tensor,
        num_out_tokens: int = None,
        padded_mode: bool = False
):
    moe_token_permute_ops = moe_token_permute_op_builder.load()
    return moe_token_permute_ops.npu_moe_token_permute(tokens, indices, num_out_tokens, padded_mode)
