# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/default.py
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
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states import interface

from vllm_ascend.worker.encoder_acl_graph import EncoderAclGraphManager
from vllm_ascend.worker.v2.model_states import init_asecnd_model_state

_original_gather_mm_embeddings = interface.ModelState.gather_mm_embeddings


def _gather_mm_embeddings(
    self: interface.ModelState, input_batch: InputBatch, draft_lookahead: int = 0
) -> tuple[list[torch.Tensor], torch.Tensor]:
    if input_batch.num_tokens == 0 and getattr(self, "pcp_manager", None) is not None:
        return [], torch.empty(0, dtype=torch.bool, device="cpu")
    return _original_gather_mm_embeddings(self, input_batch, draft_lookahead)


# prepare_attn in AscendModelState is different from vllm,
# we need to override init_model_state.
model_runner.init_model_state = init_asecnd_model_state
# ModelState imports the manager class into its own module namespace. Replace
# that exact binding so MRV2 constructs an NPU-aware encoder graph manager.
interface.EncoderCudaGraphManager = EncoderAclGraphManager

# DefaultModelState calls ModelState.gather_mm_embeddings through super().
# TODO: Remove this patch once the paired vLLM includes the fix from vllm-project/vllm#64435.
interface.ModelState.gather_mm_embeddings = _gather_mm_embeddings
