# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/input_batch.py
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
from dataclasses import dataclass, fields

import numpy as np
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import vllm_version_is


class AscendInputBuffers(InputBuffers):
    """Input buffers for Ascend NPUs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        super().__init__(
            max_num_reqs,
            max_num_tokens,
            device,
        )
        del self.query_start_loc

        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        self.query_start_loc: torch.Tensor = torch.zeros(
            max_num_reqs + 2,
            dtype=torch.int32,
            device=device,
        )

        # Create seq_lens_cpu and seq_lens_np.
        # npu's attention backend still needs seq_lens on CPU side.
        self.seq_lens_cpu: torch.Tensor = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device="cpu",
        )
        # seq_len_np and seq_lens_cpu share the same memory.
        # define seq_lens_np for easier calculation with numpy.
        self.seq_lens_np: np.ndarray = self.seq_lens_cpu.numpy()


@dataclass(kw_only=True)
class AscendInputBatch(InputBatch):
    """Input batch for Ascend NPUs."""

    # Create seq_lens_np.
    # npu's attention backend still needs seq_lens on CPU side.
    if vllm_version_is("0.27.1"):
        seq_lens_np: np.ndarray
    else:
        # main (post-0.27.1): InputBatch gained max_query_len default field,
        # requiring the child's first field to also have a default.
        seq_lens_np: np.ndarray = None  # type: ignore[assignment, no-redef]
    # attn_state is used to build attention metadata.
    attn_state: AscendAttentionState | None = None
    is_dummy: bool = False

    if vllm_version_is("0.27.1"):

        @classmethod
        def make_dummy(
            cls,
            num_reqs: int,
            num_tokens: int,
            input_buffers: AscendInputBuffers,
        ) -> "AscendInputBatch":
            """Override the make_dummy method to calculate seq_lens_np."""
            input_batch = InputBatch.make_dummy(
                num_reqs,
                num_tokens,
                input_buffers,
            )
            base_tokens = num_tokens // num_reqs
            num_extra = num_tokens % num_reqs
            input_buffers.seq_lens_np[: num_reqs - num_extra] = base_tokens
            input_buffers.seq_lens_np[num_reqs - num_extra : num_reqs] = base_tokens + 1
            input_buffers.seq_lens_np[num_reqs:] = 0
            seq_lens_np = input_buffers.seq_lens_np[:num_reqs]
            update_cos_sin(input_batch.positions)
            base_fields = {field.name: getattr(input_batch, field.name) for field in fields(InputBatch)}
            return cls(
                **base_fields,
                seq_lens_np=seq_lens_np,
                attn_state=AscendAttentionState.DecodeOnly,
                is_dummy=True,
            )

    else:

        @classmethod
        def make_dummy(
            cls,
            num_reqs: int,
            num_tokens: int,
            input_buffers: AscendInputBuffers,
            max_query_len: int | None = None,
        ) -> "AscendInputBatch":
            """Override the make_dummy method to calculate seq_lens_np."""
            input_batch = InputBatch.make_dummy(
                num_reqs,
                num_tokens,
                input_buffers,
                max_query_len=max_query_len,
            )
            base_tokens = num_tokens // num_reqs
            num_extra = num_tokens % num_reqs
            input_buffers.seq_lens_np[: num_reqs - num_extra] = base_tokens
            input_buffers.seq_lens_np[num_reqs - num_extra : num_reqs] = base_tokens + 1
            input_buffers.seq_lens_np[num_reqs:] = 0
            seq_lens_np = input_buffers.seq_lens_np[:num_reqs]
            update_cos_sin(input_batch.positions)
            base_fields = {field.name: getattr(input_batch, field.name) for field in fields(InputBatch)}
            return cls(
                **base_fields,
                seq_lens_np=seq_lens_np,
                attn_state=AscendAttentionState.DecodeOnly,
                is_dummy=True,
            )
