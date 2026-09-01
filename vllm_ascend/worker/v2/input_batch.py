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
from zlib import adler32

import numpy as np
import torch
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
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
        enable_sparse_kv_offload: bool = False,
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

        self.offload_req_ids: torch.Tensor | None = None
        self.offload_token_to_req: torch.Tensor | None = None
        if enable_sparse_kv_offload:
            self.offload_req_ids = torch.zeros(
                max_num_reqs,
                dtype=torch.int64,
                device=device,
            )
            self.offload_token_to_req = torch.zeros(
                max_num_tokens,
                dtype=torch.int32,
                device=device,
            )


@dataclass
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
    offload_req_ids: torch.Tensor | None = None
    offload_token_to_req: torch.Tensor | None = None

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
            batch = cls(
                **base_fields,
                seq_lens_np=seq_lens_np,
                attn_state=AscendAttentionState.DecodeOnly,
                is_dummy=True,
            )
            return prepare_sparse_kv_offload_metadata(batch, input_buffers)

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
            batch = cls(
                **base_fields,
                seq_lens_np=seq_lens_np,
                attn_state=AscendAttentionState.DecodeOnly,
                is_dummy=True,
            )
            return prepare_sparse_kv_offload_metadata(batch, input_buffers)


def prepare_sparse_kv_offload_metadata(
    input_batch: AscendInputBatch,
    input_buffers: AscendInputBuffers,
) -> AscendInputBatch:
    """Stage sparse-offload request metadata into MRV2 input buffers."""
    req_ids_buffer = input_buffers.offload_req_ids
    token_to_req_buffer = input_buffers.offload_token_to_req
    if req_ids_buffer is None or token_to_req_buffer is None:
        input_batch.offload_req_ids = None
        input_batch.offload_token_to_req = None
        return input_batch

    num_reqs = input_batch.num_reqs
    num_reqs_padded = input_batch.num_reqs_after_padding
    num_tokens = input_batch.num_tokens
    num_tokens_padded = input_batch.num_tokens_after_padding
    if len(input_batch.req_ids) < num_reqs:
        raise RuntimeError(
            "KV offload request metadata is shorter than the scheduled batch: "
            f"metadata={len(input_batch.req_ids)}, requests={num_reqs}"
        )

    req_ids_np = np.zeros(num_reqs_padded, dtype=np.int64)
    req_ids_np[:num_reqs] = np.fromiter(
        (
            adler32(req_id.encode("utf-8"))
            for req_id in input_batch.req_ids[:num_reqs]
        ),
        dtype=np.int64,
        count=num_reqs,
    )

    query_lens = np.diff(
        input_batch.query_start_loc_np[: num_reqs + 1]
    ).astype(np.int32, copy=False)
    token_to_req = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    if token_to_req.shape[0] < num_tokens:
        raise RuntimeError(
            "KV offload token_to_req metadata is shorter than the scheduled "
            f"token batch: metadata={token_to_req.shape[0]}, tokens={num_tokens}"
        )
    token_to_req_np = np.zeros(num_tokens_padded, dtype=np.int32)
    token_to_req_np[:num_tokens] = token_to_req[:num_tokens]

    async_copy_to_gpu(req_ids_np, out=req_ids_buffer[:num_reqs_padded])
    async_copy_to_gpu(token_to_req_np, out=token_to_req_buffer[:num_tokens_padded])
    input_batch.offload_req_ids = req_ids_buffer[:num_reqs_padded]
    input_batch.offload_token_to_req = token_to_req_buffer[:num_tokens_padded]
    return input_batch
