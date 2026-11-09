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
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    _combine_sampled_and_draft_tokens_kernel,
)

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.rotary_embedding import update_cos_sin


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

        # Logical query token count, scoped to _dummy_run before graph padding.
        # None keeps the default layout for other dummy/capture paths.
        self.dummy_num_tokens: int | None = None
        # CPP latency profiling must model one real prefill request. The generic
        # MRV2 dummy path otherwise describes every dummy batch as decode-only.
        self.profile_cpp: bool = False
        self.logits_indices_workspace: torch.Tensor | None = None

    def bind_logits_indices_workspace(self) -> None:
        """Give the target its own output storage before capture or execution."""
        if self.logits_indices_workspace is not None:
            raise RuntimeError("Logits indices workspace is already bound")
        self.logits_indices_workspace = torch.empty(self.max_num_tokens, dtype=torch.int64, device=self.device)

    def combine_sampled_and_draft_tokens(
        self,
        input_ids: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        prefill_len: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_logits: int,
        num_new_sampled_tokens: int = 1,
    ) -> torch.Tensor:
        """Write current values with Core's kernel into this target-owned prefix."""
        assert num_new_sampled_tokens in (0, 1), f"num_new_sampled_tokens must be 0 or 1, got {num_new_sampled_tokens}"
        workspace = self.logits_indices_workspace
        assert workspace is not None, "Logits indices workspace must be bound before use"
        if input_ids.device != workspace.device:
            raise ValueError("Logits indices workspace must share the input device")
        if not 0 <= num_logits <= workspace.numel():
            raise ValueError("Logits indices exceed the target input capacity")
        logits_indices = workspace[:num_logits]
        _combine_sampled_and_draft_tokens_kernel[(idx_mapping.shape[0],)](
            input_ids,
            idx_mapping,
            last_sampled_tokens,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            draft_tokens.stride(0),
            cu_num_logits,
            logits_indices,
            NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
            BLOCK_SIZE=triton.next_power_of_2(draft_tokens.shape[-1] + num_new_sampled_tokens),
        )
        return logits_indices


@dataclass
class AscendInputBatch(InputBatch):
    """Input batch for Ascend NPUs."""

    # Create seq_lens_np.
    # npu's attention backend still needs seq_lens on CPU side.
    # InputBatch gained max_query_len default field,
    # requiring the child's first field to also have a default.
    seq_lens_np: np.ndarray = None  # type: ignore[assignment, no-redef]
    # attn_state is used to build attention metadata.
    attn_state: AscendAttentionState | None = None
    is_dummy: bool = False
    is_cpp_profile: bool = False

    @staticmethod
    def _prepare_cpp_profile_dummy(
        input_batch: InputBatch,
        input_buffers: AscendInputBuffers,
        num_tokens: int,
    ) -> None:
        """Configure a generic dummy as one fresh CPP prefill request."""
        # CPP predicts prefill latency, so its synthetic request must carry
        # the same metadata as a fresh prompt. In particular, PCP uses
        # is_prefilling_np to decide whether and how to shard the sequence.
        input_batch.is_prefilling_np[:] = True
        input_batch.has_prefill = True
        input_batch.prefill_len_np[:] = input_batch.num_scheduled_tokens
        input_batch.num_computed_tokens_np[:] = 0
        input_batch.num_computed_prefill_tokens_np[:] = 0
        input_buffers.is_padding[:num_tokens].fill_(False)
        for start, end in zip(
            input_batch.query_start_loc_np[:-1],
            input_batch.query_start_loc_np[1:],
        ):
            input_buffers.positions[start:end].copy_(
                torch.arange(
                    end - start,
                    dtype=input_buffers.positions.dtype,
                    device=input_buffers.device,
                )
            )

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: AscendInputBuffers,
        max_query_len: int | None = None,
        is_padding: bool = True,
    ) -> "AscendInputBatch":
        """Override the make_dummy method to calculate seq_lens_np."""
        # The incoming count includes graph padding. Save it for the model input
        # views below, while building query metadata from the original scheduled
        # count used for DP synchronization. Only _dummy_run supplies that override.
        # TODO: Remove this workaround once upstream dummy-batch preparation
        # preserves logical query lengths separately from graph padding.
        num_tokens_after_padding = num_tokens
        if input_buffers.dummy_num_tokens is not None:
            # PIECEWISE pads model inputs without adding scheduled queries.
            num_tokens = input_buffers.dummy_num_tokens
            assert num_tokens <= num_tokens_after_padding
        input_batch = InputBatch.make_dummy(
            num_reqs,
            num_tokens,
            input_buffers,
            max_query_len=max_query_len,
            is_padding=is_padding,
        )
        if num_tokens_after_padding != num_tokens:
            input_batch.num_tokens_after_padding = num_tokens_after_padding
            input_buffers.input_ids[num_tokens:num_tokens_after_padding].zero_()
            input_buffers.positions[num_tokens:num_tokens_after_padding].zero_()
            input_buffers.is_padding[num_tokens:num_tokens_after_padding].fill_(True)
            input_batch.input_ids = input_buffers.input_ids[:num_tokens_after_padding]
            input_batch.positions = input_buffers.positions[:num_tokens_after_padding]
            input_batch.is_padding = input_buffers.is_padding[:num_tokens_after_padding]
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        input_buffers.seq_lens_np[: num_reqs - num_extra] = base_tokens
        input_buffers.seq_lens_np[num_reqs - num_extra : num_reqs] = base_tokens + 1
        input_buffers.seq_lens_np[num_reqs:] = 0
        seq_lens_np = input_buffers.seq_lens_np[:num_reqs]
        is_cpp_profile = input_buffers.profile_cpp
        if is_cpp_profile:
            cls._prepare_cpp_profile_dummy(input_batch, input_buffers, num_tokens)
        update_cos_sin(input_batch.positions)
        base_fields = {field.name: getattr(input_batch, field.name) for field in fields(InputBatch)}
        return cls(
            **base_fields,
            seq_lens_np=seq_lens_np,
            attn_state=(AscendAttentionState.PrefillNoCache if is_cpp_profile else AscendAttentionState.DecodeOnly),
            is_dummy=True,
            is_cpp_profile=is_cpp_profile,
        )
