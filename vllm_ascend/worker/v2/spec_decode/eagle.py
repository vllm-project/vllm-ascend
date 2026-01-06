# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/eagle.py.
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
from vllm.v1.worker.gpu.spec_decode.eagle import EagleSpeculator
from vllm.config import VllmConfig
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers


class AscendEagleSpeculator(EagleSpeculator):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.main_model_attn_metadata = None

    def propose(
        self,
        input_batch,
        sampling_metadata,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
    ):
        """Override GPU EagleSpeculator.propose for Ascend NPUs,
        because npu attention backends need seq_lens_cpu to work.
        """
        seq_lens_cpu = input_batch.attn_metadata.seq_lens_cpu
        self.input_buffers.seq_lens_cpu = seq_lens_cpu.clone()
        return super().propose(
            input_batch,
            sampling_metadata,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
        )

    def generate_draft(self, num_reqs, attn_metadata, num_tokens_across_dp):
        """Override GPU EagleSpeculator.generate_draft for Ascend NPUs, because
        npu attention backends need seq_lens_cpu to work.
        """
        attn_metadata.seq_lens_cpu = self.input_buffers.seq_lens_cpu
        return super().generate_draft(
            num_reqs,
            attn_metadata,
            num_tokens_across_dp,
        )
