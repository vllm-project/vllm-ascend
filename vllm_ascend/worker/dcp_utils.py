#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens


class DCPManager:
    """Prepare NPU host lengths and decode flags for DCP drafts."""

    def __init__(
        self,
        dcp_world_size: int,
        dcp_rank: int,
        vllm_config: VllmConfig,
    ) -> None:
        self.dcp_world_size = dcp_world_size
        self.dcp_world_rank = dcp_rank
        self.vllm_config = vllm_config

    def prepare_draft_dcp_metadata_inputs(
        self,
        target_seq_lens_cpu: torch.Tensor,
        is_prefilling: torch.Tensor,
        num_reqs: int,
        num_reqs_padded: int,
        step: int,
        max_model_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare common CPU lengths and decode flags for a DCP draft.

        MTP passes its draft step; parallel block drafters such as DSpark pass
        the full query-block width. SFA shares this preparation but consumes
        device-local lengths in its attention backend.
        """
        # Preserve the existing CPU progression without rejection correction.
        seq_lens_cpu = torch.clamp(target_seq_lens_cpu[:num_reqs_padded] + step, max=max_model_len)
        seq_lens_cpu[num_reqs:].fill_(0)
        draft_is_prefilling = torch.zeros(num_reqs_padded, dtype=torch.bool)
        if step <= 0:
            draft_is_prefilling[:num_reqs].copy_(is_prefilling[:num_reqs])
            return seq_lens_cpu, draft_is_prefilling

        return seq_lens_cpu, draft_is_prefilling

    def prepare_dcp_local_seq_lens_cpu(self, seq_lens_cpu: torch.Tensor) -> torch.Tensor:
        """Partition the supplied global CPU lengths for this DCP rank."""
        return get_dcp_local_seq_lens(
            seq_lens_cpu,
            dcp_size=self.dcp_world_size,
            dcp_rank=self.dcp_world_rank,
            cp_kv_cache_interleave_size=self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
        )
