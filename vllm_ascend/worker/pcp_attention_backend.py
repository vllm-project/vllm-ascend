#
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

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from vllm_ascend.attention.utils import AscendPrefillContextParallelMetadata


@dataclass
class PCPMetadataBuildContext:
    """Inputs required by a PCP attention backend to build its metadata."""

    query_lens: torch.Tensor
    num_decode_reqs: int
    pcp_world_size: int
    pcp_world_rank: int
    device: torch.device | str
    model_config: Any
    use_mla: bool
    pcp_use_hybrid_attn: bool


@dataclass
class PCPBackendMetadata:
    """Backend-owned tensors and kwargs consumed by attention implementations."""

    q_head_idx_tensor: torch.Tensor | None = None
    q_tail_idx_tensor: torch.Tensor | None = None
    q_full_idx: torch.Tensor | None = None
    kv_idx_names: dict[str, torch.Tensor] = field(default_factory=dict)
    attn_chunk_seqlens: torch.Tensor | None = None
    extra_long_seq_kwargs: dict[str, list[int]] = field(default_factory=dict)


class PCPAttentionBackend(Protocol):
    """Adapter interface between PCP scheduling layout and attention metadata."""

    name: str

    def build_metadata(self, ctx: PCPMetadataBuildContext) -> PCPBackendMetadata: ...

    def apply_metadata(
        self,
        long_seq_metadata: AscendPrefillContextParallelMetadata,
        metadata: PCPBackendMetadata,
    ) -> None: ...


def select_pcp_attention_backend(vllm_config: Any) -> PCPAttentionBackend:
    """Select the PCP attention backend for the current model/config."""
    from vllm_ascend.worker.pcp_metadata import DUAL_CHUNK_PCP_ATTENTION_BACKEND

    return DUAL_CHUNK_PCP_ATTENTION_BACKEND
