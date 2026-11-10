#
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

"""Backend-facing contracts for the context-parallel attention state.

``AscendPCPAttentionContext`` was historically defined in the worker PCP
manager (``vllm_ascend/worker/v2/pcp_manager.py``). It now lives in the
attention layer so that attention code never needs a runtime dependency on
worker modules; the worker module re-exports it for backward compatibility.
The :class:`PCPAttentionContext` protocol documents the fields that attention
backends consume without importing the concrete worker dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.input_batch import AscendInputBatch


@runtime_checkable
class PCPAttentionContext(Protocol):
    """Backend-facing global PCP view for one attention step, implemented by the worker PCP manager."""

    global_batch: Any
    global_block_tables: tuple[torch.Tensor, ...]
    global_slot_mappings: torch.Tensor
    hidden_restore_idx: torch.Tensor
    padded_gather_idx: torch.Tensor | None
    gathered_kv_write_mask: torch.Tensor | None
    global_block_table_num_blocks: torch.Tensor | None
    local_to_global_req_indices: tuple[int, ...] | None
    local_token_indices: torch.Tensor | None


@dataclass(frozen=True)
class AscendPCPAttentionContext:
    """Canonical global PCP view for one attention step."""

    # The global batch and its associated metadata, used to build DSA attention metadata.
    global_batch: AscendInputBatch
    global_block_tables: tuple[torch.Tensor, ...]
    global_slot_mappings: torch.Tensor
    hidden_restore_idx: torch.Tensor
    padded_gather_idx: torch.Tensor | None = None
    gathered_kv_write_mask: torch.Tensor | None = None
    # Device snapshot of allocated kernel-block counts in global request order.
    global_block_table_num_blocks: torch.Tensor | None = None
    # Original batch row for each local request, in upstream segment order.
    local_to_global_req_indices: tuple[int, ...] | None = None
    # Actual local tokens in the unpartitioned batch, excluding padding.
    local_token_indices: torch.Tensor | None = None
