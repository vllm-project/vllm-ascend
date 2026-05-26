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

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class V1SamplingContext:
    """Per-step sampling context bridging v1 data to the new sampler pipeline.

    Encapsulates the mapping tensors that the state-driven sampler expects,
    constructed from v1 model-runner inputs each step.
    """

    # [num_logits] - maps each logits row to its request index
    expanded_idx_mapping: torch.Tensor

    # [num_logits] (numpy) - same mapping on CPU, for numpy-based early-exit checks
    idx_mapping_np: np.ndarray

    # [num_logits] - position of each token in its request's sequence
    pos: torch.Tensor

    # [num_logits] - input token ID at each logits position
    input_ids: torch.Tensor

    # [num_logits] - position within the expanded batch (0 for base path)
    expanded_local_pos: torch.Tensor

    # [num_reqs+1] (numpy) - cumulative logits count per request for logprobs
    cu_num_logits_np: np.ndarray | None

    # Whether logits rows are expanded (speculative decoding)
    expanded_logits: bool

    # Number of active requests
    num_reqs: int

    # Active request IDs ordered by request index. Used for per-request state.
    req_ids: tuple[str, ...] | None = None

    @property
    def num_logits(self) -> int:
        return int(self.expanded_idx_mapping.shape[0])

    @property
    def is_identity_request_mapping(self) -> bool:
        if self.num_logits != self.num_reqs:
            return False
        if self.num_logits == 0:
            return True
        return bool(np.array_equal(self.idx_mapping_np, np.arange(self.num_reqs, dtype=np.int32)))

    @property
    def num_logits_per_req_np(self) -> np.ndarray:
        return np.bincount(self.idx_mapping_np, minlength=self.num_reqs).astype(np.int32)

    @staticmethod
    def _is_grouped_by_request(idx_mapping_np: np.ndarray) -> bool:
        if idx_mapping_np.size <= 1:
            return True
        group_starts: np.ndarray = np.empty(idx_mapping_np.size, dtype=bool)
        group_starts[0] = True
        group_starts[1:] = idx_mapping_np[1:] != idx_mapping_np[:-1]
        group_ids = idx_mapping_np[group_starts]
        return np.unique(group_ids).size == group_ids.size

    @staticmethod
    def from_model_runner_inputs(
        num_reqs: int,
        positions_at_logits: torch.Tensor,
        input_ids_at_logits: torch.Tensor,
        req_indices_at_logits: torch.Tensor,
        device: torch.device,
        req_ids: tuple[str, ...] | None = None,
        expanded_local_pos: torch.Tensor | None = None,
        cu_num_logits_np: np.ndarray | None = None,
        idx_mapping_np: np.ndarray | None = None,
    ) -> "V1SamplingContext":
        """Construct sampling context for v1 logits rows.

        The caller must select positions/input IDs at the same rows used to
        compute logits. ``req_indices_at_logits`` maps each logits row back to
        its active request index.
        """
        num_logits = int(req_indices_at_logits.shape[0])
        if int(positions_at_logits.shape[0]) != num_logits:
            raise ValueError("positions_at_logits must have one entry per logits row")
        if int(input_ids_at_logits.shape[0]) != num_logits:
            raise ValueError("input_ids_at_logits must have one entry per logits row")
        if req_ids is not None and len(req_ids) != num_reqs:
            raise ValueError("req_ids length must match num_reqs")

        if idx_mapping_np is None:
            expanded_idx_mapping = req_indices_at_logits.to(device=device, dtype=torch.int32)
            idx_mapping = expanded_idx_mapping.detach().cpu().numpy().astype(np.int32)
        else:
            idx_mapping = np.asarray(idx_mapping_np, dtype=np.int32)
            if int(idx_mapping.shape[0]) != num_logits:
                raise ValueError("idx_mapping_np must have one entry per logits row")
            if req_indices_at_logits.device.type == "cpu":
                req_indices_np = req_indices_at_logits.detach().numpy().astype(np.int32, copy=False)
                if not np.array_equal(idx_mapping, req_indices_np):
                    raise ValueError("idx_mapping_np must match req_indices_at_logits")
            expanded_idx_mapping = req_indices_at_logits.to(device=device, dtype=torch.int32)
        if idx_mapping.size and (idx_mapping.min() < 0 or idx_mapping.max() >= num_reqs):
            raise ValueError("req_indices_at_logits contains an out-of-range request index")

        expanded_logits = num_logits != num_reqs or not np.array_equal(idx_mapping, np.arange(num_reqs, dtype=np.int32))
        if expanded_logits and not V1SamplingContext._is_grouped_by_request(idx_mapping):
            raise ValueError("expanded logits rows must be grouped by request")

        if expanded_local_pos is None:
            if not expanded_logits:
                expanded_local_pos = torch.zeros(num_logits, device=device, dtype=torch.int64)
            elif num_logits == 0:
                expanded_local_pos = torch.empty(0, device=device, dtype=torch.int64)
            else:
                group_starts_np: np.ndarray = np.empty(num_logits, dtype=bool)
                group_starts_np[0] = True
                group_starts_np[1:] = idx_mapping[1:] != idx_mapping[:-1]
                group_start_rows = np.flatnonzero(group_starts_np)
                group_lengths = np.diff(np.append(group_start_rows, num_logits))
                local_pos_np = np.arange(num_logits, dtype=np.int64) - np.repeat(group_start_rows, group_lengths)
                expanded_local_pos = torch.from_numpy(local_pos_np).to(device=device)
        else:
            expanded_local_pos = expanded_local_pos.to(device=device, dtype=torch.int64)

        if cu_num_logits_np is None and expanded_logits:
            cu_num_logits_np = np.concatenate(
                (np.array([0], dtype=np.int32), np.cumsum(np.bincount(idx_mapping, minlength=num_reqs)))
            ).astype(np.int32)

        return V1SamplingContext(
            expanded_idx_mapping=expanded_idx_mapping,
            idx_mapping_np=idx_mapping,
            pos=positions_at_logits,
            input_ids=input_ids_at_logits,
            expanded_local_pos=expanded_local_pos,
            cu_num_logits_np=cu_num_logits_np,
            expanded_logits=expanded_logits,
            num_reqs=num_reqs,
            req_ids=req_ids,
        )
