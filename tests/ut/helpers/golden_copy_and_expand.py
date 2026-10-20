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
"""CPU golden reference for ``npu_copy_and_expand_eagle_inputs``.

Adapted from ``tests/e2e/nightly/single_node/ops/singlecard_ops/test_copy_and_expand_eagle_inputs.py``.
"""

from __future__ import annotations

import numpy as np
import torch


def golden_copy_and_expand_eagle_inputs(
    target_token_ids: np.ndarray,
    target_positions: np.ndarray,
    next_token_ids: np.ndarray,
    query_start_loc: np.ndarray,
    query_end_loc: np.ndarray,
    padding_token_id: int,
    parallel_drafting_token_id: int,
    num_padding_slots: int,
    shift_input_ids: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_reqs = len(next_token_ids)

    total_draft_tokens = 0
    for r in range(num_reqs):
        qs = query_start_loc[r]
        nqs = query_start_loc[r + 1]
        qe = query_end_loc[r]
        num_rejected = max(nqs - qe - 1, 0)
        if shift_input_ids:
            num_valid = max(qe - qs, 0)
        else:
            num_valid = max(qe - qs + 1, 0)
        total_draft_tokens += num_valid + num_padding_slots + num_rejected

    out_ids = np.zeros(total_draft_tokens, dtype=np.int32)
    out_pos = np.zeros(total_draft_tokens, dtype=np.int32)
    out_rej = np.zeros(total_draft_tokens, dtype=np.int8)
    out_msk = np.zeros(total_draft_tokens, dtype=np.int8)
    out_nti = np.zeros(num_reqs * num_padding_slots, dtype=np.int32)
    total_input_tokens = len(target_token_ids)
    out_hsm = np.zeros(total_input_tokens, dtype=np.int32)

    for r in range(num_reqs):
        qs = query_start_loc[r]
        nqs = query_start_loc[r + 1]
        qe = query_end_loc[r]

        num_rejected = max(nqs - qe - 1, 0)

        if shift_input_ids:
            num_valid = max(qe - qs, 0)
            output_start = qs + r * (num_padding_slots - 1)
        else:
            num_valid = max(qe - qs + 1, 0)
            output_start = qs + r * num_padding_slots

        start_pos = target_positions[qs]
        next_token_id = next_token_ids[r]

        if shift_input_ids:
            read_start = qs + 1
            read_count = min(num_valid, total_input_tokens - read_start)
            if read_count < 0:
                read_count = 0
            for j in range(num_valid):
                idx = min(j, read_count - 1) if read_count > 0 else 0
                out_ids[output_start + j] = target_token_ids[read_start + idx] if read_count > 0 else 0
                out_pos[output_start + j] = start_pos + j
                out_rej[output_start + j] = 0
                out_msk[output_start + j] = 0
        else:
            num_input = nqs - qs
            for j in range(num_valid):
                idx = min(j, num_input - 1)
                out_ids[output_start + j] = target_token_ids[qs + idx]
                out_pos[output_start + j] = start_pos + j
                out_rej[output_start + j] = 0
                out_msk[output_start + j] = 0

        out_ids[output_start + num_valid] = next_token_id
        out_pos[output_start + num_valid] = start_pos + num_valid
        out_rej[output_start + num_valid] = 0
        out_msk[output_start + num_valid] = 0

        for k in range(1, num_padding_slots):
            j = num_valid + k
            out_ids[output_start + j] = parallel_drafting_token_id
            out_pos[output_start + j] = start_pos + j
            out_rej[output_start + j] = 0
            out_msk[output_start + j] = 1

        for k in range(num_rejected):
            j = num_valid + num_padding_slots + k
            out_ids[output_start + j] = padding_token_id
            out_pos[output_start + j] = 0
            out_rej[output_start + j] = 1
            out_msk[output_start + j] = 0

        for k in range(num_padding_slots):
            out_nti[r * num_padding_slots + k] = output_start + num_valid + k

        if shift_input_ids:
            num_input = nqs - qs
            for j in range(num_input):
                out_hsm[qs + j] = output_start + j

    return out_ids, out_pos, out_rej, out_msk, out_nti, out_hsm


def npu_copy_and_expand_eagle_inputs_stub(
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    next_token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_end_loc: torch.Tensor,
    padding_token_id: int,
    parallel_drafting_token_id: int,
    num_padding_slots_per_request: int,
    shift_input_ids: bool,
    total_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del total_draft_tokens  # computed by the golden reference
    device = target_token_ids.device
    out_ids, out_pos, out_rej, out_msk, out_nti, out_hsm = golden_copy_and_expand_eagle_inputs(
        target_token_ids.detach().cpu().numpy(),
        target_positions.detach().cpu().numpy(),
        next_token_ids.detach().cpu().numpy(),
        query_start_loc.detach().cpu().numpy(),
        query_end_loc.detach().cpu().numpy(),
        padding_token_id,
        parallel_drafting_token_id,
        num_padding_slots_per_request,
        shift_input_ids,
    )
    return (
        torch.from_numpy(out_ids).to(device=device),
        torch.from_numpy(out_pos).to(device=device),
        torch.from_numpy(out_rej).to(device=device),
        torch.from_numpy(out_msk).to(device=device),
        torch.from_numpy(out_nti).to(device=device),
        torch.from_numpy(out_hsm).to(device=device),
    )
