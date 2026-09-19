#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""Regression tests: drafter slot mapping with padded block_table rows.

Reproduces the draft_model + FULL-graph batch-shrink crash: the cad handed
to the drafter comes from AscendCommonAttentionMetadata.unpadded(), which
slices query_start_loc to the real request count while intentionally
keeping the FULL-graph-padded block_table (phantom rows included). Upstream
compute_new_slot_mapping derives batch_size from block_table rows, so the
disagreement crashes torch.repeat_interleave.
"""

from __future__ import annotations

import pytest
import torch
from vllm.v1.spec_decode.utils import compute_new_slot_mapping

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, align_block_table_for_slot_mapping

BLOCK_SIZE = 128
NUM_REAL_REQS = 5  # batch shrunk to 5 running requests
TOKENS_PER_REQ = 6  # K+1 = 6 verify tokens per request
PHANTOM_ROWS = 1  # 5*(K+1)=30 fell between buckets 24 and 36 -> pad to 36
NUM_NEW_TOKENS = 1  # net_num_new_slots_per_request for draft_model (K+2 accounting)
N_BLOCKS_PER_REQ = 32
BASE_POS_PER_REQ = 1000  # distinct position ranges make slot attribution checkable


def _build_padded_cad() -> AscendCommonAttentionMetadata:
    """Mimic the cad the drafter receives after unpadded(30, 5).

    qsl describes 5 requests x 6 tokens; block_table carries 6 rows
    (5 real + 1 zero-filled phantom row) because unpadded() keeps it padded.
    """
    total_reqs = NUM_REAL_REQS + PHANTOM_ROWS
    qsl = torch.arange(0, (NUM_REAL_REQS + 1) * TOKENS_PER_REQ, TOKENS_PER_REQ, dtype=torch.int32)
    bt = torch.arange(1, total_reqs * N_BLOCKS_PER_REQ + 1, dtype=torch.int32).reshape(total_reqs, N_BLOCKS_PER_REQ)
    bt[NUM_REAL_REQS:].fill_(0)  # phantom rows are zero-filled at padding time
    return AscendCommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        seq_lens=torch.full((NUM_REAL_REQS,), 64, dtype=torch.int64),
        num_reqs=NUM_REAL_REQS,
        num_actual_tokens=NUM_REAL_REQS * TOKENS_PER_REQ,
        max_query_len=TOKENS_PER_REQ,
        max_seq_len=BASE_POS_PER_REQ * NUM_REAL_REQS + TOKENS_PER_REQ + NUM_NEW_TOKENS,
        block_table_tensor=bt,
        slot_mapping=torch.zeros(NUM_REAL_REQS * TOKENS_PER_REQ, dtype=torch.int64),
    )


def _new_positions() -> torch.Tensor:
    # R*(K+2) = 35 positions: 7 contiguous positions per real request
    # (6 verify slots + 1 extra seed slot, matching drafter accounting).
    return torch.cat(
        [
            torch.arange(
                BASE_POS_PER_REQ * r,
                BASE_POS_PER_REQ * r + TOKENS_PER_REQ + NUM_NEW_TOKENS,
                dtype=torch.int64,
            )
            for r in range(NUM_REAL_REQS)
        ]
    )


def _expected_slots() -> list[int]:
    pos = _new_positions()
    bt = _build_padded_cad().block_table_tensor
    slots = []
    for i, p in enumerate(pos.tolist()):
        r = i // (TOKENS_PER_REQ + NUM_NEW_TOKENS)
        slots.append(int(bt[r, p // BLOCK_SIZE]) * BLOCK_SIZE + p % BLOCK_SIZE)
    return slots


def _rejected_mask() -> torch.Tensor:
    return torch.zeros(len(_new_positions()), dtype=torch.bool)


class TestBlockTableQslAlignment:
    def test_raw_padded_cad_reproduces_upstream_crash(self):
        """Anchor: the pre-fix shapes crash inside upstream's function."""
        cad = _build_padded_cad()
        with pytest.raises(RuntimeError, match="repeats must have the same size as input along dim"):
            compute_new_slot_mapping(
                cad=cad,
                new_positions=_new_positions(),
                is_rejected_token_mask=_rejected_mask(),
                block_size=BLOCK_SIZE,
                num_new_tokens=NUM_NEW_TOKENS,
                max_model_len=BASE_POS_PER_REQ * NUM_REAL_REQS * 2,
            )

    def test_aligned_cad_slots_correctly_attributed(self):
        cad = _build_padded_cad()
        aligned = align_block_table_for_slot_mapping(cad)
        assert aligned.block_table_tensor.shape[0] == NUM_REAL_REQS
        slots = compute_new_slot_mapping(
            cad=aligned,
            new_positions=_new_positions(),
            is_rejected_token_mask=_rejected_mask(),
            block_size=BLOCK_SIZE,
            num_new_tokens=NUM_NEW_TOKENS,
            max_model_len=BASE_POS_PER_REQ * NUM_REAL_REQS * 2,
        )
        expected = _expected_slots()
        assert slots.tolist() == expected  # request idx / block id / offset all correct

    def test_alignment_does_not_mutate_original_cad(self):
        cad = _build_padded_cad()
        original_rows = cad.block_table_tensor.shape[0]
        aligned = align_block_table_for_slot_mapping(cad)
        assert aligned is not cad
        assert cad.block_table_tensor.shape[0] == original_rows  # still padded

    def test_noop_when_block_table_already_consistent(self):
        cad = _build_padded_cad()
        consistent = cad.replace(block_table_tensor=cad.block_table_tensor[:NUM_REAL_REQS])
        assert align_block_table_for_slot_mapping(consistent) is consistent
