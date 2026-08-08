# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Offline (CPU-comparable) tests for the structured-output bitmask wiring.

The Triton kernel launch itself requires an NPU and is verified separately in
the e2e suite. Here we test the device-independent pieces:

* ``build_grammar_bitmask_indices`` -- the reorder/spec-decode index math that
  maps compacted xgrammar bitmask rows onto batch-aligned logits rows.
* a CPU reference apply that mirrors the kernel semantics (bit == 0 -> -inf),
  confirming the gathered bitmask + index mapping masks exactly the positions a
  reference full-batch apply would.
"""

import numpy as np
import torch

from vllm_ascend.worker.v2.structured_outputs import (
    BITMASK_BITS_PER_WORD,
    build_grammar_bitmask_indices,
)


def _pack_bitmask(allowed: np.ndarray) -> np.ndarray:
    """Pack a boolean [rows, vocab] allow-mask into xgrammar int32 layout.

    A set bit means the token is allowed; a clear bit means it is blocked.
    Bit 31 lands on the int32 sign bit, so we pack as uint32 and reinterpret -
    exactly how xgrammar produces the mask.
    """
    rows, vocab = allowed.shape
    num_words = (vocab + BITMASK_BITS_PER_WORD - 1) // BITMASK_BITS_PER_WORD
    packed = np.zeros((rows, num_words), dtype=np.uint32)
    for r in range(rows):
        for t in range(vocab):
            if allowed[r, t]:
                word = t // BITMASK_BITS_PER_WORD
                bit = t % BITMASK_BITS_PER_WORD
                packed[r, word] |= np.uint32(1) << np.uint32(bit)
    return packed.view(np.int32)


def _reference_apply(
    logits: torch.Tensor, packed_bitmask: np.ndarray, bitmask_rows: list[int], out_indices: list[int]
) -> torch.Tensor:
    """CPU reference of what the NPU kernel does: blocked positions -> -inf."""
    out = logits.clone()
    vocab = logits.shape[-1]
    for k, logit_row in enumerate(out_indices):
        words = packed_bitmask[bitmask_rows[k]]
        for t in range(vocab):
            word = t // BITMASK_BITS_PER_WORD
            bit = t % BITMASK_BITS_PER_WORD
            allowed = (int(np.uint32(words[word])) >> bit) & 1
            if not allowed:
                out[logit_row, t] = float("-inf")
    return out


def test_indices_simple_subset_in_order():
    # Batch: r0, r1, r2; only r0 and r2 are structured-output requests.
    req_ids = ["r0", "r1", "r2"]
    struct_ids = ["r0", "r2"]
    bitmask_rows, out_indices = build_grammar_bitmask_indices(req_ids, struct_ids, scheduled_spec_decode_tokens={})
    # Compacted bitmask rows map 1:1; out_indices point at r0 (row 0) and r2 (row 2).
    assert bitmask_rows == [0, 1]
    assert out_indices == [0, 2]


def test_indices_reordered_subset():
    # structured_output_request_ids order differs from batch order.
    req_ids = ["a", "b", "c", "d"]
    struct_ids = ["c", "a"]  # different order than the batch
    bitmask_rows, out_indices = build_grammar_bitmask_indices(req_ids, struct_ids, scheduled_spec_decode_tokens={})
    # bitmask row 0 -> req "c" -> logits row 2; bitmask row 1 -> req "a" -> row 0.
    assert bitmask_rows == [0, 1]
    assert out_indices == [2, 0]


def test_indices_spec_decode_row_expansion():
    # r1 has 2 spec tokens -> occupies 3 logits rows (1 + 2) and shifts r2.
    req_ids = ["r0", "r1", "r2"]
    struct_ids = ["r1", "r2"]
    spec = {"r1": (101, 102), "r2": ()}
    bitmask_rows, out_indices = build_grammar_bitmask_indices(req_ids, struct_ids, scheduled_spec_decode_tokens=spec)
    # logits layout: r0 -> 0; r1 -> 1,2,3; r2 -> 4.
    # Compacted bitmask: rows 0,1,2 are r1's 3 rows; row 3 is r2.
    assert bitmask_rows == [0, 1, 2, 3]
    assert out_indices == [1, 2, 3, 4]


def test_indices_skips_request_absent_from_batch():
    # "ghost" is a structured-output request not in the current batch; its
    # bitmask rows must be skipped while keeping alignment for the rest.
    req_ids = ["r0", "r1"]
    struct_ids = ["ghost", "r1"]
    bitmask_rows, out_indices = build_grammar_bitmask_indices(req_ids, struct_ids, scheduled_spec_decode_tokens={})
    # "ghost" occupies source bitmask row 0 (skipped); "r1" is row 1 -> logits 1.
    assert bitmask_rows == [1]
    assert out_indices == [1]


def test_indices_empty_when_no_struct_reqs_in_batch():
    bitmask_rows, out_indices = build_grammar_bitmask_indices(["r0", "r1"], ["ghost"], scheduled_spec_decode_tokens={})
    assert bitmask_rows == []
    assert out_indices == []


def test_cpu_reference_masking_matches_full_batch_apply():
    # End-to-end check on CPU: gathering the compacted bitmask by bitmask_rows
    # and masking at out_indices must equal masking the full batch directly.
    torch.manual_seed(0)
    vocab = 70  # spans 3 int32 words to exercise packing boundaries
    num_logits_rows = 4
    logits = torch.randn(num_logits_rows, vocab, dtype=torch.bfloat16)

    req_ids = ["a", "b", "c", "d"]
    struct_ids = ["c", "a"]  # reordered subset; rows 2 and 0 are constrained
    bitmask_rows, out_indices = build_grammar_bitmask_indices(req_ids, struct_ids, scheduled_spec_decode_tokens={})

    # Source compacted bitmask: row 0 -> "c", row 1 -> "a" (struct_ids order).
    rng = np.random.default_rng(0)
    allowed = rng.integers(0, 2, size=(len(struct_ids), vocab)).astype(bool)
    packed = _pack_bitmask(allowed)

    got = _reference_apply(logits, packed, bitmask_rows, out_indices)

    # Independent expectation: build a full-batch allow-mask and apply directly.
    expected = logits.clone()
    # struct_ids[0] == "c" -> logits row 2; struct_ids[1] == "a" -> logits row 0.
    logit_row_for_struct = {0: 2, 1: 0}
    for src_row, logit_row in logit_row_for_struct.items():
        for t in range(vocab):
            if not allowed[src_row, t]:
                expected[logit_row, t] = float("-inf")

    assert torch.equal(got, expected)
    # Unconstrained rows (1 and 3) must be untouched.
    assert torch.equal(got[1], logits[1])
    assert torch.equal(got[3], logits[3])
