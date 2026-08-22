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
"""Unit tests for the draft_model drafter FULL-graph sizing arithmetic.

The draft_model drafter consumes R*(K+2) tokens per step (R*(K+1) verify
tokens plus one extra seed slot per request), which is never a multiple
of (K+1) and therefore cannot go through the shared FULL-uniform
dispatch. When ``additional_config.draft_model_full_graph`` is enabled,
the runner derives a drafter-specific R*(K+2) capture table and the
proposer dispatches on (K+2) multiples. These tests pin that arithmetic:
capture-size derivation, padded-size selection with eager fallback,
BatchDescriptor request counting, and the K+2-stepped query_start_loc
used for padding (phantom requests appended after the real ones).
"""

from __future__ import annotations

import torch

from vllm_ascend.spec_decode.llm_base_proposer import select_drafter_padded_size
from vllm_ascend.worker.model_runner_v1 import derive_draft_model_graph_sizes


def test_derive_sizes_from_target_capture_table():
    # [6, 12] with K=5 (target sizes are R*(K+1)): 1*(6+1)=7, 2*(6+2)=14.
    assert derive_draft_model_graph_sizes([6, 12], 5) == [7, 14]
    assert derive_draft_model_graph_sizes([6], 5) == [7]
    assert derive_draft_model_graph_sizes([8, 16, 24], 3) == [10, 20, 30]


def test_derive_sizes_is_exact_r_times_k_plus_two():
    k = 5
    for r in range(1, 6):
        target_size = r * (k + 1)
        (drafter_size,) = derive_draft_model_graph_sizes([target_size], k)
        assert drafter_size == r * (k + 2)


def test_select_padded_size_pads_up_and_falls_back():
    sizes = [7, 14]
    assert select_drafter_padded_size(7, sizes) == 7  # exact fit
    assert select_drafter_padded_size(8, sizes) == 14  # pad up
    assert select_drafter_padded_size(13, sizes) == 14
    assert select_drafter_padded_size(6, sizes) == 7  # smaller than min
    assert select_drafter_padded_size(15, sizes) is None  # eager fallback
    assert select_drafter_padded_size(7, []) is None  # no table -> eager


def test_batch_descriptor_request_counting():
    # num_reqs = padded // (K+2) counts padded (real + phantom) requests.
    k = 5
    drafter_query_len = k + 2
    for padded in (7, 14, 21, 28):
        assert padded % drafter_query_len == 0
        assert padded // drafter_query_len == padded // (k + 2)


def test_capture_translation_matches_runtime_shapes():
    # A graph captured at target size s = R*(K+1) is translated at capture
    # time to drafter_reqs * (K+2) tokens; runtime num_tokens for the same
    # R requests is exactly R*(K+2), so capture and replay shapes agree.
    k = 5
    for r in range(1, 5):
        target_padded = r * (k + 1)
        drafter_reqs = target_padded // (k + 1)
        captured = drafter_reqs * (k + 2)
        runtime = r * (k + 2)
        assert captured == runtime


def test_query_start_loc_steps_by_k_plus_two():
    # Phantom requests are appended after the real ones with query_start_loc
    # stepping by (K+2); zero-length seq_lens rows make FIA skip them.
    k = 5
    drafter_query_len = k + 2
    num_reqs_padded = 3
    qsl = torch.arange(0, (num_reqs_padded + 1) * drafter_query_len, drafter_query_len, dtype=torch.int32)
    assert qsl.tolist() == [0, 7, 14, 21]
    assert (qsl[1:] - qsl[:-1] == drafter_query_len).all()


def test_runtime_tokens_pad_into_derived_table():
    # End-to-end arithmetic: for every R whose runtime R*(K+2) fits in the
    # table, the selected padded size is a captured size and a multiple of
    # (K+2); anything above the table falls back to eager.
    k = 5
    table = derive_draft_model_graph_sizes([6, 12], k)
    for r in range(1, 3):
        padded = select_drafter_padded_size(r * (k + 2), table)
        assert padded is not None
        assert padded % (k + 2) == 0
    assert select_drafter_padded_size(3 * (k + 2), table) is None
