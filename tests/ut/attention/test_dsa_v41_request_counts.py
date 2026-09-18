# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import torch

from vllm_ascend.attention.dsa_v41 import _request_counts


def _common(is_prefilling, query_start_loc_cpu):
    return SimpleNamespace(
        is_prefilling=is_prefilling,
        query_start_loc_cpu=query_start_loc_cpu,
    )


def test_mrv2_real_request_count_with_padded_start_loc():
    # MRV2 keeps is_prefilling at the real request count while
    # query_start_loc_cpu is padded to the graph request count. The mask
    # length (not the padded count) must drive the indexing, otherwise a
    # single prefill request crashes with mask [1] vs tensor [2].
    common = _common(
        is_prefilling=torch.tensor([True]),
        query_start_loc_cpu=torch.tensor([0, 24, 24], dtype=torch.int32),
    )
    assert _request_counts(common, num_reqs=2) == (0, 0, 1, 24)


def test_mrv2_mixed_batch_padding_rows_not_counted():
    # One prefill (10 tokens) + one decode, padded to 4 requests with
    # zero-length padding rows at the tail.
    common = _common(
        is_prefilling=torch.tensor([True, False]),
        query_start_loc_cpu=torch.tensor([0, 10, 11, 11, 11], dtype=torch.int32),
    )
    assert _request_counts(common, num_reqs=4) == (1, 1, 1, 10)


def test_mrv1_padded_flag_length_keeps_legacy_counts():
    # MRV1 pads is_prefilling to the padded request count with padding rows
    # explicitly set to False; those rows keep counting as decodes.
    common = _common(
        is_prefilling=torch.tensor([True, False, False, False]),
        query_start_loc_cpu=torch.tensor([0, 10, 11, 11, 11], dtype=torch.int32),
    )
    assert _request_counts(common, num_reqs=4) == (3, 1, 1, 10)


def test_non_cpu_or_missing_inputs_return_zeros():
    assert _request_counts(SimpleNamespace(), num_reqs=2) == (0, 0, 0, 0)
    meta = _common(
        is_prefilling=torch.zeros(2, dtype=torch.bool, device="meta"),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    assert _request_counts(meta, num_reqs=2) == (0, 0, 0, 0)
