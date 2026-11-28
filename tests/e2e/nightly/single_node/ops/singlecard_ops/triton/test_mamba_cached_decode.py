# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.mamba.cached_decode import materialize_cached_gdn_decode

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Mamba cached-decode tests require an Ascend NPU",
)

# Widths and row counts cover MTP widths, graph padding and the BLOCK tail.
_CASES = [
    dict(actual_rows=2, graph_rows=2, width=4, token_count=8, query=[0, 4, 8]),
    dict(actual_rows=1, graph_rows=2, width=4, token_count=4, query=[0, 4]),
    dict(actual_rows=0, graph_rows=2, width=4, token_count=0, query=[0]),
    dict(actual_rows=3, graph_rows=5, width=3, token_count=9, query=[1, 3, 6, 10]),
    dict(actual_rows=4, graph_rows=8, width=4, token_count=12, query=[0, 2, 5, 9, 12]),
    dict(actual_rows=2, graph_rows=2, width=1, token_count=0, query=[7, 8, 9]),
]


def _golden(case):
    actual_rows = case["actual_rows"]
    graph_rows = case["graph_rows"]
    width = case["width"]
    device = torch.device("npu")

    query_source = torch.tensor(case["query"], dtype=torch.int32, device=device)
    state_source = torch.arange(actual_rows * width, dtype=torch.int32, device=device).view(actual_rows, width) + 100
    accepted_source = torch.arange(actual_rows, dtype=torch.int32, device=device) + 1
    token_source = torch.arange(case["token_count"], dtype=torch.int32, device=device) + 10

    state_output = torch.full((graph_rows, width), -1, dtype=torch.int32, device=device)
    query_output = torch.full((graph_rows + 1,), -1, dtype=torch.int32, device=device)
    accepted_output = torch.full((graph_rows,), -1, dtype=torch.int32, device=device)
    token_output = torch.full((case["token_count"],), -1, dtype=torch.int32, device=device)
    mask_output = torch.zeros(graph_rows, dtype=torch.bool, device=device)
    lengths_output = torch.full((graph_rows + 1,), -1, dtype=torch.int32, device=device)

    expected_state = torch.zeros((graph_rows, width), dtype=torch.int32, device=device)
    expected_state[:actual_rows] = state_source
    expected_accepted = torch.ones(graph_rows, dtype=torch.int32, device=device)
    expected_accepted[:actual_rows] = accepted_source
    expected_mask = torch.arange(graph_rows, device=device) < actual_rows
    expected_query = query_source[
        torch.minimum(torch.arange(graph_rows + 1, device=device), torch.tensor(actual_rows, device=device))
    ]
    expected_lengths = torch.diff(expected_query, prepend=torch.zeros(1, dtype=torch.int32, device=device))

    materialize_cached_gdn_decode(
        state_source,
        query_source,
        accepted_source,
        token_source,
        state_output,
        query_output,
        accepted_output,
        token_output,
        mask_output,
        lengths_output,
        actual_rows,
        graph_rows,
    )

    assert torch.equal(state_output, expected_state)
    assert torch.equal(query_output, expected_query)
    assert torch.equal(accepted_output, expected_accepted)
    assert torch.equal(token_output, token_source)
    assert torch.equal(mask_output, expected_mask)
    assert torch.equal(lengths_output, expected_lengths)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("case", _CASES)
def test_cached_decode_materialization_is_bit_exact(case):
    _golden(case)
