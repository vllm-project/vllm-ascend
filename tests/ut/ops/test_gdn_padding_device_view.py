# SPDX-License-Identifier: Apache-2.0
"""Regression: graph padding must not upload CPU boundaries synchronously."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.gdn_attn_builder import _remove_spec_graph_padding_queries


@pytest.mark.parametrize("live,capacity", [(1, 2), (5, 6), (5, 7), (7, 8)])
@pytest.mark.parametrize("strided", [False, True])
def test_padding_uses_existing_device_queries_without_host_upload(live, capacity, strided):
    original = torch.arange(capacity + 1, dtype=torch.int32) * 6
    if strided:
        original = torch.stack((original, original), dim=1)[:, 0]
    cpu = original.clone()
    metadata = SimpleNamespace(
        num_reqs=capacity,
        seq_lens_cpu_upper_bound=torch.tensor([262144] * live + [0] * (capacity - live)),
        query_start_loc_cpu=cpu,
        query_start_loc=original,
    )
    metadata.replace = lambda **changes: SimpleNamespace(**{**vars(metadata), **changes})
    drafts = torch.tensor([5] * live + [-1] * (capacity - live))
    # The old .to(device) introduces an NPU stream synchronization even when
    # only a few int32 boundaries need correcting. Forbid that transfer API.
    with patch.object(torch.Tensor, "to", side_effect=AssertionError("unexpected host upload")):
        result = _remove_spec_graph_padding_queries(metadata, drafts)
    expected = [min(i, live) * 6 for i in range(capacity + 1)]
    assert result.query_start_loc.tolist() == expected
    assert result.query_start_loc_cpu.tolist() == expected
    assert result.num_actual_tokens == live * 6
    assert original.tolist() == [i * 6 for i in range(capacity + 1)]
    assert cpu.tolist() == original.tolist()


def test_no_padding_preserves_input_metadata():
    metadata = SimpleNamespace(
        num_reqs=2,
        seq_lens_cpu_upper_bound=torch.tensor([100, 100]),
    )
    assert _remove_spec_graph_padding_queries(metadata, torch.tensor([5, 5])) is metadata


def test_non_suffix_padding_is_rejected():
    metadata = SimpleNamespace(
        num_reqs=3,
        seq_lens_cpu_upper_bound=torch.tensor([100, 0, 100]),
    )
    with pytest.raises(ValueError, match="request suffix"):
        _remove_spec_graph_padding_queries(metadata, torch.tensor([5, -1, 5]))
