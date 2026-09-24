# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder


@pytest.mark.parametrize("dcp", [False, True])
@pytest.mark.parametrize("state", [None, AscendAttentionState.DecodeOnly])
@pytest.mark.parametrize("has_flags", [False, True])
def test_mla_capture_defaults_preserve_common_metadata(dcp, state, has_flags):
    builder = AscendMLAMetadataBuilder.__new__(AscendMLAMetadataBuilder)
    builder.dcp_enabled = dcp
    builder.reorder_batch_threshold = 3
    builder.build = MagicMock(side_effect=lambda prefix, metadata: metadata)
    flags = torch.tensor([True, False]) if has_flags else None
    positions = torch.arange(6)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=6,
        max_query_len=3,
        attn_state=state,
        is_prefilling=flags,
        positions=positions,
    )

    captured = builder.build_for_cudagraph_capture(common)

    builder.build.assert_called_once_with(0, captured)
    assert captured is not common
    assert captured.attn_state == (AscendAttentionState.ChunkedPrefill if state is None else state)
    assert common.attn_state is state
    assert common.is_prefilling is flags
    assert captured.positions is positions
    if dcp and not has_flags:
        assert captured.is_prefilling.dtype == torch.bool
        assert captured.is_prefilling.device.type == "cpu"
        assert captured.is_prefilling.tolist() == [False, False]
    else:
        assert captured.is_prefilling is flags


def test_mla_capture_keeps_upstream_decode_validation():
    builder = AscendMLAMetadataBuilder.__new__(AscendMLAMetadataBuilder)
    builder.dcp_enabled = True
    builder.reorder_batch_threshold = 3
    builder.build = MagicMock()
    common = SimpleNamespace(num_reqs=2, num_actual_tokens=8, max_query_len=4, attn_state=None, is_prefilling=None)

    with pytest.raises(AssertionError):
        builder.build_for_cudagraph_capture(common)

    builder.build.assert_not_called()
    assert common.attn_state is None
    assert common.is_prefilling is None
