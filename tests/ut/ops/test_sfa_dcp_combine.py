# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.triton import sfa_cp


@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize("dtype,packed_extra", [(torch.float32, 1), (torch.bfloat16, 4)])
def test_deferred_fake_layout(scatter_dim, dtype, packed_extra):
    output = torch.empty(8, 12, 96, dtype=dtype, device="meta")
    lse = torch.empty(8, 12, 1, device="meta")
    actual = sfa_cp.sfa_dcp_a2a_fused_fake(output, lse, 2, scatter_dim, "dcp", defer_combine=True)
    assert actual.shape == (2, output.shape[scatter_dim] // 2, output.shape[1 - scatter_dim], 96 + packed_extra)
    assert actual.dtype == dtype


@pytest.mark.parametrize("scatter_size", [1, 2])
def test_deferred_exchange_skips_combine(monkeypatch, scatter_size):
    send = torch.empty(scatter_size, 2, 3, 5)
    monkeypatch.setattr(sfa_cp, "pack_sfa_dcp_output_lse", Mock(return_value=send))
    collective = Mock()
    monkeypatch.setattr(sfa_cp.dist, "all_to_all_single", collective)
    combine = Mock(side_effect=AssertionError("deferred exchange must not combine"))
    monkeypatch.setattr(sfa_cp, "fused_sfa_dcp_lse_combine", combine)
    result = sfa_cp.sfa_dcp_a2a_fused_combine(
        torch.empty(2, 3, 4), torch.empty(2, 3, 1), scatter_size, 1, object(), defer_combine=True
    )
    assert result.shape == send.shape
    assert collective.call_count == int(scatter_size > 1)
    combine.assert_not_called()
    if scatter_size == 1:
        assert result is send
