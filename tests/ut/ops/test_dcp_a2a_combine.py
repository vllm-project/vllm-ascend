# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_ascend.ops.triton.dcp import dcp_a2a


@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize("dtype,packed_extra", [(torch.float32, 1), (torch.bfloat16, 4)])
def test_deferred_fake_layout(scatter_dim, dtype, packed_extra):
    output = torch.empty(8, 12, 96, dtype=dtype, device="meta")
    lse = torch.empty(8, 12, 1, device="meta")
    actual = dcp_a2a.dcp_a2a_fused_fake(output, lse, 2, scatter_dim, "dcp", defer_combine=True)
    assert actual.shape == (2, output.shape[scatter_dim] // 2, output.shape[1 - scatter_dim], 96 + packed_extra)
    assert actual.dtype == dtype


@pytest.mark.parametrize("scatter_size", [1, 2])
def test_deferred_exchange_skips_combine(monkeypatch, scatter_size):
    send = torch.empty(scatter_size, 2, 3, 5)
    monkeypatch.setattr(dcp_a2a, "pack_dcp_output_lse", Mock(return_value=send))
    collective = Mock()
    monkeypatch.setattr(dcp_a2a.dist, "all_to_all_single", collective)
    combine = Mock(side_effect=AssertionError("deferred exchange must not combine"))
    monkeypatch.setattr(dcp_a2a, "fused_dcp_lse_combine", combine)
    result = dcp_a2a.dcp_a2a_fused_combine(
        torch.empty(2, 3, 4), torch.empty(2, 3, 1), scatter_size, 1, object(), defer_combine=True
    )
    assert result.shape == send.shape
    assert collective.call_count == int(scatter_size > 1)
    combine.assert_not_called()
    if scatter_size == 1:
        assert result is send


@pytest.mark.parametrize(
    "device_type", [dcp_a2a.AscendDeviceType.A2, dcp_a2a.AscendDeviceType.A3, dcp_a2a.AscendDeviceType.A5]
)
@pytest.mark.parametrize("tokens", [15, 16, 17])
def test_rank_tiled_qualified_device_and_token_boundary(monkeypatch, device_type, tokens):
    monkeypatch.setattr(dcp_a2a, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(dcp_a2a, "is_950", lambda: device_type == dcp_a2a.AscendDeviceType.A5)
    monkeypatch.setattr(dcp_a2a, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(dcp_a2a, "get_vectorcore_num", lambda: 32)
    tiled, scalar, batched = Mock(), Mock(), Mock()
    for name, launch in [("rank_tiled", tiled), ("", scalar), ("batched", batched)]:
        suffix = "_" + name if name else ""
        kernel = Mock()
        kernel.__getitem__ = Mock(return_value=launch)
        monkeypatch.setattr(dcp_a2a, f"_fused_dcp_lse_combine{suffix}_kernel", kernel)
    with FakeTensorMode():
        recv = torch.empty(16, 4, tokens, 516, dtype=torch.bfloat16, device="npu")
        local = torch.empty(tokens, 4, 512, dtype=torch.bfloat16, device="npu")
        lse = torch.empty(tokens, 4, 1, dtype=torch.float32, device="npu")
        output = dcp_a2a.fused_dcp_lse_combine(recv, 512, 1, local_output=local, local_lse=lse)
    assert output.shape == local.shape
    assert tiled.call_count == int(device_type == dcp_a2a.AscendDeviceType.A3 and tokens >= 16)
    assert tiled.call_count + scalar.call_count + batched.call_count == 1
