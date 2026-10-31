# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_ascend.device.device_config import AscendDeviceType
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


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize(
    "ranks,dim,tokens,heads,stride,eligible",
    [
        (2, 512, 16, 4, 1, False),
        (4, 96, 1, 4, 1, False),
        (4, 96, 16, 4, 1, True),
        (8, 256, 4, 4, 1, False),
        (8, 256, 16, 4, 1, True),
        (8, 256, 256, 4, 1, True),
        (12, 160, 16, 4, 1, True),
        (16, 512, 1, 4, 1, False),
        (16, 512, 4, 4, 1, True),
        (16, 512, 16, 1, 1, True),
        (16, 1024, 4, 4, 1, True),
        (32, 512, 4, 4, 1, False),
        (33, 512, 16, 4, 1, False),
        (16, 512, 16, 4, 2, False),
        (16, 64, 4, 4, 1, False),
        (32, 2048, 16, 4, 1, False),
    ],
)
def test_rank_tiled_bounded_work_dispatch(monkeypatch, device_type, ranks, dim, tokens, heads, stride, eligible):
    # CPU CI uses a Triton stub without its launch-size helpers.
    power_of_two = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setattr(dcp_a2a.triton, "next_power_of_2", power_of_two, raising=False)
    monkeypatch.setattr(dcp_a2a.triton, "cdiv", lambda a, b: (a + b - 1) // b, raising=False)
    monkeypatch.setattr(dcp_a2a, "is_950", lambda: device_type == AscendDeviceType.A5)
    monkeypatch.setattr(dcp_a2a, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(dcp_a2a, "get_vectorcore_num", lambda: 48)
    tiled, scalar, batched = Mock(), Mock(), Mock()
    for name, launch in [("rank_tiled", tiled), ("", scalar), ("batched", batched)]:
        suffix = "_" + name if name else ""
        kernel = Mock()
        kernel.__getitem__ = Mock(return_value=launch)
        monkeypatch.setattr(dcp_a2a, f"_fused_dcp_lse_combine{suffix}_kernel", kernel)
    with FakeTensorMode():
        recv = torch.empty(ranks, heads, tokens, dim + 4, dtype=torch.bfloat16, device="npu:0")
        # A view operation on a fake NPU tensor still enters the device guard
        # in CPU-only PyTorch; construct the same layout without slicing.
        local = torch.empty_strided(
            (tokens, heads, dim),
            (heads * dim * stride, dim * stride, stride),
            dtype=torch.bfloat16,
            device="npu:0",
        )
        lse = torch.empty(tokens, heads, 1, dtype=torch.float32, device="npu:0")
        output = dcp_a2a.fused_dcp_lse_combine(recv, dim, 1, local_output=local, local_lse=lse)
    assert output.shape == local.shape
    batched_a5 = device_type == AscendDeviceType.A5 and (ranks, dim, tokens, heads, stride) == (
        8,
        256,
        256,
        4,
        1,
    )
    expected = eligible and device_type != AscendDeviceType.A5 and not batched_a5
    assert tiled.call_count == int(expected)
    assert batched.call_count == int(batched_a5)
    assert tiled.call_count + scalar.call_count + batched.call_count == 1
    if expected:
        assert tiled.call_args.kwargs["BLOCK_R"] * tiled.call_args.kwargs["BLOCK_D"] <= 2048
