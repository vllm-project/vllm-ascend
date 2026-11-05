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


@pytest.fixture
def combine_launch_spies(monkeypatch):
    monkeypatch.setattr(dcp_a2a.triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length(), raising=False)
    monkeypatch.setattr(dcp_a2a.triton, "cdiv", lambda a, b: (a + b - 1) // b, raising=False)
    monkeypatch.setattr(
        dcp_a2a, "is_950", Mock(side_effect=AssertionError("combine must not query chip model")), raising=False
    )
    monkeypatch.setattr(dcp_a2a, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(dcp_a2a, "get_vectorcore_num", lambda: 48)
    launches = {}
    for suffix in ("", "_batched"):
        launch = Mock()
        kernel = Mock()
        kernel.__getitem__ = Mock(return_value=launch)
        monkeypatch.setattr(dcp_a2a, f"_fused_dcp_lse_combine{suffix}_kernel", kernel)
        launches[suffix] = launch
    return launches


@pytest.mark.parametrize("tokens,dim,stride", [(48, 512, 1), (128, 512, 1), (128, 512, 2), (17, 1024, 1), (1, 512, 1)])
def test_pack_shape_dispatch_without_chip_probe(monkeypatch, tokens, dim, stride):
    monkeypatch.setattr(
        dcp_a2a, "is_950", Mock(side_effect=AssertionError("pack must not query chip model")), raising=False
    )
    monkeypatch.setattr(dcp_a2a, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(dcp_a2a, "get_vectorcore_num", lambda: 40)
    launches = {}
    for suffix in ("", "_batched"):
        kernel = Mock()
        launch = Mock()
        kernel.__getitem__ = Mock(return_value=launch)
        monkeypatch.setattr(dcp_a2a, f"_pack_dcp_output_lse{suffix}_kernel", kernel)
        launches[suffix] = launch
    with FakeTensorMode():
        output = torch.empty((tokens, 96, dim * stride), device="npu:0", dtype=torch.bfloat16)[..., ::stride]
        lse = torch.empty((tokens, 96, 1), device="npu:0", dtype=torch.float32)
        dcp_a2a.pack_dcp_output_lse(output, lse, 16, 1)
    expected = tokens * 96 >= 8 * 40
    assert launches["_batched"].call_count == int(expected)
    assert sum(launch.call_count for launch in launches.values()) == 1
    if expected:
        assert launches["_batched"].call_args.kwargs["BLOCK_ROWS"] == (
            32 if dim == 512 and tokens >= 128 and stride == 1 else 8
        )


@pytest.mark.parametrize(
    "ranks,tokens,heads",
    [
        (1, 48, 6),
        (2, 48, 6),
        (4, 4, 8),
        (4, 64, 24),
        (5, 12, 24),
        (8, 32, 12),
        (9, 32, 24),
        (16, 48, 6),
        (17, 49, 6),
        (16, 128, 12),
        (8, 31, 24),
        (8, 65, 24),
        (16, 1, 6),
    ],
)
def test_streaming_batched_shape_dispatch(combine_launch_spies, ranks, tokens, heads):
    expected = tokens * heads >= 4 * 48
    with FakeTensorMode():
        recv = torch.empty(ranks, heads, tokens, 516, dtype=torch.bfloat16, device="npu:0")
        local = torch.empty(tokens, heads, 512, dtype=torch.bfloat16, device="npu:0")
        lse = torch.empty(tokens, heads, 1, dtype=torch.float32, device="npu:0")
        output = dcp_a2a.fused_dcp_lse_combine(recv, 512, 1, local_output=local, local_lse=lse)
    assert output.shape == local.shape
    batched = combine_launch_spies["_batched"]
    assert batched.call_count == int(expected)
    assert sum(launch.call_count for launch in combine_launch_spies.values()) == 1
    if expected:
        assert batched.call_args.kwargs["COALESCED_LSE"] is True
        assert batched.call_args.kwargs["BLOCK_ROWS"] == 8
        expected_unroll = ranks if tokens * heads >= 2 * 8 * 48 and ranks * 8 * 512 <= 36 * 1024 else 1
        assert batched.call_args.kwargs["RANK_UNROLL"] == expected_unroll


@pytest.mark.parametrize("guard", ["scatter", "dtype", "local_dtype", "no_local", "feature_stride", "row_stride"])
def test_streaming_batched_contract_guards(monkeypatch, combine_launch_spies, guard):
    dtype = torch.float16 if guard == "dtype" else torch.bfloat16
    local_dtype = torch.float32 if guard == "local_dtype" else dtype
    scatter = int(guard != "scatter")
    with FakeTensorMode():
        recv_shape = (8, 24, 64, 516) if scatter else (8, 64, 24, 516)
        recv = torch.empty(recv_shape, dtype=dtype, device="npu:0")
        stride = 2 if guard == "feature_stride" else 1
        row_stride = 2 if guard == "row_stride" else 1
        local = torch.empty_strided(
            (64, 24, 512),
            (24 * 512 * stride * row_stride, 512 * stride, stride),
            dtype=local_dtype,
            device="npu:0",
        )
        lse = torch.empty(64, 24, 1, dtype=torch.float32, device="npu:0")
        if guard == "no_local":
            local = lse = None
        dcp_a2a.fused_dcp_lse_combine(recv, 512, scatter, local_output=local, local_lse=lse)
    assert sum(launch.call_count for launch in combine_launch_spies.values()) == 1
    batched = combine_launch_spies["_batched"]
    assert batched.call_count == int(guard not in ("scatter", "dtype", "feature_stride"))
    if batched.called:
        assert batched.call_args.kwargs["COALESCED_LSE"] is True
