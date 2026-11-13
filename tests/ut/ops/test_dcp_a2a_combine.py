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
        (4, 96, 16, 4, 1, False),
        (8, 256, 4, 4, 1, False),
        (8, 256, 16, 4, 1, False),
        (8, 256, 256, 4, 1, False),
        (12, 160, 16, 4, 1, False),
        (16, 512, 1, 4, 1, False),
        (16, 512, 4, 4, 1, True),
        (16, 512, 16, 1, 1, True),
        (16, 512, 256, 4, 1, True),
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
    monkeypatch.setattr(dcp_a2a, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(dcp_a2a, "_A3_STATIC_SIGNATURES", frozenset())
    monkeypatch.setattr(dcp_a2a, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(dcp_a2a, "get_vectorcore_num", lambda: 48)
    tiled, static, scalar, batched = Mock(), Mock(), Mock(), Mock()
    for name, launch in [
        ("rank_tiled", tiled),
        ("rank_tiled_static", static),
        ("", scalar),
        ("batched", batched),
    ]:
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
    # The parallel rank tile is A5-only (A3 lowers a runtime rank-row stride to
    # scalar-rate transfers) and runs before the batched path.
    expected_tiled = eligible and device_type == AscendDeviceType.A5
    expected_batched = not expected_tiled and dim <= 512 and tokens * heads >= 4 * 48 and stride == 1
    assert tiled.call_count == int(expected_tiled)
    assert static.call_count == 0
    assert batched.call_count == int(expected_batched)
    assert tiled.call_count + static.call_count + scalar.call_count + batched.call_count == 1
    if expected_tiled:
        assert tiled.call_args.kwargs["BLOCK_R"] * tiled.call_args.kwargs["BLOCK_D"] <= 8192
        assert tiled.call_args.kwargs["BLOCK_D"] <= 512


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize(
    "warmed,ranks,tokens",
    [
        (False, 16, 16),
        (True, 16, 16),
        (True, 8, 16),
        (True, 8, 4),
        (True, 16, 4),
        (True, 16, 13),
        (True, 16, 191),
        (True, 16, 2),
    ],
)
def test_a3_static_bucket_dispatch(monkeypatch, device_type, warmed, ranks, tokens):
    power_of_two = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setattr(dcp_a2a.triton, "next_power_of_2", power_of_two, raising=False)
    monkeypatch.setattr(dcp_a2a.triton, "cdiv", lambda a, b: (a + b - 1) // b, raising=False)
    monkeypatch.setattr(dcp_a2a, "is_950", lambda: device_type == AscendDeviceType.A5)
    monkeypatch.setattr(dcp_a2a, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(dcp_a2a, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(dcp_a2a, "get_vectorcore_num", lambda: 48)
    monkeypatch.setattr(dcp_a2a, "_A3_WARMING", False)
    monkeypatch.setattr(dcp_a2a, "_UNWARMED_LOGGED", set())
    # The warmed key includes the local strides, which participate in the
    # Triton cache key.
    full_key = (4, tokens, 512, ranks, 4 * 512, 512, 1, 4, 1)
    monkeypatch.setattr(dcp_a2a, "_A3_STATIC_SIGNATURES", frozenset({full_key}) if warmed else frozenset())
    tiled, static, scalar, batched = Mock(), Mock(), Mock(), Mock()
    for name, launch in [
        ("rank_tiled", tiled),
        ("rank_tiled_static", static),
        ("", scalar),
        ("batched", batched),
    ]:
        suffix = "_" + name if name else ""
        kernel = Mock()
        kernel.__getitem__ = Mock(return_value=launch)
        monkeypatch.setattr(dcp_a2a, f"_fused_dcp_lse_combine{suffix}_kernel", kernel)
    with FakeTensorMode():
        recv = torch.empty(ranks, 4, tokens, 516, dtype=torch.bfloat16, device="npu:0")
        local = torch.empty_strided((tokens, 4, 512), (4 * 512, 512, 1), dtype=torch.bfloat16, device="npu:0")
        lse = torch.empty(tokens, 4, 1, dtype=torch.float32, device="npu:0")
        dcp_a2a.fused_dcp_lse_combine(recv, 512, 1, local_output=local, local_lse=lse)
    # A3 uses the static twin only for a warmed signature; A5 keeps the
    # dynamic kernel; A2 keeps the scalar path.
    min_rows = max((2 * 48 + ranks - 1) // ranks, ranks)
    expected_static = warmed and device_type == AscendDeviceType.A3 and tokens * 4 >= min_rows
    expected_dynamic = device_type == AscendDeviceType.A5 and tokens * 4 >= min_rows and 8 <= ranks <= 16
    assert static.call_count == int(expected_static)
    assert tiled.call_count == int(expected_dynamic)
    assert tiled.call_count + static.call_count + scalar.call_count + batched.call_count == 1
    if expected_static:
        assert static.call_args.kwargs["NUM_TOKENS"] == tokens
        assert static.call_args.kwargs["NUM_HEADS"] == 4
        assert static.call_args.kwargs["BLOCK_D"] == 512


def test_a3_static_warmup_accumulates(monkeypatch):
    monkeypatch.setattr(dcp_a2a, "_A3_STATIC_SIGNATURES", frozenset())
    monkeypatch.setattr(dcp_a2a, "_A3_RECORDED", set())
    first = (4, 16, 512, 16, 2048, 512, 1, 4, 1)
    second = (4, 32, 512, 16, 2048, 512, 1, 4, 1)
    dcp_a2a.begin_a3_static_warmup({(4, 16, 512, 16)})
    dcp_a2a._A3_RECORDED.add(first)
    dcp_a2a.end_a3_static_warmup()
    dcp_a2a.begin_a3_static_warmup({(4, 32, 512, 16)})
    dcp_a2a._A3_RECORDED.add(second)
    dcp_a2a.end_a3_static_warmup()
    assert frozenset({first, second}) == dcp_a2a._A3_STATIC_SIGNATURES


@pytest.mark.parametrize("device_type", [AscendDeviceType.A3, AscendDeviceType.A5])
def test_return_lse_keeps_scalar_path(monkeypatch, device_type):
    power_of_two = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setattr(dcp_a2a.triton, "next_power_of_2", power_of_two, raising=False)
    monkeypatch.setattr(dcp_a2a.triton, "cdiv", lambda a, b: (a + b - 1) // b, raising=False)
    monkeypatch.setattr(dcp_a2a, "is_950", lambda: device_type == AscendDeviceType.A5)
    monkeypatch.setattr(dcp_a2a, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(dcp_a2a, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(dcp_a2a, "get_vectorcore_num", lambda: 48)
    monkeypatch.setattr(dcp_a2a, "_A3_WARMING", False)
    monkeypatch.setattr(dcp_a2a, "_UNWARMED_LOGGED", set())
    key = (4, 16, 512, 16, 2048, 512, 1, 4, 1)
    monkeypatch.setattr(dcp_a2a, "_A3_STATIC_SIGNATURES", frozenset({key}))
    tiled, static, scalar, batched = Mock(), Mock(), Mock(), Mock()
    for name, launch in [
        ("rank_tiled", tiled),
        ("rank_tiled_static", static),
        ("", scalar),
        ("batched", batched),
    ]:
        suffix = "_" + name if name else ""
        kernel = Mock()
        kernel.__getitem__ = Mock(return_value=launch)
        monkeypatch.setattr(dcp_a2a, f"_fused_dcp_lse_combine{suffix}_kernel", kernel)
    with FakeTensorMode():
        recv = torch.empty(16, 4, 16, 516, dtype=torch.bfloat16, device="npu:0")
        local = torch.empty_strided((16, 4, 512), (2048, 512, 1), dtype=torch.bfloat16, device="npu:0")
        lse = torch.empty(16, 4, 1, dtype=torch.float32, device="npu:0")
        output = torch.empty(16, 4, 513, dtype=torch.bfloat16, device="npu:0")

        def do_launch(kernel, grid, *args, **kwargs):
            kernel[grid](*args, **kwargs)

        dcp_a2a._launch_fused_dcp_lse_combine(recv, output, 512, 1, True, local, lse, 48, do_launch)
    # Neither rank-tiled kernel writes the LSE lane, so return_lse keeps the
    # scalar kernel on every device.
    assert scalar.call_count == 1
    assert tiled.call_count + static.call_count + batched.call_count == 0
