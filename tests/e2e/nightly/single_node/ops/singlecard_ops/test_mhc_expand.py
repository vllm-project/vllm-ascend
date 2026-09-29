# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True)
def require_mhc_expand():
    if not get_current_hardware_profile().supports(HardwareCapability.MHC_EXPAND):
        pytest.skip("VllmMhcExpand is built for A2")
    assert enable_custom_op()


def assert_bits_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.is_contiguous()
    # Compare on CPU as integers: NaN payloads and signed zeros must survive.
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mult", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "tokens,hidden",
    [(0, 17), (3, 0), (1, 1), (3, 17), (33, 64), (3, 4096), (16, 7168), (3, 8191), (3, 8192), (3, 8193)],
)
def test_mhc_expand(dtype, mult, tokens, hidden):
    for seed in (7, 19):
        torch.manual_seed(seed)
        x = torch.randn(tokens, hidden, dtype=dtype, device="npu")
        original = x.clone()
        actual = torch.ops._C_ascend.npu_mhc_expand(x, mult)
        assert_bits_equal(actual, x.unsqueeze(1).repeat(1, mult, 1))
        assert_bits_equal(x, original)
        if x.numel():
            assert actual.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_expand_bit_patterns(dtype):
    # All 65536 patterns, including NaNs and subnormals, without float conversion.
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).reshape(16, 4096)
    x = bits.view(dtype).to("npu")
    actual = torch.ops._C_ascend.npu_mhc_expand(x, 4)
    assert_bits_equal(actual, bits.view(dtype).unsqueeze(1).repeat(1, 4, 1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_expand_npu_graph(dtype):
    x = torch.randn(16, 7168, dtype=dtype, device="npu")
    for _ in range(3):
        torch.ops._C_ascend.npu_mhc_expand(x, 4)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        y = torch.ops._C_ascend.npu_mhc_expand(x, 4)
    for value in (1.0, -3.0, 0.0):
        x.fill_(value)
        graph.replay()
        assert_bits_equal(y, x.unsqueeze(1).repeat(1, 4, 1))


@pytest.mark.parametrize("device", ["npu", "meta"])
def test_mhc_expand_validation(device):
    x = torch.empty(3, 17, dtype=torch.float16, device=device)
    for mult in (0, -1):
        with pytest.raises(RuntimeError, match="positive"):
            torch.ops._C_ascend.npu_mhc_expand(x, mult)
    with pytest.raises(RuntimeError, match="rank-2"):
        torch.ops._C_ascend.npu_mhc_expand(x.unsqueeze(0), 4)
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops._C_ascend.npu_mhc_expand(x.t(), 4)
    with pytest.raises(RuntimeError, match="float16 and bfloat16"):
        torch.ops._C_ascend.npu_mhc_expand(x.float(), 4)


def test_mhc_expand_meta():
    x = torch.empty(3, 17, device="meta", dtype=torch.bfloat16)
    y = torch.ops._C_ascend.npu_mhc_expand(x, 4)
    assert y.shape == (3, 4, 17)
    assert y.device.type == "meta"
    assert y.dtype == x.dtype
    assert y.is_contiguous()


def test_mhc_expand_compile_dynamic():
    compiled = torch.compile(torch.ops._C_ascend.npu_mhc_expand.default, backend="eager", dynamic=True, fullgraph=True)
    for tokens in (3, 11):
        x = torch.randn(tokens, 17, device="npu", dtype=torch.bfloat16)
        assert_bits_equal(compiled(x, 4), x.unsqueeze(1).repeat(1, 4, 1))
