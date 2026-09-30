# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.glm5next.ops.mhc_ops import hc_expand
from vllm_ascend.ops.mhc import mhc_expand
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
@pytest.mark.parametrize("tokens", [16, 17])
@pytest.mark.parametrize("entry", ["raw", "helper", "glm"])
def test_mhc_expand_nz_storage(dtype, tokens, entry):
    # The config property is write-only in torch_npu; preserve its raw option.
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    assert previous is not None
    try:
        torch.npu.config.allow_internal_format = True
        # Cover all 16-bit payloads and a padded NZ row boundary.
        bits = (torch.arange(tokens * 4096, dtype=torch.int32) % 65536 - 32768).to(torch.int16)
        expected = bits.view(dtype).reshape(tokens, 4096)
        x = torch_npu.npu_format_cast(expected.to("npu"), 29)
        assert torch_npu.get_npu_format(x) == 29
        assert x.is_contiguous()
        fn = {"raw": torch.ops._C_ascend.npu_mhc_expand, "helper": mhc_expand, "glm": hc_expand}[entry]
        assert_bits_equal(fn(x, 4), expected.unsqueeze(1).repeat(1, 4, 1))
        assert_bits_equal(x, expected)
        assert torch_npu.get_npu_format(x) == 29
    finally:
        torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": previous.decode()})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mult", [1, 2, 4, 8])
@pytest.mark.parametrize("hidden", [1023, 1025, 8191, 8193, 16385, 32769])
def test_mhc_expand_unaligned_output_ownership(dtype, mult, hidden):
    # Offset the contiguous input, and vary every row/stream boundary's values.
    # Repeated launches help expose races between neighboring output tiles.
    tokens = 5
    for seed in (7, 19, 43):
        generator = torch.Generator().manual_seed(seed)
        bits = torch.randint(-32768, 32768, (tokens * hidden + 1,), dtype=torch.int16, generator=generator)
        x = bits.view(dtype).to("npu")[1:].reshape(tokens, hidden)
        expected = x.cpu().unsqueeze(1).repeat(1, mult, 1)
        for _ in range(3):
            assert_bits_equal(torch.ops._C_ascend.npu_mhc_expand(x, mult), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,hidden", [(16, 7168), (3, 8193), (5, 32769)])
def test_mhc_expand_npu_graph(dtype, tokens, hidden):
    x = torch.randn(tokens, hidden, dtype=dtype, device="npu")
    for _ in range(3):
        torch.ops._C_ascend.npu_mhc_expand(x, 4)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        y = torch.ops._C_ascend.npu_mhc_expand(x, 4)
    for seed in (7, 19, 43):
        torch.manual_seed(seed)
        # Distinct row/column values also detect incorrect input mapping.
        x.normal_()
        graph.replay()
        assert_bits_equal(y, x.unsqueeze(1).repeat(1, 4, 1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_expand_addresses_and_streams(dtype):
    # Keep every allocation live: dispatch must preserve both addresses
    # and must not overwrite an earlier result or share state between streams.
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    pending = []
    for stream_index, stream in enumerate(streams):
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for iteration in range(6):
                offset = iteration % 3
                generator = torch.Generator().manual_seed(101 + 11 * stream_index + iteration)
                bits = torch.randint(-32768, 32768, (7 * 4096 + offset,), dtype=torch.int16, generator=generator)
                x = bits.view(dtype).to("npu")[offset:].reshape(7, 4096)
                expected = bits[offset:].view(dtype).reshape(7, 4096)
                y = torch.ops._C_ascend.npu_mhc_expand(x, 4)
                pending.append((x, y, expected))
    torch.npu.synchronize()
    for x, y, expected in pending:
        assert_bits_equal(y, expected.unsqueeze(1).repeat(1, 4, 1))
        assert_bits_equal(x, expected)


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_expand_temporary_inputs(dtype):
    # Only retain outputs on the caller. Submitted work must safely consume each
    # temporary input while later allocations put pressure on its storage.
    pending = []
    for iteration in range(12):
        generator = torch.Generator().manual_seed(401 + iteration)
        bits = torch.randint(-32768, 32768, (7, 4096), dtype=torch.int16, generator=generator)
        expected = bits.view(dtype)
        output = torch.ops._C_ascend.npu_mhc_expand(expected.to("npu"), 4)
        pending.append((output, expected))
    torch.npu.synchronize()
    for output, expected in pending:
        assert_bits_equal(output, expected.unsqueeze(1).repeat(1, 4, 1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("change_output", [False, True])
def test_mhc_expand_metadata_snapshot(dtype, change_output):
    pending = []
    for iteration in range(12):
        generator = torch.Generator().manual_seed(601 + iteration)
        bits = torch.randint(-32768, 32768, (7, 4096), dtype=torch.int16, generator=generator)
        expected = bits.view(dtype)
        x = expected.to("npu")
        output = torch.ops._C_ascend.npu_mhc_expand(x, 4)
        # Metadata mutation is immediate on the caller; both dispatch paths must
        # finish reading the original descriptor before returning.
        if change_output:
            output.transpose_(0, 2)
        else:
            x.transpose_(0, 1)
        pending.append((x, output, expected))
    torch.npu.synchronize()
    for _, output, expected in pending:
        if change_output:
            output = output.transpose(0, 2)
        assert_bits_equal(output, expected.unsqueeze(1).repeat(1, 4, 1))


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


@pytest.mark.parametrize("expand", [mhc_expand, hc_expand], ids=["helper", "glm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 4096), (16, 7168), (128, 4096), (3, 8193), (3, 17), (0, 17), (3, 0)])
def test_mhc_expand_dispatch(expand, dtype, shape, monkeypatch):
    x = torch.randn(shape, dtype=dtype, device="npu")
    original = torch.ops._C_ascend.npu_mhc_expand_if_supported
    calls = []

    def traced(x, mult):
        result = original(x, mult)
        calls.append((tuple(x.shape), mult, result is not None))
        return result

    monkeypatch.setattr(torch.ops._C_ascend, "npu_mhc_expand_if_supported", traced)
    assert_bits_equal(expand(x, 4), x.unsqueeze(1).repeat(1, 4, 1))
    assert calls == [(shape, 4, x.numel() > 0 and shape[1] % 16 == 0)]


@pytest.mark.parametrize("expand", [mhc_expand, hc_expand], ids=["helper", "glm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mult", [2, 8])
def test_mhc_expand_other_multipliers_use_native(expand, dtype, mult, monkeypatch):
    original = torch.ops._C_ascend.npu_mhc_expand_if_supported
    calls = []

    def checked(x, mult):
        result = original(x, mult)
        assert result is None
        calls.append(mult)
        return result

    monkeypatch.setattr(torch.ops._C_ascend, "npu_mhc_expand_if_supported", checked)
    x = torch.randn(3, 64, dtype=dtype, device="npu")
    assert_bits_equal(expand(x, mult), x.unsqueeze(1).repeat(1, mult, 1))
    assert calls == [mult]


@pytest.mark.parametrize("expand", [mhc_expand, hc_expand], ids=["helper", "glm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_expand_gradient_fallback(expand, dtype, monkeypatch):
    def unexpected_custom(*args):
        raise AssertionError("Gradient-requiring input must use native expansion")

    monkeypatch.setattr(torch.ops._C_ascend, "npu_mhc_expand_if_supported", unexpected_custom)
    x = torch.randn(3, 64, dtype=dtype, device="npu", requires_grad=True)
    expand(x, 4).sum().backward()
    assert_bits_equal(x.grad, torch.full((3, 64), 4, dtype=dtype))


@pytest.mark.parametrize("expand", [mhc_expand, hc_expand], ids=["helper", "glm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_expand_helper_graph(expand, dtype):
    x = torch.randn(16, 4096, dtype=dtype, device="npu")
    for _ in range(3):
        expand(x, 4)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        y = expand(x, 4)
    for value in (2.0, -1.0):
        x.fill_(value)
        graph.replay()
        assert_bits_equal(y, x.unsqueeze(1).repeat(1, 4, 1))


@pytest.mark.parametrize("device", ["npu", "meta"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "reason",
    ["supported", "dtype", "rank", "noncontiguous", "unaligned", "gradient", "empty", "zero_hidden", "trivial", "mult"],
)
def test_mhc_expand_optional_eligibility(device, dtype, reason):
    shape = {"unaligned": (3, 17), "empty": (0, 64), "zero_hidden": (3, 0)}.get(reason, (3, 64))
    x = torch.empty(shape, device=device, dtype=torch.float32 if reason == "dtype" else dtype)
    if reason == "rank":
        x = x.unsqueeze(0)
    elif reason == "noncontiguous":
        x = x.t()
    elif reason == "gradient":
        x.requires_grad_()
    mult = {"trivial": 1, "mult": 8}.get(reason, 4)
    result = torch.ops._C_ascend.npu_mhc_expand_if_supported(x, mult)
    if reason != "supported":
        assert result is None
    else:
        assert result.shape == (3, 4, 64)
        assert result.dtype == dtype
        assert result.device == x.device
        assert result.is_contiguous()
        if device == "npu":
            assert_bits_equal(result, x.unsqueeze(1).repeat(1, 4, 1))
            assert result.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_expand_fallback_aliasing(dtype):
    x = torch.randn(3, 64, device="npu", dtype=dtype)
    y = mhc_expand(x, 1)
    assert y.data_ptr() == x.data_ptr()
    y.fill_(2)
    assert_bits_equal(x, torch.full((3, 64), 2, dtype=dtype))
    empty = x[:0]
    result = mhc_expand(empty, 4)
    assert result.untyped_storage().data_ptr() == empty.untyped_storage().data_ptr()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_expand_optional_compile_dynamic(dtype):
    def expand(x, mult):
        result = torch.ops._C_ascend.npu_mhc_expand_if_supported(x, mult)
        return result if result is not None else x.unsqueeze(1).expand(-1, mult, -1).contiguous()

    compiled = torch.compile(expand, backend="eager", dynamic=True, fullgraph=True)
    for tokens, hidden in ((3, 64), (11, 64), (7, 128), (3, 17), (5, 33), (0, 64), (3, 0)):
        x = torch.randn(tokens, hidden, device="npu", dtype=dtype)
        assert_bits_equal(compiled(x, 4), x.unsqueeze(1).repeat(1, 4, 1))


@pytest.mark.parametrize("expand", [mhc_expand, hc_expand], ids=["helper", "glm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_expand_helper_compile_dynamic(expand, dtype):
    compiled = torch.compile(expand, backend="eager", dynamic=True, fullgraph=True)
    for tokens, hidden in ((3, 64), (11, 128), (3, 17), (0, 64)):
        x = torch.randn(tokens, hidden, device="npu", dtype=dtype)
        assert_bits_equal(compiled(x, 4), x.unsqueeze(1).repeat(1, 4, 1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [4096, 8192])
@pytest.mark.parametrize("tokens", [39, 40, 41, 79, 80, 81, 127, 128, 129, 159, 160, 161, 205])
def test_mhc_expand_row_group_boundaries(dtype, hidden, tokens):
    generator = torch.Generator().manual_seed(tokens + hidden)
    bits = torch.randint(-32768, 32768, (tokens, hidden), dtype=torch.int16, generator=generator)
    x = bits.view(dtype).to("npu")
    for mult in (1, 2, 4, 8):
        actual = torch.ops._C_ascend.npu_mhc_expand(x, mult)
        assert_bits_equal(actual, bits.view(dtype).unsqueeze(1).repeat(1, mult, 1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_expand_stream_core_limit(dtype):
    # Core-controlled calls retain the ACLNN adapter path. Use a separate stream
    # and restore its configuration so subsequent callers retain their limits.
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    generator = torch.Generator().manual_seed(1401)
    bits = torch.randint(-32768, 32768, (33, 8193), dtype=torch.int16, generator=generator)
    with torch.npu.stream(stream):
        try:
            torch.npu.set_stream_limit(stream, vector_num=8)
            x = bits.view(dtype).to("npu")
            result = torch.ops._C_ascend.npu_mhc_expand(x, 4)
            assert_bits_equal(result, bits.view(dtype).unsqueeze(1).repeat(1, 4, 1))
        finally:
            stream.synchronize()
            torch.npu.reset_stream_limit(stream)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("separate_stream", [False, True], ids=["current", "separate"])
def test_mhc_expand_queued_producers_and_consumers(dtype, separate_stream):
    stream = torch.npu.Stream() if separate_stream else torch.npu.current_stream()
    stream.wait_stream(torch.npu.current_stream())
    pending = []
    with torch.npu.stream(stream):
        x = torch.empty(17, 4096, device="npu", dtype=dtype)
        for value in range(12):
            # No intermediate synchronization: every expansion must observe its
            # producer, and the consumer must finish before a later write to x.
            x.fill_(value)
            y = torch.ops._C_ascend.npu_mhc_expand(x, 4)
            pending.append((y + 3, value + 3))
    stream.synchronize()
    for actual, expected in pending:
        assert_bits_equal(actual, torch.full((17, 4, 4096), expected, dtype=dtype))
