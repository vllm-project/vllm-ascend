# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.rearrange_qkv import rearrange_mixed_qkv
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def reference(x, dims=(1024, 1024, 3072)):
    return torch.cat([part.reshape(-1) for part in x.split(dims, dim=-1)])


@pytest.fixture(scope="module", autouse=True)
def load_op():
    enable_custom_op()
    if get_ascend_device_type() not in {AscendDeviceType.A2, AscendDeviceType.A3}:
        pytest.skip("npu_rearrange_qkv is only built for A2 and A3")


@pytest.mark.parametrize("tokens", [0, 1, 4, 15, 16, 17, 19, 20, 21, 64, 319, 320, 321, 1024, 4096])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@torch.inference_mode()
def test_bitwise_copy(tokens, offset, dtype):
    # Arbitrary bits cover NaNs, infinities, signed zero, and subnormals.
    bits = torch.randint(-32768, 32768, (tokens + offset, 5120), dtype=torch.int16)
    mixed_qkv = bits.view(dtype).npu()[offset:]
    actual = torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, 1024, 1024, 3072)

    assert actual.shape == (tokens * 5120,)
    assert actual.dtype == dtype
    assert torch.equal(actual.view(torch.int16).cpu(), reference(bits[offset:]))


@pytest.mark.parametrize(
    ("q_dim", "v_dim", "tp_size"),
    [(128, 256, 1), (512, 1536, 4), (1024, 3072, 2), (2048, 6144, 1)],
)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@torch.inference_mode()
def test_gdn_dispatch(q_dim, v_dim, tp_size, dtype):
    mixed_qkv = torch.randn(37, 2 * q_dim + v_dim, device="npu", dtype=dtype)

    def unexpected_fallback(_):
        pytest.fail("Supported GDN layout did not use npu_rearrange_qkv")

    layer = SimpleNamespace(
        key_dim=q_dim * tp_size,
        value_dim=v_dim * tp_size,
        tp_size=tp_size,
        head_k_dim=128,
        head_v_dim=128,
        rearrange_mixed_qkv=unexpected_fallback,
    )
    outputs = rearrange_mixed_qkv(layer, mixed_qkv)
    expected_parts = mixed_qkv.split([q_dim, q_dim, v_dim], dim=-1)
    expected_heads = (q_dim // 128, q_dim // 128, v_dim // 128)
    for output, expected, heads in zip(outputs, expected_parts, expected_heads):
        assert output.shape == (1, 37, heads, 128)
        assert output.is_contiguous()
        assert torch.equal(output.reshape(37, -1).view(torch.int16), expected.contiguous().view(torch.int16))


@pytest.mark.parametrize(
    ("dims", "tokens"),
    [((1024, 1024, 3072), 321), ((16, 48, 80), 37), ((16, 16, 81904), 3)],
)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@torch.inference_mode()
def test_nondefault_stream_and_graph_replay(dims, tokens, dtype):
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        mixed_qkv = torch.randn(tokens, sum(dims), device="npu", dtype=dtype)
        for _ in range(3):
            eager = torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, *dims)
    stream.synchronize()
    assert torch.equal(eager.cpu(), reference(mixed_qkv.cpu(), dims))

    graph = torch.npu.NPUGraph()
    with torch.npu.stream(stream):
        with torch.npu.graph(graph, stream=stream):
            captured = torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, *dims)
        for _ in range(3):
            mixed_qkv.normal_()
            graph.replay()
            stream.synchronize()
            assert torch.equal(captured.cpu(), reference(mixed_qkv.cpu(), dims))


@pytest.mark.parametrize("kind", ["dtype", "width", "rank", "stride"])
def test_reject_unsafe_inputs(kind):
    if kind == "dtype":
        mixed_qkv = torch.empty(2, 5120, device="npu", dtype=torch.float32)
    elif kind == "width":
        mixed_qkv = torch.empty(2, 2560, device="npu", dtype=torch.bfloat16)
    elif kind == "rank":
        mixed_qkv = torch.empty(1, 2, 5120, device="npu", dtype=torch.bfloat16)
    else:
        mixed_qkv = torch.empty(2, 10240, device="npu", dtype=torch.bfloat16)[:, ::2]

    with pytest.raises(RuntimeError, match="npu_rearrange_qkv requires"):
        torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, 1024, 1024, 3072)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_meta(dtype):
    mixed_qkv = torch.empty(17, 5120, device="meta", dtype=dtype)
    output = torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, 1024, 1024, 3072)
    assert output.shape == (17 * 5120,)
    assert output.dtype == mixed_qkv.dtype
    assert output.device.type == "meta"


@pytest.mark.parametrize(
    ("dims", "tokens"),
    [
        ((16, 16, 16), 65),
        ((96, 144, 240), 37),
        ((256, 512, 1024), 641),
        ((512, 512, 1536), 321),
        ((2048, 2048, 6144), 161),
        ((16, 16, 81888), 21),
        ((16, 16, 81904), 21),
        ((163840, 16, 32), 3),
        ((16, 1048576, 32), 3),
        ((16, 32, 1048576), 3),
    ],
)
@pytest.mark.parametrize("offset", [0, 16])
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@torch.inference_mode()
def test_aligned_dynamic_widths(dims, tokens, offset, dtype):
    width = sum(dims)
    bits = torch.randint(-32768, 32768, (tokens * width + offset,), dtype=torch.int16)
    mixed_qkv = bits.view(dtype).npu()[offset:].view(tokens, width)

    actual = torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, *dims)
    expected = reference(bits[offset:].view(tokens, width), dims)

    assert torch.equal(actual.view(torch.int16).cpu(), expected)


@pytest.mark.parametrize("dims", [(17, 16, 16), (16, 17, 16), (16, 16, 17), (0, 16, 16), (-16, 16, 32)])
def test_reject_unaligned_or_nonpositive_widths(dims):
    mixed_qkv = torch.empty(2, sum(dims), device="npu", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="npu_rearrange_qkv requires"):
        torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, *dims)


def test_reject_unaligned_address():
    mixed_qkv = torch.empty(2 * 48 + 1, device="npu", dtype=torch.bfloat16)[1:].view(2, 48)
    with pytest.raises(RuntimeError, match="32-byte-aligned"):
        torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, 16, 16, 16)
