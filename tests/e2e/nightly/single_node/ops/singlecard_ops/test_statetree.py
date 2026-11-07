# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.statetree import native, operator
from vllm_ascend.ops.statetree.codec import pack
from vllm_ascend.ops.statetree.methods import StateTreeLinearMethod

pytest.importorskip("torch_npu")

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or not is_950(),
    reason="requires Ascend 950",
)


def _assert_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
    )


@pytest.mark.parametrize("rows", [1, 4, 8, 16, 32, 64, 65, 128, 129, 255])
def test_fused_gemm_is_bit_exact(rows: int) -> None:
    generator = torch.Generator().manual_seed(302 + rows)
    weight = torch.randn(384, 768, generator=generator).to(torch.bfloat16).npu()
    x = torch.randn(rows, 768, generator=generator).to(torch.bfloat16).npu()
    packed = pack(weight, device=weight.device)

    expected = torch.nn.functional.linear(x, weight)
    actual = native.linear(x, packed)

    _assert_bits_equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize("rows", [16, 65, 128, 129, 255, 256, 257, 1024])
def test_operator_graph_replay_reads_updated_input(rows: int) -> None:
    generator = torch.Generator().manual_seed(43)
    weight = torch.randn(384, 768, generator=generator).to(torch.bfloat16).npu()
    packed = pack(weight, device=weight.device)
    native.prepare(packed)
    x = torch.randn(rows, 768, generator=generator).to(torch.bfloat16).npu()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        actual = operator.apply(
            x,
            packed.payload,
            packed.shape,
            (packed.n_tile, packed.k_tile, packed.escape_slots),
        )

    for _ in range(3):
        x.copy_(torch.randn(rows, 768, generator=generator).to(torch.bfloat16).npu())
        graph.replay()
        torch.npu.synchronize()
        expected = torch.nn.functional.linear(x, weight)
        _assert_bits_equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize("rows", [256, 257, 512, 1024])
def test_large_prefill_native_gemm_is_bit_exact(rows: int) -> None:
    generator = torch.Generator().manual_seed(65)
    weight = torch.randn(384, 768, generator=generator).to(torch.bfloat16).npu()
    packed = pack(weight, device=weight.device)
    x = torch.randn(rows, 768, generator=generator).to(torch.bfloat16).npu()

    expected = torch.nn.functional.linear(x, weight)
    actual = operator.apply(
        x,
        packed.payload,
        packed.shape,
        (packed.n_tile, packed.k_tile, packed.escape_slots),
    )

    _assert_bits_equal(actual.cpu(), expected.cpu())


def test_vllm_method_replaces_dense_weight_and_preserves_bits() -> None:
    generator = torch.Generator().manual_seed(91)
    dense = torch.randn(2048, 1024, generator=generator).to(torch.bfloat16).npu()
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(dense.clone(), requires_grad=False),
    )
    layer.register_parameter("bias", None)
    layer.prefix = "model.layers.0.mlp.down_proj"
    layer.custom_op = None
    method = StateTreeLinearMethod()

    method.process_weights_after_loading(layer)

    assert layer.weight.numel() == 0
    assert layer._statetree_payload.numel() < dense.numel() * dense.element_size()
    for rows in (1, 4, 8, 16, 32, 65, 128, 129, 255, 256, 513):
        x = torch.randn(rows, 1024, generator=generator).to(torch.bfloat16).npu()
        expected = torch.nn.functional.linear(x, dense)
        actual = method.apply(layer, x)
        _assert_bits_equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize("shuffle", [False, True])
def test_direct_device_decode_preserves_every_bf16_pattern(shuffle: bool) -> None:
    weight = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    if shuffle:
        weight = weight[torch.randperm(weight.numel(), generator=torch.Generator().manual_seed(7))]
    weight = weight.reshape(256, 256)
    packed = pack(weight)
    _assert_bits_equal(native.decode(packed).cpu(), weight)


def test_direct_decode_graph_preserves_padding_and_updated_payload() -> None:
    generator = torch.Generator().manual_seed(409)
    first = torch.randn(208, 256, generator=generator).to(torch.bfloat16)
    second = torch.randn(208, 256, generator=generator).to(torch.bfloat16)
    packed = pack(first, n_tile=192, k_tile=128)
    replacement = pack(second, n_tile=192, k_tile=128)
    assert packed.escape_slots == replacement.escape_slots
    native.prepare(packed)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        restored = native.decode(packed)
    stream.synchronize()
    packed.payload.copy_(replacement.payload)
    graph.replay()
    torch.npu.synchronize()
    _assert_bits_equal(restored.cpu(), second)
