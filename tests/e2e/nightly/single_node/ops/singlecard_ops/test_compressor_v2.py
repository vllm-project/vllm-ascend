# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CompressorV2 numerical and mutation checks; run on Ascend A2/A3."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
# Loading the extension registers torch.ops; it exposes no typed Python API.
importlib.import_module("vllm_ascend.vllm_ascend_C")


def _reference(x, wkv, wgate, state, blocks, offsets, used, starts):
    # Deliberately use explicit absolute positions, not native compact offsets.
    kv = x.float() @ wkv.float().T
    scores = x.float() @ wgate.float().T
    width = wkv.shape[0]
    output = []
    for block, offset, count, start in zip(blocks, offsets, used, starts):
        history = state[block].clone()
        for local in range(count):
            position = start + local
            row = position % state.shape[1]
            history[row, :width] = kv[offset + local]
            history[row, width:] = scores[offset + local]
            if position % 2:
                pair = history[torch.tensor([(position - 1) % state.shape[1], row])]
                output.append((pair[:, :width] * pair[:, width:].softmax(0)).sum(0))
        state[block].copy_(history)
    if output:
        return torch.stack(output).to(x.dtype)
    return x.new_empty((0, width))


@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("hidden", [1024, 5120])
@pytest.mark.parametrize("length,start", [(1, 0), (1, 1), (5, 31), (33, 1), (67, 1), (1003, 0), (1003, 1)])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_compressor_v2_ring(width, hidden, length, start, strided):
    if not get_current_hardware_profile().supports(HardwareCapability.DSV41_RING_COMPRESSOR):
        pytest.skip("32-row ring adapter currently targets A2/A3")
    torch.manual_seed(17)
    # Mix one long request, a single-token request, and an unused padded slot.
    blocks, offsets, used, starts = [2, 1, 0], [0, length, length + 1], [length, 1, 0], [start, 7, 0]
    x = torch.randn(length + 4, hidden, dtype=torch.bfloat16) * 0.1
    wkv = torch.randn(width, hidden, dtype=torch.bfloat16) * 0.02
    wgate = torch.randn_like(wkv) * 0.02
    storage = torch.randn(3, 64 if strided else 32, 2 * width, dtype=torch.float32) * 0.1
    initial = storage.clone()
    expected_state = storage[:, :32].clone()
    expected = _reference(x, wkv, wgate, expected_state, blocks, offsets, used, starts)
    device_storage = storage.npu()
    state = device_storage[:, :32]
    controls = [torch.tensor(v, dtype=torch.int32, device="npu") for v in (blocks, offsets + [len(x)], used, starts)]
    actual = torch.ops._C_ascend.compressor_v2(x.npu(), wkv.npu(), wgate.npu(), state, *controls, 2)
    assert actual.shape == (min(len(x), len(x) // 2 + len(blocks)), width)
    torch.testing.assert_close(actual[: len(expected)].cpu(), expected, rtol=0.02, atol=0.002)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=1e-4, atol=1e-5)
    # seqused=0 must leave the framework's null page untouched.
    torch.testing.assert_close(device_storage[0].cpu(), initial[0], rtol=0, atol=0)
    if strided:
        torch.testing.assert_close(device_storage[:, 32:].cpu(), initial[:, 32:], rtol=0, atol=0)


@torch.inference_mode()
def test_compressor_v2_graph_replay():
    if not get_current_hardware_profile().supports(HardwareCapability.DSV41_RING_COMPRESSOR):
        pytest.skip("32-row ring adapter currently targets A2/A3")
    x = torch.randn(4, 1024, dtype=torch.bfloat16, device="npu") * 0.1
    weights = torch.randn(128, 1024, dtype=torch.bfloat16, device="npu") * 0.02
    state = torch.zeros(2, 32, 256, device="npu")
    blocks = torch.tensor([1], dtype=torch.int32, device="npu")
    offsets = torch.tensor([0, 4], dtype=torch.int32, device="npu")
    used = torch.tensor([4], dtype=torch.int32, device="npu")
    starts = torch.tensor([0], dtype=torch.int32, device="npu")

    def run(cache):
        return torch.ops._C_ascend.compressor_v2(x, weights, weights, cache, blocks, offsets, used, starts, 2)

    for _ in range(3):
        run(state)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run(state)
    for count, start in [(1, 1), (4, 31), (0, 35)]:
        used.fill_(count)
        starts.fill_(start)
        before = state.clone()
        expected = run(before)
        graph.replay()
        torch.npu.synchronize()
        groups = (start + count) // 2 - start // 2
        torch.testing.assert_close(captured[:groups], expected[:groups], rtol=0, atol=0)
        torch.testing.assert_close(state, before, rtol=0, atol=0)


@torch.inference_mode()
def test_compressor_v2_speculative_compact_mapping():
    if not get_current_hardware_profile().supports(HardwareCapability.DSV41_RING_COMPRESSOR):
        pytest.skip("32-row ring adapter currently targets A2/A3")
    from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41Compressor

    torch.manual_seed(17)
    x = torch.randn(24, 5120, dtype=torch.bfloat16, device="npu") * 0.1
    wkv = torch.randn(512, 5120, dtype=torch.bfloat16, device="npu") * 0.02
    wgate = torch.randn_like(wkv) * 0.02
    state = torch.randn(5, 32, 1024, device="npu") * 0.1
    ring = torch.tensor(
        [[0] * 4, [6] * 4, [0, 6, 12, 18], [0, 6, 12, 18], [1, 2, 3, 4]],
        dtype=torch.int32,
        device="npu",
    )
    complete = (torch.arange(6, device="npu").repeat(4) % 2) == 1
    metadata = SimpleNamespace(c2_ring_metadata=ring, c2_complete_mask=complete)
    adapter = SimpleNamespace(
        wkv=SimpleNamespace(weight=wkv),
        wgate=SimpleNamespace(weight=wgate),
        state_cache=SimpleNamespace(kv_cache=[state.unsqueeze(-2)]),
        ratio=2,
        _ring_pooled=torch.empty(24, 512, dtype=torch.bfloat16, device="npu"),
        norm=lambda value: value,
    )

    def run():
        return DeepseekV41Compressor.compress_native(adapter, x, metadata)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for start in (0, 1):
        ring[0].fill_(start)
        complete.copy_(((torch.arange(6, device="npu").repeat(4) + start) % 2) == 1)
        expected_state = state.cpu()
        expected = _reference(
            x.cpu(),
            wkv.cpu(),
            wgate.cpu(),
            expected_state,
            [1, 2, 3, 4],
            [0, 6, 12, 18],
            [6] * 4,
            [start] * 4,
        )
        graph.replay()
        torch.npu.synchronize()
        actual, mask = captured.cpu(), complete.cpu()
        torch.testing.assert_close(actual[mask], expected, rtol=0.02, atol=0.002)
        assert torch.count_nonzero(actual[~mask]) == 0
        torch.testing.assert_close(state.cpu(), expected_state, rtol=1e-4, atol=1e-5)


def test_compressor_v2_meta_and_schema():
    x = torch.empty(9, 1024, device="meta", dtype=torch.bfloat16)
    weights = torch.empty(128, 1024, device="meta", dtype=x.dtype)
    state = torch.empty(3, 32, 256, device="meta")
    offsets = torch.empty(3, device="meta", dtype=torch.int32)
    result = torch.ops._C_ascend.compressor_v2(x, weights, weights, state, None, offsets, None, None, 2)
    assert result.shape == (6, 128) and result.dtype == x.dtype
    schema = torch.ops._C_ascend.compressor_v2.default._schema
    assert schema.arguments[3].alias_info.is_write
    with pytest.raises(RuntimeError, match="positive"):
        torch.ops._C_ascend.compressor_v2(x, weights, weights, state, None, offsets, None, None, 0)
