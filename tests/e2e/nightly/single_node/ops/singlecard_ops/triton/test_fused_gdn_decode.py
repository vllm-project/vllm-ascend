# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.fused_gdn_decode import fused_gdn_decode
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.fixture(scope="module", autouse=True)
def initialize_device():
    torch.npu.set_device(0)
    init_device_properties_triton()


def make_inputs(batch, heads=2, value_heads=4, strided=False, state_dtype=torch.float32):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    # Sliced packed projections exercise the actual non-contiguous Q/K/V ABI.
    packed = torch.randn(batch, (2 * heads + value_heads) * 128, device="npu", dtype=dtype)
    q, k, v = packed.split([heads * 128, heads * 128, value_heads * 128], dim=-1)
    q, k, v = q.view(batch, heads, 128), k.view(batch, heads, 128), v.view(batch, value_heads, 128)
    gates = torch.randn(batch, value_heads * 2, device="npu", dtype=dtype)
    a, b = gates[:, ::2], gates[:, 1::2]
    if not strided:
        q, k, v, a, b = (x.contiguous() for x in (q, k, v, a, b))
    A_log = torch.randn(value_heads, device="npu")
    dt_bias = torch.randn(value_heads, device="npu")
    state = torch.randn(batch + 3, value_heads, 128, 128, device="npu", dtype=state_dtype) * 0.1
    index_storage = torch.zeros(batch * 2, device="npu", dtype=torch.int32)
    indices = index_storage[::2]
    indices.copy_(torch.arange(batch - 1, -1, -1, device="npu", dtype=torch.int32))
    boundaries = torch.arange(batch + 1, device="npu", dtype=torch.int32)
    return [q, k, v, a, b, A_log, dt_bias, state, indices, boundaries]


def reference_step(inputs):
    """Independent CPU golden, preserving the separate operators' rounding."""
    q, k, v, a, b, A_log, dt_bias, state, indices, boundaries = [x.cpu() for x in inputs]
    q = (q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype).float()
    k = (k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype).float()
    q = q.repeat_interleave(v.shape[1] // q.shape[1], dim=1) * 128**-0.5
    k = k.repeat_interleave(v.shape[1] // k.shape[1], dim=1)
    decay = torch.exp(-A_log.float().exp() * F.softplus(a.float() + dt_bias.float()))
    beta = b.float().sigmoid().to(b.dtype).float()
    out = torch.zeros_like(v)
    for token in range(int(boundaries[0]), int(boundaries[-1])):
        slot = int(indices[token])
        if slot < 0:
            continue
        h = state[slot].float() * decay[token, :, None, None]
        delta = (v[token].float() - (h * k[token, :, None, :]).sum(-1)) * beta[token, :, None]
        h = h + delta[:, :, None] * k[token, :, None, :]
        out[token] = (h * q[token, :, None, :]).sum(-1).to(v.dtype)
        state[slot] = h.to(state.dtype)
    return out, state


@pytest.mark.parametrize("batch", [1, 8, 32, 128])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_decode_matches_reference(batch, strided, state_dtype):
    inputs = make_inputs(batch, strided=strided, state_dtype=state_dtype)
    expected, expected_state = reference_step(inputs)
    actual = fused_gdn_decode(*inputs)
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-4, rtol=1e-2)
    torch.testing.assert_close(inputs[7].cpu(), expected_state, atol=1e-3, rtol=1e-2)


def test_decode_padding_and_zero_slot():
    inputs = make_inputs(8, strided=True)
    inputs[8].copy_(torch.tensor([0, 2, -1, 4, 0, 0, 0, 0], device="npu", dtype=torch.int32))
    inputs[9].copy_(torch.tensor([0, 1, 2, 3, 4, 4, 4, 4, 4], device="npu", dtype=torch.int32))
    initial = inputs[7].clone()
    expected, expected_state = reference_step(inputs)
    actual = fused_gdn_decode(*inputs)
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-4, rtol=1e-2)
    torch.testing.assert_close(inputs[7].cpu(), expected_state, atol=1e-5, rtol=1e-3)
    untouched = torch.tensor([1, 3, 5, 6, 7, 8, 9, 10], device="npu")
    torch.testing.assert_close(inputs[7][untouched], initial[untouched], atol=0, rtol=0)
    assert torch.count_nonzero(actual[4:]).item() == 0


def test_decode_preserves_bfloat16_beta_rounding():
    inputs = make_inputs(1)
    inputs[0].zero_()
    inputs[1].zero_()
    inputs[1][:, :, 0] = 1
    inputs[2].fill_(1)
    inputs[4].copy_(torch.tensor([[0.125, 0.25, -0.25, 1.0]], device="npu", dtype=torch.bfloat16))
    inputs[7].zero_()
    expected_beta = inputs[4].cpu().float().sigmoid().to(torch.bfloat16).float()
    fused_gdn_decode(*inputs)
    # With zero history and one-hot K, the updated state contains beta exactly.
    torch.testing.assert_close(inputs[7][0, :, 0, 0].cpu(), expected_beta[0], atol=0, rtol=0)


def test_decode_multistep_extreme_gates():
    inputs = make_inputs(8, strided=True)
    inputs[3][:, 0] = -30
    inputs[3][:, 1] = 30
    inputs[4][:, 0] = -30
    inputs[4][:, 1] = 30
    reference_inputs = [x.clone() for x in inputs]
    for _ in range(32):
        expected, expected_state = reference_step(reference_inputs)
        reference_inputs[7].copy_(expected_state)
        actual = fused_gdn_decode(*inputs)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(inputs[7].cpu(), expected_state, atol=1e-3, rtol=1e-2)


def test_decode_graph_replay_changes_slots_and_padding():
    inputs = make_inputs(8, strided=True)
    fused_gdn_decode(*inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fused_gdn_decode(*inputs)
    inputs[8].copy_(torch.tensor([1, 0, 3, 2, 0, 0, 0, 0], device="npu", dtype=torch.int32))
    inputs[9].copy_(torch.tensor([0, 1, 2, 3, 4, 4, 4, 4, 4], device="npu", dtype=torch.int32))
    expected, expected_state = reference_step(inputs)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, atol=5e-4, rtol=1e-2)
    torch.testing.assert_close(inputs[7].cpu(), expected_state, atol=1e-5, rtol=1e-3)


def test_decode_empty():
    inputs = make_inputs(0)
    initial = inputs[7].clone()
    assert fused_gdn_decode(*inputs).shape == (0, 4, 128)
    torch.testing.assert_close(inputs[7], initial)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 8, 32])
def test_decode_padded_state_pages(batch, state_dtype):
    inputs = make_inputs(batch, heads=4, value_heads=8, strided=True, state_dtype=state_dtype)
    slot_elements = 8 * 128 * 128
    padding = 256
    storage = torch.full((batch + 3, slot_elements + 2 * padding), 17, device="npu", dtype=state_dtype)
    state = storage[:, padding:-padding].view(batch + 3, 8, 128, 128)
    state.copy_(inputs[7])
    inputs[7] = state
    assert not state.is_contiguous()
    expected, expected_state = reference_step(inputs)
    actual = fused_gdn_decode(*inputs)
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-4, rtol=1e-2)
    torch.testing.assert_close(state.cpu(), expected_state, atol=1e-3, rtol=1e-2)
    assert torch.all(storage[:, :padding] == 17)
    assert torch.all(storage[:, -padding:] == 17)
