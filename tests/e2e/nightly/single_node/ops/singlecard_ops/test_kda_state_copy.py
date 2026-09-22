# SPDX-License-Identifier: Apache-2.0
"""KDA state gather/clear and scatter: payload, page gaps, graph and 64-bit offsets."""

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 1, 129), (12, 128, 128)])
@torch.inference_mode()
def test_kda_state_copy_preserves_gaps_offsets_and_invalid_rows(dtype, index_dtype, shape):
    payload = shape[0] * shape[1] * shape[2]
    stride, offset = 3 * payload + 32, 16
    backing = torch.full((7 * stride + offset,), -23, dtype=dtype, device="npu")
    state = backing.as_strided((7, *shape), (stride, shape[1] * shape[2], shape[2], 1), offset)
    for row in range(7):
        state[row].fill_(row + 1)
    original = backing.cpu()
    indices = torch.tensor([6, 1, -1, 3, 7], dtype=index_dtype, device="npu")
    flags = torch.tensor([True, False, True, True, False], device="npu")
    packed = torch.full((5, *shape), 99, dtype=dtype, device="npu")
    torch.ops._C_ascend.kda_state_copy(state, packed, indices, flags, False)
    expected = torch.zeros((5, *shape), dtype=dtype)
    expected[0].fill_(7)
    expected[3].fill_(4)
    torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(backing.cpu(), original, rtol=0, atol=0)

    # Without flags every valid row is gathered, including duplicate reads.
    repeated = torch.tensor([1, 1, -1, 0, 7], dtype=index_dtype, device="npu")
    torch.ops._C_ascend.kda_state_copy(state, packed, repeated, None, False)
    expected.zero_()
    expected[0:2].fill_(2)
    expected[3].fill_(1)
    torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)

    updates = torch.arange(10, 15, dtype=torch.float32, device="npu").to(dtype)
    packed.copy_(updates[:, None, None, None].expand_as(packed))
    # Initial-state flags must not suppress final-state writes during scatter.
    torch.ops._C_ascend.kda_state_copy(state, packed, indices, flags, True)
    expected_backing = original.clone()
    expected_state = expected_backing.as_strided(state.shape, state.stride(), offset)
    expected_state[6].fill_(10)
    expected_state[1].fill_(11)
    expected_state[3].fill_(13)
    torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)


@torch.inference_mode()
def test_kda_state_copy_graph_replay_changed_indices_and_flags():
    backing = torch.full((12, 12, 128, 128), -23, dtype=torch.float32, device="npu")
    state = backing[1::3]
    for row in range(4):
        state[row].fill_(row + 1)
    indices = torch.tensor([0, 3, -1, 4], dtype=torch.int32, device="npu")
    flags = torch.ones(4, dtype=torch.bool, device="npu")
    packed = torch.empty((4, 12, 128, 128), device="npu")
    torch.ops._C_ascend.kda_state_copy(state, packed, indices, flags, False)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        torch.ops._C_ascend.kda_state_copy(state, packed, indices, flags, False)
    indices.copy_(torch.tensor([2, 1, 3, -1], dtype=torch.int32, device="npu"))
    flags.copy_(torch.tensor([True, False, True, True], device="npu"))
    graph.replay()
    expected = torch.zeros_like(packed, device="cpu")
    expected[0].fill_(3)
    expected[2].fill_(4)
    torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)
    assert torch.all(backing[0::3] == -23) and torch.all(backing[2::3] == -23)


@torch.inference_mode()
def test_kda_state_copy_address_beyond_four_gib():
    # Only touch small guards and two payloads; no 4 GiB host copy is needed.
    stride, offset, payload = 2**30 + 32, 16, 16
    backing = torch.empty(stride + offset + payload + 32, dtype=torch.float32, device="npu")
    backing[:64].fill_(-23)
    backing[-64:].fill_(-23)
    state = backing.as_strided((2, 1, 2, 8), (stride, 16, 8, 1), offset)
    state[0].fill_(3)
    state[1].fill_(7)
    assert state.stride(0) * state.element_size() > 2**32
    indices = torch.tensor([1, 0, -1, 2], dtype=torch.int32, device="npu")
    packed = torch.empty((4, 1, 2, 8), device="npu")
    torch.ops._C_ascend.kda_state_copy(state, packed, indices, None, False)
    expected = torch.zeros((4, 1, 2, 8))
    expected[0].fill_(7)
    expected[1].fill_(3)
    torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)
    packed.add_(10)
    torch.ops._C_ascend.kda_state_copy(state, packed, indices, None, True)
    torch.testing.assert_close(state[0].cpu(), torch.full((1, 2, 8), 13.0), rtol=0, atol=0)
    torch.testing.assert_close(state[1].cpu(), torch.full((1, 2, 8), 17.0), rtol=0, atol=0)
    assert torch.all(backing[:offset] == -23) and torch.all(backing[-32:] == -23)


@torch.inference_mode()
def test_kda_state_copy_empty_selection_and_inner_stride_rejection():
    state = torch.ones((2, 2, 3, 4), device="npu")
    empty = torch.empty((0, 2, 3, 4), device="npu")
    indices = torch.empty(0, dtype=torch.int32, device="npu")
    torch.ops._C_ascend.kda_state_copy(state, empty, indices, None, False)
    torch.ops._C_ascend.kda_state_copy(state, empty, indices, None, True)
    assert torch.all(state == 1)
    with pytest.raises(RuntimeError, match="dense inner"):
        torch.ops._C_ascend.kda_state_copy(
            state.transpose(-1, -2), torch.empty((0, 2, 4, 3), device="npu"), indices, None, False
        )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_kda_state_copy_noncontiguous_indices_and_flags(index_dtype):
    state = torch.arange(4 * 2 * 3 * 4, dtype=torch.float32, device="npu").reshape(4, 2, 3, 4)
    original = state.cpu()
    indices = torch.tensor([99, 3, 99, 1, 99, -1, 99, 2], dtype=index_dtype, device="npu")[1::2]
    flags = torch.tensor([False, True, True, False, False, True, False, True], device="npu")[1::2]
    assert not indices.is_contiguous() and not flags.is_contiguous()
    packed = torch.empty((4, 2, 3, 4), device="npu")
    torch.ops._C_ascend.kda_state_copy(state, packed, indices, flags, False)
    expected = torch.zeros_like(packed, device="cpu")
    expected[0], expected[3] = original[3], original[2]
    torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)
    packed.add_(100)
    torch.ops._C_ascend.kda_state_copy(state, packed, indices, None, True)
    original[3], original[1], original[2] = expected[0] + 100, expected[1] + 100, expected[3] + 100
    torch.testing.assert_close(state.cpu(), original, rtol=0, atol=0)
