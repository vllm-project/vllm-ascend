# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.models.glm5next.ops.causal_conv1d import (
    CONV_STATE_COPY_MAX_PROGRAMS,
    _copy_conv_state,
    causal_conv1d,
)


@pytest.mark.parametrize("dim_first", [False, True])
def test_conv_state_copy_masks_invalid_slots_and_preserves_shared_pages(dim_first):
    slots, width, dim = 4, 6, 384
    # Leave a gap after each physical state and preserve the alternate layout.
    backing = torch.arange(slots * width * dim * 2, dtype=torch.float32, device="npu")
    strides = (width * dim * 2, 1, width) if dim_first else (width * dim * 2, dim, 1)
    cache = backing.as_strided((slots, width, dim), strides)
    saved = backing.clone()
    indices = torch.tensor([2, -1, slots, 1, 3], dtype=torch.int32, device="npu")
    starts = torch.tensor([0, 1, 2, 3, 3, 4], dtype=torch.int32, device="npu")
    packed = torch.empty(5, width, dim, device="npu")
    packed_indices = torch.empty_like(indices)
    args = (
        cache,
        packed,
        indices,
        starts,
        packed_indices,
        cache.stride(0),
        1,
        slots,
        5,
        width,
        dim,
        *cache.stride()[1:],
    )
    grid = (5 * width * triton.cdiv(dim, 256),)
    _copy_conv_state[grid](*args, WRITE_BACK=False, BLOCK=256)
    torch.testing.assert_close(packed_indices.cpu(), torch.tensor([0, -1, -1, -1, 4], dtype=torch.int32))
    torch.testing.assert_close(packed[0], cache[2])
    torch.testing.assert_close(packed[4], cache[3])
    assert torch.count_nonzero(packed[1:4]).item() == 0
    packed.fill_(17)
    _copy_conv_state[grid](*args, WRITE_BACK=True, BLOCK=256)
    expected = saved.as_strided(cache.shape, strides)
    expected[2].fill_(17)
    expected[3].fill_(17)
    torch.testing.assert_close(backing, saved)


def _reference_update(x, weight, state, indices, lengths, accepted, initial):
    """Independent CPU reference, including the sliding MTP history window."""
    output = torch.zeros_like(x)
    width = weight.shape[0]
    start = 0
    for request, (slot, length) in enumerate(zip(indices, lengths)):
        if slot >= 0 and length > 0:
            offset = 0 if accepted is None else accepted[request] - 1
            history = state[slot, offset : offset + width - 1].clone()
            if initial is not None and not initial[request]:
                history.zero_()
            tokens = x[start : start + length]
            sequence = torch.cat((history, tokens))
            convolved = F.conv1d(sequence.T.unsqueeze(0).float(), weight.T.unsqueeze(1).float(), groups=x.shape[1])
            output[start : start + length] = F.silu(convolved).squeeze(0).T.to(x.dtype)
            updated = sequence[-(width - 1) :] if accepted is None else sequence[1:]
            state[slot, : updated.shape[0]].copy_(updated)
        start += length
    return output


@torch.inference_mode()
@pytest.mark.parametrize("layout", ["contiguous", "paged", "transposed"])
@pytest.mark.parametrize("mode", ["prefill", "decode", "spec"])
@pytest.mark.parametrize("dim", [384, 6144])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_graph", [False, True])
def test_conv_output_and_original_state_writeback(layout, mode, dim, dtype, use_graph):
    torch.manual_seed(0)
    slots, width, state_len = 5, 4, 6
    page_size = state_len * dim + (64 if layout != "contiguous" else 0)
    storage_offset = 32
    strides = (page_size, 1, state_len) if layout == "transposed" else (page_size, dim, 1)
    backing = torch.full((storage_offset + slots * page_size,), -7, device="npu", dtype=dtype)
    state = backing.as_strided((slots, state_len, dim), strides, storage_offset=storage_offset)
    initial_state = torch.randn(state.shape, dtype=dtype)
    state.copy_(initial_state)
    original = backing.cpu()
    reference = initial_state.clone()
    indices_cpu = [3, -1, 1, 2]
    lengths = {"prefill": [7, 1, 4, 0], "decode": [1, 1, 1, 0], "spec": [4, 0, 2, 1]}[mode]
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="npu", dtype=torch.int32)
    # GDN metadata can pass a block table; convolution uses its first column.
    indices = torch.tensor([[slot, -1] for slot in indices_cpu], device="npu", dtype=torch.int32)
    x_cpu = torch.randn(sum(lengths), dim, dtype=dtype)
    weight_cpu = torch.randn(width, dim, dtype=dtype)
    x, weight = x_cpu.to("npu"), weight_cpu.to("npu")
    # Include full acceptance (4) and rejection to a shorter history window.
    accepted_cpu = [4, 1, 2, 1] if mode == "spec" else None
    accepted = torch.tensor(accepted_cpu, device="npu", dtype=torch.int32) if accepted_cpu is not None else None
    initial_cpu = [True, False, False, True] if mode == "prefill" else None
    initial = torch.tensor(initial_cpu, device="npu", dtype=torch.bool) if initial_cpu is not None else None

    def run():
        return causal_conv1d(
            x,
            weight,
            state,
            starts,
            indices,
            run_mode=0 if mode == "prefill" else 1,
            initial_state_mode=initial,
            num_accepted_tokens=accepted,
        )

    if use_graph:
        for _ in range(2):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run()
        backing.copy_(original)

    for _ in range(2):
        expected = _reference_update(x_cpu, weight_cpu, reference, indices_cpu, lengths, accepted_cpu, initial_cpu)
        if use_graph:
            graph.replay()
        else:
            actual = run()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state.cpu(), reference, rtol=0, atol=0)
        expected_backing = original.clone()
        expected_backing.as_strided(state.shape, strides, storage_offset=storage_offset).copy_(reference)
        torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)


@torch.inference_mode()
def test_conv_copy_preserves_page_addresses_above_int32():
    dim, history = 384, 3
    row_stride = 2**31 + 256
    # Reserve a sparse view across the int32 offset boundary; only the two
    # small state rows are touched. Absolute page addresses must stay int64.
    backing = torch.empty(row_stride + history * dim, device="npu", dtype=torch.bfloat16)
    state = backing.as_strided((2, history, dim), (row_stride, dim, 1))
    values = torch.arange(2 * history * dim, device="npu").reshape(2, history, dim).to(torch.bfloat16)
    state.copy_(values)
    slots = torch.tensor([[1], [-1], [2], [0]], device="npu", dtype=torch.int32)
    starts = torch.tensor([0, 1, 2, 3, 3], device="npu", dtype=torch.int32)
    packed = torch.empty((4, history, dim), device="npu", dtype=torch.bfloat16)
    packed_ids = torch.empty(4, device="npu", dtype=torch.int32)
    args = (state, packed, slots, starts, packed_ids, row_stride, 1, 2, 4, history, dim, dim, 1)
    grid = (4 * history * ((dim + 255) // 256),)
    _copy_conv_state[grid](*args, WRITE_BACK=False, BLOCK=256)
    assert torch.equal(packed[0], values[1])
    assert torch.count_nonzero(packed[1:]) == 0
    assert torch.equal(packed_ids.cpu(), torch.tensor([0, -1, -1, -1], dtype=torch.int32))
    packed.fill_(17)
    _copy_conv_state[grid](*args, WRITE_BACK=True, BLOCK=256)
    assert torch.equal(state[0], values[0])
    assert (state[1] == 17).all()


@torch.inference_mode()
@pytest.mark.parametrize(
    "requests,state_len,dim",
    [
        (1, 1, 1),
        (3, 2, 127),
        (7, 3, 128),
        (17, 6, 255),
        (31, 3, 256),
        (33, 10, 257),
        (64, 2, 383),
        (65, 3, 384),
        (127, 6, 385),
        (128, 3, 3072),
        (129, 6, 6144),
        (256, 3, 12288),
        (257, 10, 24576),
    ],
)
@pytest.mark.parametrize("layout", ["paged", "transposed", "channel_slice"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_graph", [False, True])
def test_conv_copy_shape_boundaries(requests, state_len, dim, layout, dtype, use_graph):
    """Check the copy kernel independently of the native convolution's shape limits."""
    torch.manual_seed(1024)
    num_slots, storage_offset = requests + 3, 17
    step = 2 if layout == "channel_slice" else 1
    page_size = state_len * dim * step + 67
    strides = (page_size, 1, state_len) if layout == "transposed" else (page_size, dim * step, step)
    backing = torch.full((storage_offset + num_slots * page_size,), -7, device="npu", dtype=dtype)
    state = backing.as_strided((num_slots, state_len, dim), strides, storage_offset=storage_offset)
    state.copy_(torch.randn(state.shape, dtype=dtype))
    expected_backing = backing.cpu()
    reference = expected_backing.as_strided(state.shape, strides, storage_offset=storage_offset)
    slot_ids = (torch.arange(requests) + 2) % num_slots
    if requests > 1:
        slot_ids[1::7] = -1
        slot_ids[2::11] = num_slots
    indices = torch.full((requests, 3), -99, device="npu", dtype=torch.int32)
    indices[:, 1].copy_(slot_ids)
    indices = indices[:, 1:]
    lengths = ((torch.arange(requests) % 3) != 1).to(torch.int32)
    starts = torch.zeros(requests + 1, device="npu", dtype=torch.int32)
    starts[1:].copy_(lengths.cumsum(0))
    packed = torch.empty((requests, state_len, dim), device="npu", dtype=dtype)
    packed_ids = torch.empty(requests, device="npu", dtype=torch.int32)
    args = (
        state,
        packed,
        indices,
        starts,
        packed_ids,
        page_size,
        indices.stride(0),
        num_slots,
        requests,
        state_len,
        dim,
        strides[1],
        strides[2],
    )
    grid = (min(requests * state_len * ((dim + 255) // 256), CONV_STATE_COPY_MAX_PROGRAMS),)

    def copy(write_back):
        _copy_conv_state[grid](*args, WRITE_BACK=write_back, BLOCK=256)

    if use_graph:
        copy(False)
        copy(True)
        read_graph, write_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
        with torch.npu.graph(read_graph):
            copy(False)
        with torch.npu.graph(write_graph):
            copy(True)

    for empty in (False, True):
        if empty:
            starts.zero_()
        active = (slot_ids >= 0) & (slot_ids < num_slots) & (lengths > 0) & (not empty)
        expected = torch.zeros(packed.shape, dtype=dtype)
        expected[active] = reference[slot_ids[active]]
        packed.fill_(float("nan"))
        read_graph.replay() if use_graph else copy(False)
        assert torch.equal(packed.cpu(), expected)
        assert torch.equal(packed_ids.cpu(), torch.where(active, torch.arange(requests), -1).to(torch.int32))
        replacement = torch.randn(packed.shape, dtype=dtype)
        packed.copy_(replacement)
        reference[slot_ids[active]] = replacement[active]
        write_graph.replay() if use_graph else copy(True)
        assert torch.equal(backing.cpu(), expected_backing)
