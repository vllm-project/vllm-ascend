# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from torch.nn import functional as F

from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d
from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state, prepare_conv_indices, prepare_conv_indices_pair


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
    copy_conv_state(cache, packed, indices, starts, packed_indices, write_back=False)
    torch.testing.assert_close(packed_indices.cpu(), torch.tensor([0, -1, -1, -1, 4], dtype=torch.int32))
    torch.testing.assert_close(packed[0], cache[2])
    torch.testing.assert_close(packed[4], cache[3])
    assert torch.count_nonzero(packed[1:4]).item() == 0
    packed.fill_(17)
    copy_conv_state(cache, packed, indices, starts, packed_indices, write_back=True)
    expected = saved.as_strided(cache.shape, strides)
    expected[2].fill_(17)
    expected[3].fill_(17)
    torch.testing.assert_close(backing, saved)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "page_strided", "row_strided", "dim_first"])
@pytest.mark.parametrize("run_mode,tokens,initial", [(0, 127, False), (0, 128, True), (1, 1, True)])
def test_causal_conv_first_packed_row_matches_reference(dtype, layout, run_mode, tokens, initial):
    slots, state_len, dim = 4, 3, 384
    generator = torch.Generator().manual_seed(42)
    storage = (torch.randn(slots * state_len * dim * 2, generator=generator) * 0.2).to(dtype)
    if layout == "contiguous":
        strides = (state_len * dim, dim, 1)
    elif layout == "page_strided":
        strides = (state_len * dim * 2, dim, 1)
    elif layout == "row_strided":
        strides = (state_len * dim * 2, dim * 2, 1)
    else:
        strides = (state_len * dim * 2, 1, state_len)
    backing = storage.npu()
    state = backing.as_strided((slots, state_len, dim), strides)
    expected_storage = storage.clone()
    expected_state = expected_storage.as_strided(state.shape, strides)
    x = (torch.randn(tokens, dim, generator=generator) * 0.2).to(dtype)
    weight = (torch.randn(state_len + 1, dim, generator=generator) * 0.2).to(dtype)
    # Dim-first states pack slot two into slot zero; SD views keep slot two.
    history = expected_state[2].float() if initial else torch.zeros(state_len, dim)
    combined = torch.cat((history, x.float()))
    expected_output = (
        F.silu(F.conv1d(combined.T.unsqueeze(0), weight.float().T.unsqueeze(1), groups=dim)).squeeze(0).T.to(dtype)
    )
    expected_state[2].copy_(combined[-state_len:].to(dtype))
    result = causal_conv1d(
        x.npu(),
        weight.npu(),
        state,
        torch.tensor([0, tokens], dtype=torch.int32, device="npu"),
        torch.tensor([2], dtype=torch.int32, device="npu"),
        run_mode=run_mode,
        initial_state_mode=torch.tensor([initial], dtype=torch.bool, device="npu"),
    )
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(result.cpu(), expected_output, atol=tolerance, rtol=tolerance)
    # Check the entire allocation: unrelated slots and gaps must not change.
    torch.testing.assert_close(backing.cpu(), expected_storage, atol=0, rtol=0)


@pytest.mark.parametrize("run_mode", [0, 1])
@pytest.mark.parametrize("columns", [1, 2])
def test_causal_conv_gdn_index_matrix_matches_first_column(run_mode, columns):
    slots, state_len, dim = 4, 3, 384
    generator = torch.Generator().manual_seed(44)
    storage = (torch.randn(slots * state_len * dim * 2, generator=generator) * 0.2).to(torch.bfloat16)
    backing = storage.npu()
    state = backing.as_strided((slots, state_len, dim), (state_len * dim * 2, dim, 1))
    reference_state = state.contiguous()
    x = (torch.randn(2, dim, generator=generator) * 0.2).to(torch.bfloat16).npu()
    weight = (torch.randn(4, dim, generator=generator) * 0.2).to(torch.bfloat16).npu()
    starts = torch.tensor([0, 1, 2], dtype=torch.int32, device="npu")
    indices = torch.tensor([2, 3], dtype=torch.int32, device="npu")
    matrix = torch.zeros(2, columns, dtype=torch.int32, device="npu")
    matrix[:, 0] = indices
    initial = torch.ones(2, dtype=torch.bool, device="npu")
    expected_output = causal_conv1d(
        x.clone(), weight, reference_state, starts, indices, run_mode=run_mode, initial_state_mode=initial
    )
    result = causal_conv1d(x.clone(), weight, state, starts, matrix, run_mode=run_mode, initial_state_mode=initial)
    torch.testing.assert_close(result, expected_output, atol=2e-2, rtol=2e-2)
    expected_storage = storage.clone()
    expected_storage.as_strided(state.shape, state.stride()).copy_(reference_state.cpu())
    torch.testing.assert_close(backing.cpu(), expected_storage, atol=0, rtol=0)


@pytest.mark.parametrize("layout", ["page_strided", "row_strided", "dim_first"])
@pytest.mark.parametrize("accepted", [1, 3, 4])
def test_causal_conv_mtp_state_writeback_matches_contiguous(layout, accepted):
    slots, state_len, dim = 4, 6, 384
    generator = torch.Generator().manual_seed(43)
    storage = (torch.randn(slots * state_len * dim * 2, generator=generator) * 0.2).to(torch.bfloat16)
    if layout == "page_strided":
        strides = (state_len * dim * 2, dim, 1)
    elif layout == "row_strided":
        strides = (state_len * dim * 2, dim * 2, 1)
    else:
        strides = (state_len * dim * 2, 1, state_len)
    backing = storage.npu()
    state = backing.as_strided((slots, state_len, dim), strides)
    reference_state = state.contiguous()
    x = (torch.randn(4, dim, generator=generator) * 0.2).to(torch.bfloat16).npu()
    weight = (torch.randn(4, dim, generator=generator) * 0.2).to(torch.bfloat16).npu()
    starts = torch.tensor([0, 4], dtype=torch.int32, device="npu")
    indices = torch.tensor([2], dtype=torch.int32, device="npu")
    accepted_tokens = torch.tensor([accepted], dtype=torch.int32, device="npu")
    # The update interface may reuse x for its output; each run needs fresh input.
    expected_output = causal_conv1d(
        x.clone(), weight, reference_state, starts, indices, run_mode=1, num_accepted_tokens=accepted_tokens
    )
    result = causal_conv1d(x.clone(), weight, state, starts, indices, run_mode=1, num_accepted_tokens=accepted_tokens)
    torch.testing.assert_close(result, expected_output, atol=2e-2, rtol=2e-2)
    expected_storage = storage.clone()
    expected_storage.as_strided(state.shape, strides).copy_(reference_state.cpu())
    torch.testing.assert_close(backing.cpu(), expected_storage, atol=0, rtol=0)


@pytest.mark.parametrize("run_mode", [0, 1])
@pytest.mark.parametrize("byte_backing", [False, True])
def test_causal_conv_batched_wide_state_with_storage_offset_matches_packed(run_mode, byte_backing):
    slots, state_len, dim, page_stride, offset = 8, 6, 3072, 280576, 128
    generator = torch.Generator().manual_seed(45)
    size = offset + (slots - 1) * page_stride + state_len * dim
    storage = (torch.randn(size, generator=generator) * 0.2).to(torch.bfloat16)
    # MRV1 also builds typed state views from a shared uint8 allocation.
    backing = storage.view(torch.uint8).npu().view(storage.dtype) if byte_backing else storage.npu()
    state = backing.as_strided((slots, state_len, dim), (page_stride, dim, 1), offset)
    # A DS reference exercises the original pack/restore path for this batch.
    reference_state = state.transpose(1, 2).contiguous().transpose(1, 2)
    x = (torch.randn(16, dim, generator=generator) * 0.2).to(torch.bfloat16).npu()
    weight = (torch.randn(4, dim, generator=generator) * 0.2).to(torch.bfloat16).npu()
    starts = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device="npu")
    indices = torch.tensor([[2], [4], [5], [7]], dtype=torch.int32, device="npu")
    initial = torch.tensor([False, True, False, True], dtype=torch.bool, device="npu")
    accepted = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="npu") if run_mode == 1 else None
    expected_output = causal_conv1d(
        x.clone(),
        weight,
        reference_state,
        starts,
        indices,
        run_mode=run_mode,
        initial_state_mode=initial,
        num_accepted_tokens=accepted,
    )
    result = causal_conv1d(
        x.clone(),
        weight,
        state,
        starts,
        indices,
        run_mode=run_mode,
        initial_state_mode=initial,
        num_accepted_tokens=accepted,
    )
    torch.testing.assert_close(result, expected_output, atol=0, rtol=0)
    expected_storage = storage.clone()
    expected_storage.as_strided(state.shape, state.stride(), offset).copy_(reference_state.cpu())
    torch.testing.assert_close(backing.cpu(), expected_storage, atol=0, rtol=0)


@pytest.mark.parametrize("valid_slot", [0, 1])
@pytest.mark.parametrize("invalid_slot", [-1, 4])
@pytest.mark.parametrize("run_mode,accepted", [(0, None), (1, None), (1, 1), (1, 4)])
def test_causal_conv_pad_slot_preserves_guard_page_and_matches_packed(valid_slot, invalid_slot, run_mode, accepted):
    slots, state_len, dim = 4, 6, 384
    page_stride = state_len * dim * 2
    # Keep a guard page before the view: an unmasked -1 cache index would
    # write here, inside the allocation, making the error safe to detect.
    storage = torch.full((page_stride * (slots + 1),), 0.5, dtype=torch.bfloat16)
    backing = storage.npu()
    state = backing.as_strided((slots, state_len, dim), (page_stride, dim, 1), page_stride)
    reference_state = state.transpose(1, 2).contiguous().transpose(1, 2)
    tokens = 4 if accepted is not None else 1
    x = torch.full((tokens * 2, dim), 0.25, dtype=torch.bfloat16, device="npu")
    weight = torch.full((4, dim), 0.125, dtype=torch.bfloat16, device="npu")
    starts = torch.tensor([0, tokens, tokens * 2], dtype=torch.int32, device="npu")
    indices = torch.tensor([[valid_slot, 0], [invalid_slot, 0]], dtype=torch.int32, device="npu")
    initial = torch.ones(2, dtype=torch.bool, device="npu")
    accepted_tokens = torch.full((2,), accepted, dtype=torch.int32, device="npu") if accepted is not None else None
    kwargs = dict(run_mode=run_mode, initial_state_mode=initial, num_accepted_tokens=accepted_tokens)
    expected_output = causal_conv1d(x.clone(), weight, reference_state, starts, indices, **kwargs)
    result = causal_conv1d(x.clone(), weight, state, starts, indices, **kwargs)
    # FLA leaves skipped output rows unspecified; compare active tokens only.
    torch.testing.assert_close(result[:tokens], expected_output[:tokens], atol=0, rtol=0)
    expected_storage = storage.clone()
    expected_storage.as_strided(state.shape, state.stride(), page_stride).copy_(reference_state.cpu())
    torch.testing.assert_close(backing.cpu(), expected_storage, atol=0, rtol=0)


@pytest.mark.parametrize("num_slots", [4, 8])
def test_prepared_conv_index_pair_matches_single_and_reuses_buffers(num_slots):
    first = torch.tensor([[0, 9], [-1, 9], [6, 9]], dtype=torch.int64, device="npu")
    first_starts = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="npu")
    second = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.int32, device="npu")
    second_starts = torch.tensor([0, 1, 1, 2], dtype=torch.int32, device="npu")
    first_out = torch.empty(8, dtype=torch.int32, device="npu")
    second_out = torch.empty_like(first_out)
    expected_first = prepare_conv_indices(first, first_starts, num_slots)
    expected_second = prepare_conv_indices(second, second_starts, num_slots)
    prepared_first, prepared_second = prepare_conv_indices_pair(
        first, first_starts, first_out, second, second_starts, second_out, num_slots
    )
    torch.testing.assert_close(prepared_first, expected_first, atol=0, rtol=0)
    torch.testing.assert_close(prepared_second, expected_second, atol=0, rtol=0)
    assert prepared_first.data_ptr() == first_out.data_ptr()
    assert prepared_second.data_ptr() == second_out.data_ptr()
    second[:, 0].fill_(-1)
    prepare_conv_indices_pair(first, first_starts, first_out, second, second_starts, second_out, num_slots)
    assert prepared_second.cpu().tolist() == [num_slots] * 3


@pytest.mark.parametrize("run_mode,accepted", [(0, None), (1, None), (1, 3)])
def test_prepared_conv_indices_match_per_layer_path(run_mode, accepted):
    slots, state_len, dim = 4, 6, 384
    backing = torch.full((slots * 2, state_len, dim), 0.5, dtype=torch.bfloat16, device="npu")
    state = backing[::2]
    reference = state.clone()
    tokens = 4 if accepted is not None else 1
    starts = torch.tensor([0, tokens, tokens * 2], dtype=torch.int32, device="npu")
    raw = torch.tensor([[0, 1], [-1, 0]], dtype=torch.int32, device="npu")
    prepared = prepare_conv_indices(raw, starts, slots, out=torch.empty(8, dtype=torch.int32, device="npu"))
    x = torch.full((tokens * 2, dim), 0.25, dtype=torch.bfloat16, device="npu")
    weight = torch.full((4, dim), 0.125, dtype=torch.bfloat16, device="npu")
    kwargs = dict(
        run_mode=run_mode,
        initial_state_mode=torch.ones(2, dtype=torch.bool, device="npu"),
        num_accepted_tokens=torch.full((2,), accepted, dtype=torch.int32, device="npu") if accepted else None,
    )
    expected = causal_conv1d(x.clone(), weight, reference, starts, raw, **kwargs)
    result = causal_conv1d(
        x.clone(),
        weight,
        state,
        starts,
        raw,
        prepared_cache_indices=prepared,
        prepared_null_block_id=slots,
        **kwargs,
    )
    torch.testing.assert_close(result[:tokens], expected[:tokens], atol=0, rtol=0)
    torch.testing.assert_close(state, reference, atol=0, rtol=0)
    assert torch.all(backing[1::2] == 0.5)


def test_prepared_conv_graph_replay_refreshes_indices_shared_by_layers():
    slots, state_len, dim = 8, 6, 384
    backings = [torch.full((slots * 2, state_len, dim), 0.5, dtype=torch.bfloat16, device="npu") for _ in range(2)]
    states = [backing[::2] for backing in backings]
    references = [state.clone() for state in states]
    weight = torch.full((4, dim), 0.125, dtype=torch.bfloat16, device="npu")
    starts = [torch.tensor(values, dtype=torch.int32, device="npu") for values in ([0, 1, 2], [0, 4, 8])]
    raw = [torch.tensor(values, dtype=torch.int32, device="npu") for values in ([[0, 1], [1, 2]], [[2, 3], [3, 4]])]
    buffers = [torch.empty(2, dtype=torch.int32, device="npu") for _ in range(2)]
    prepared = prepare_conv_indices_pair(raw[0], starts[0], buffers[0], raw[1], starts[1], buffers[1], slots)
    accepted = torch.tensor([1, 3], dtype=torch.int32, device="npu")
    inputs = [
        [torch.full((tokens, dim), 0.25, dtype=torch.bfloat16, device="npu") for tokens in (2, 8)] for _ in states
    ]

    def run_layers():
        return [
            [
                causal_conv1d(
                    inputs[layer][branch],
                    weight,
                    state,
                    starts[branch],
                    raw[branch],
                    run_mode=1,
                    num_accepted_tokens=accepted if branch else None,
                    prepared_cache_indices=prepared[branch],
                    prepared_null_block_id=slots,
                )
                for branch in range(2)
            ]
            for layer, state in enumerate(states)
        ]

    run_layers()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = run_layers()
    for backing in backings:
        backing.fill_(0.5)

    # Change physical slots, valid slot zero, invalid slots and zero-length
    # suffixes without recapturing. Both layers must consume the refreshed view.
    for non_spec, spec, non_spec_end, spec_end in (
        ([0, 1], [2, 3], 2, 8),
        ([4, 5], [6, 7], 2, 8),
        ([0, -1], [2, slots], 2, 8),
        ([1, 5], [3, 7], 1, 4),
    ):
        raw[0][:, 0].copy_(torch.tensor(non_spec, dtype=torch.int32, device="npu"))
        raw[1][:, 0].copy_(torch.tensor(spec, dtype=torch.int32, device="npu"))
        starts[0][-1] = non_spec_end
        starts[1][-1] = spec_end
        prepare_conv_indices_pair(raw[0], starts[0], buffers[0], raw[1], starts[1], buffers[1], slots)
        expected = []
        for layer, state in enumerate(references):
            row = []
            for branch in range(2):
                inputs[layer][branch].fill_(0.25)
                row.append(
                    causal_conv1d(
                        inputs[layer][branch].clone(),
                        weight,
                        state,
                        starts[branch],
                        raw[branch],
                        run_mode=1,
                        num_accepted_tokens=accepted if branch else None,
                    )
                )
            expected.append(row)
        graph.replay()
        for layer, state in enumerate(states):
            torch.testing.assert_close(state, references[layer], atol=0, rtol=0)
            torch.testing.assert_close(outputs[layer][0][:1], expected[layer][0][:1], atol=0, rtol=0)
            torch.testing.assert_close(outputs[layer][1][:4], expected[layer][1][:4], atol=0, rtol=0)
            assert torch.all(backings[layer][1::2] == 0.5)
