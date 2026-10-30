# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from torch.nn import functional as F

from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d
from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state


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
    packed = torch.empty(6, width, dim, device="npu")
    packed_indices = torch.empty_like(indices)
    copy_conv_state(cache, packed, indices, starts, packed_indices, write_back=False)
    torch.testing.assert_close(packed_indices.cpu(), torch.tensor([0, 5, 5, 5, 4], dtype=torch.int32))
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
@pytest.mark.parametrize("layout", ["contiguous", "page_strided", "dim_first"])
@pytest.mark.parametrize("run_mode,tokens,initial", [(0, 127, False), (0, 128, True), (1, 1, True)])
def test_causal_conv_first_packed_row_matches_reference(dtype, layout, run_mode, tokens, initial):
    slots, state_len, dim = 4, 3, 384
    generator = torch.Generator().manual_seed(42)
    storage = (torch.randn(slots * state_len * dim * 2, generator=generator) * 0.2).to(dtype)
    if layout == "contiguous":
        strides = (state_len * dim, dim, 1)
    elif layout == "page_strided":
        strides = (state_len * dim * 2, dim, 1)
    else:
        strides = (state_len * dim * 2, 1, state_len)
    backing = storage.npu()
    state = backing.as_strided((slots, state_len, dim), strides)
    expected_storage = storage.clone()
    expected_state = expected_storage.as_strided(state.shape, strides)
    x = (torch.randn(tokens, dim, generator=generator) * 0.2).to(dtype)
    weight = (torch.randn(state_len + 1, dim, generator=generator) * 0.2).to(dtype)
    # Persistent slot two becomes packed slot zero for both strided layouts.
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "page_strided", "dim_first"])
@pytest.mark.parametrize("batch", [4, 8, 16])
@pytest.mark.parametrize("use_graph", [False, True])
def test_causal_conv_decode_padding_and_graph_replay(dtype, layout, batch, use_graph):
    """Changing live rows must not address state[-1] or mutate other pages."""
    state_len, dim = 3, 1536
    slots = batch + 3
    generator = torch.Generator().manual_seed(53)
    storage = (torch.randn(slots * state_len * dim * 2, generator=generator) * 0.2).to(dtype)
    if layout == "contiguous":
        strides = (state_len * dim, dim, 1)
    elif layout == "page_strided":
        strides = (state_len * dim * 2, dim, 1)
    else:
        strides = (state_len * dim * 2, 1, state_len)
    backing = storage.npu()
    state = backing.as_strided((slots, state_len, dim), strides)
    weight_cpu = (torch.randn(state_len + 1, dim, generator=generator) * 0.2).to(dtype)
    weight = weight_cpu.npu()
    x = torch.zeros(batch, dim, dtype=dtype, device="npu")
    indices = torch.arange(2, batch + 2, dtype=torch.int32, device="npu")
    starts = torch.arange(batch + 1, dtype=torch.int32, device="npu")

    def run():
        return causal_conv1d(x, weight, state, starts, indices, run_mode=1)

    graph = None
    if use_graph:
        # Compile Triton before capture. Replay must consume the updated
        # device metadata, rather than the all-active capture-time values.
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = run()

    # Includes 5/6/7 live rows in an 8-row graph, an entirely idle batch,
    # and restoring a full batch after padding has previously been present.
    live_counts = [batch, batch - 1, batch - 2, batch // 2 + 1, 0, batch]
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    for live in live_counts:
        backing.copy_(storage)
        input_cpu = (torch.randn(batch, dim, generator=generator) * 0.2).to(dtype)
        x.copy_(input_cpu)
        indices_cpu = torch.arange(2, batch + 2, dtype=torch.int32)
        indices_cpu[live:] = 0
        indices.copy_(indices_cpu)
        starts.copy_(torch.arange(batch + 1, dtype=torch.int32).clamp(max=live))
        expected_storage = storage.clone()
        expected_state = expected_storage.as_strided(state.shape, strides)
        expected_output = torch.zeros_like(input_cpu)
        for request in range(live):
            slot = int(indices_cpu[request])
            combined = torch.cat((expected_state[slot].float(), input_cpu[request : request + 1].float()))
            expected_output[request] = (
                F.silu(F.conv1d(combined.T.unsqueeze(0), weight_cpu.float().T.unsqueeze(1), groups=dim))
                .reshape(dim)
                .to(dtype)
            )
            expected_state[slot].copy_(combined[-state_len:].to(dtype))
        if graph is None:
            result = run()
        else:
            graph.replay()
        torch.testing.assert_close(result.cpu(), expected_output, atol=tolerance, rtol=tolerance)
        assert torch.count_nonzero(result[live:]).item() == 0
        # Includes null slot zero, other requests, and inter-page padding.
        torch.testing.assert_close(backing.cpu(), expected_storage, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim_first", [False, True])
def test_causal_conv_prefill_with_empty_request_rows(dtype, dim_first):
    """The packed null slot must also preserve prefill's initial-state rules."""
    slots, state_len, dim = 6, 3, 384
    generator = torch.Generator().manual_seed(54)
    storage = (torch.randn(slots * state_len * dim * 2, generator=generator) * 0.2).to(dtype)
    strides = (state_len * dim * 2, 1, state_len) if dim_first else (state_len * dim * 2, dim, 1)
    backing = storage.npu()
    state = backing.as_strided((slots, state_len, dim), strides)
    starts = [0, 2, 2, 3, 3, 6]
    indices = [2, 0, 3, 0, 4]
    initial = [True, False, False, False, True]
    input_cpu = (torch.randn(starts[-1], dim, generator=generator) * 0.2).to(dtype)
    weight_cpu = (torch.randn(state_len + 1, dim, generator=generator) * 0.2).to(dtype)
    expected_storage = storage.clone()
    expected_state = expected_storage.as_strided(state.shape, strides)
    expected_output = torch.zeros_like(input_cpu)
    for request, slot in enumerate(indices):
        start, end = starts[request : request + 2]
        if end == start:
            continue
        history = expected_state[slot].float() if initial[request] else torch.zeros(state_len, dim)
        combined = torch.cat((history, input_cpu[start:end].float()))
        expected_output[start:end] = (
            F.silu(F.conv1d(combined.T.unsqueeze(0), weight_cpu.float().T.unsqueeze(1), groups=dim))
            .squeeze(0)
            .T.to(dtype)
        )
        expected_state[slot].copy_(combined[-state_len:].to(dtype))
    result = causal_conv1d(
        input_cpu.npu(),
        weight_cpu.npu(),
        state,
        torch.tensor(starts, dtype=torch.int32, device="npu"),
        torch.tensor(indices, dtype=torch.int32, device="npu"),
        run_mode=0,
        initial_state_mode=torch.tensor(initial, dtype=torch.bool, device="npu"),
    )
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(result.cpu(), expected_output, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(backing.cpu(), expected_storage, atol=0, rtol=0)
