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
    # Persistent slot two becomes packed slot one; packed slot zero is null.
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


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("requests", [1, 2])
def test_packed_decode_padding_preserves_output_and_shared_pages(dim_first, use_graph, requests):
    # Match the full GLM-5.3-Flash convolution width and shared-cache page stride.
    # The caller retains only the output; staging allocations remain in the wrapper.
    slots, state_len, dim, page_stride = 4, 3, 24576, 2293760
    dtype = torch.bfloat16
    strides = (page_stride, 1, state_len) if dim_first else (page_stride, dim, 1)
    x_cpu = ((torch.arange(dim, dtype=torch.float32) % 17) / 64).to(dtype).repeat(requests, 1)
    history = torch.stack([x_cpu[0] + offset for offset in (0.125, 0.25, 0.5)])
    storage = torch.full((slots * page_stride,), -3.0, dtype=dtype)
    storage.as_strided((slots, state_len, dim), strides)[1 : requests + 1].copy_(history)
    backing = storage.npu()
    state = backing.as_strided((slots, state_len, dim), strides)
    x = x_cpu.npu()
    weight = torch.full((state_len + 1, dim), 0.25, dtype=dtype, device="npu")
    indices = torch.arange(1, requests + 1, dtype=torch.int32, device="npu")
    starts = torch.arange(requests + 1, dtype=torch.int32, device="npu")

    def run():
        return causal_conv1d(x, weight, state, starts, indices, run_mode=1)

    result = run()
    torch.npu.synchronize()
    if use_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = run()
        torch.npu.synchronize()

    # Reuse the same graph across active, idle and invalid requests, then active
    # again. FLA decode ignores a zero query length, so the null slot must mask it.
    for slot, tokens in [(1, 1), (1, 0), (-1, 1), (slots, 1), (1, 1)]:
        backing.copy_(storage)
        x.copy_(x_cpu)
        indices.copy_(torch.tensor([slot, *range(2, requests + 1)], dtype=torch.int32))
        starts.copy_(torch.tensor([0, *range(tokens, tokens + requests)], dtype=torch.int32))
        if use_graph:
            graph.replay()
        else:
            result = run()
        torch.npu.synchronize()
        expected_storage = storage.clone()
        expected_output = torch.zeros_like(x_cpu)
        for request in range(requests):
            if request > 0 or (slot == 1 and tokens):
                current_x = x_cpu[request : request + 1]
                expected_storage.as_strided(state.shape, strides)[request + 1].copy_(
                    torch.cat((history[1:], current_x))
                )
                expected_output[request].copy_(F.silu((history.float().sum(dim=0) + current_x[0].float()) * 0.25))
        torch.testing.assert_close(result.cpu(), expected_output, atol=2e-2, rtol=2e-2)
        if slot != 1 or not tokens:
            torch.testing.assert_close(result[0].cpu(), torch.zeros_like(x_cpu[0]), atol=0, rtol=0)
        torch.testing.assert_close(backing.cpu(), expected_storage, atol=0, rtol=0)
        torch.testing.assert_close(x.cpu(), x_cpu, atol=0, rtol=0)
