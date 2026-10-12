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
