# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_


@pytest.mark.parametrize("padding", [0, 13])
def test_gather_and_scatter_ssm_states(padding: int) -> None:
    num_states = 5
    state_shape = (2, 4, 8)
    row_size = 64
    state_stride = row_size + padding
    storage_size = (num_states - 1) * state_stride + row_size

    storage = torch.full(
        (storage_size,),
        -1.0,
        dtype=torch.float32,
        device="npu",
    )
    state = torch.as_strided(
        storage,
        size=(num_states, *state_shape),
        stride=(state_stride, 32, 8, 1),
    )
    for state_idx in range(num_states):
        state[state_idx].fill_(state_idx + 0.25)

    indices = torch.tensor([3, 0, 2], dtype=torch.int32, device="npu")
    has_initial_state = torch.tensor([True, False, True], device="npu")

    gathered = gather_ssm_states(
        state,
        indices,
        has_initial_state,
        output_dtype=torch.bfloat16,
    )
    expected_gather = torch.stack(
        [state[3], torch.zeros_like(state[0]), state[2]]
    ).to(torch.bfloat16)
    torch.testing.assert_close(gathered, expected_gather, rtol=0, atol=0)

    source = torch.arange(
        indices.numel() * row_size,
        dtype=torch.float32,
        device="npu",
    ).reshape(indices.numel(), *state_shape).to(torch.bfloat16)
    scatter_ssm_states_(state, indices, source)

    for source_idx, state_idx in enumerate((3, 0, 2)):
        torch.testing.assert_close(
            state[state_idx],
            source[source_idx].to(state.dtype),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        state[1],
        torch.full_like(state[1], 1.25),
        rtol=0,
        atol=0,
    )

    if padding:
        storage_cpu = storage.cpu()
        for state_idx in range(num_states - 1):
            padding_start = state_idx * state_stride + row_size
            padding_end = (state_idx + 1) * state_stride
            torch.testing.assert_close(
                storage_cpu[padding_start:padding_end],
                torch.full((padding,), -1.0),
                rtol=0,
                atol=0,
            )
