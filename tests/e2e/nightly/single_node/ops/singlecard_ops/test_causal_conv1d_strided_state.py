# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check persistent convolution state, including the GDN payload page gap."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

_DIM = 6144
_STATE_LENGTH = 3
_NUM_PAGES = 6
_SELECTED_PAGES = (1, _NUM_PAGES - 1)
_STORAGE_OFFSET = 32
_CANARY = 7.0
_OUTPUT_ATOL = 1e-4
_OUTPUT_RTOL = 0.02


@pytest.mark.parametrize(
    "page_stride",
    [_STATE_LENGTH * _DIM, 542720, 655360],
    ids=["dense", "typed-page", "uniform-page"],
)
def test_causal_conv1d_writes_back_strided_state(page_stride):
    generator = torch.Generator().manual_seed(20260908)
    prefill = (torch.randn(6, _DIM, generator=generator) * 0.1).to(torch.bfloat16)
    decode = (torch.randn(2, _DIM, generator=generator) * 0.1).to(torch.bfloat16)
    weight = (torch.randn(4, _DIM, generator=generator) * 0.1).to(torch.bfloat16)
    backing = torch.full(
        (_NUM_PAGES * page_stride + 2 * _STORAGE_OFFSET,),
        _CANARY,
        dtype=torch.bfloat16,
        device="npu",
    )
    state = torch.as_strided(
        backing,
        (_NUM_PAGES, _STATE_LENGTH, _DIM),
        (page_stride, _DIM, 1),
        _STORAGE_OFFSET,
    )
    state.zero_()
    expected_backing = backing.cpu()
    expected_state = torch.as_strided(expected_backing, state.shape, state.stride(), _STORAGE_OFFSET)
    cache_indices = torch.tensor(_SELECTED_PAGES, dtype=torch.int32, device="npu")
    initial_state = torch.zeros(len(_SELECTED_PAGES), dtype=torch.bool, device="npu")

    for run_mode, inputs, seq_length in [(0, prefill, 3), (1, decode, 1)]:
        expected_outputs = []
        for batch_index, page in enumerate(_SELECTED_PAGES):
            history = expected_state[page].clone()
            tokens = inputs[batch_index * seq_length : (batch_index + 1) * seq_length]
            for token in tokens:
                window = torch.cat((history, token.unsqueeze(0)))
                expected_outputs.append(F.silu((window.float() * weight.float()).sum(0)))
                history = window[1:]
            expected_state[page].copy_(history)
        output = torch.empty_like(inputs, device="npu")
        starts = torch.arange(0, (len(_SELECTED_PAGES) + 1) * seq_length, seq_length, dtype=torch.int32, device="npu")
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            output,
            inputs.npu(),
            weight.npu(),
            conv_state=state,
            bias_opt=None,
            query_start_loc_opt=starts,
            cache_indices_opt=cache_indices,
            initial_state_mode_opt=initial_state if run_mode == 0 else None,
            num_accepted_tokens_opt=None,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=run_mode,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(
            output.cpu(),
            torch.stack(expected_outputs).to(torch.bfloat16),
            rtol=_OUTPUT_RTOL,
            atol=_OUTPUT_ATOL,
        )
        # Exact comparison includes selected/untouched pages, adjacent recurrent
        # payload space, the nonzero storage offset, and both outer guards.
        torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)
