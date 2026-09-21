# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

"""NPU coverage requiring this branch's custom OPP and CANN decode extension.

Run on the target NPU/CANN installation; missing kernels must fail this test.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    offsets: list[int],
    cache_indices: list[int],
    initial_state: list[bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU depthwise convolution with vLLM's leading W-1 history rows."""
    history_len = weight.shape[0] - 1
    channels = x.shape[-1]
    output = torch.zeros_like(x)
    updated_state = state.clone()
    conv_weight = weight.float().transpose(0, 1).unsqueeze(1)
    for request, (start, end) in enumerate(zip(offsets, offsets[1:])):
        slot = max(cache_indices[request], 0)
        if start == end or slot == 0:
            continue
        history = updated_state[slot, :history_len]
        if not initial_state[request]:
            history = torch.zeros_like(history)
        sequence = torch.cat((history, x[start:end]), dim=0)
        convolved = F.conv1d(
            sequence.float().transpose(0, 1).unsqueeze(0),
            conv_weight,
            groups=channels,
        )
        output[start:end] = F.silu(convolved).squeeze(0).transpose(0, 1).to(x.dtype)
        updated_state[slot, :history_len] = sequence[-history_len:]
    return output, updated_state


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("state_len", [3, 5])
@torch.inference_mode()
def test_kimi_prefill_then_decode(dtype: torch.dtype, state_len: int):
    bootstrap_custom_op_env()
    from vllm_ascend.ops.kimi_kda import AscendKimiGatedDeltaNetAttention

    generator = torch.Generator().manual_seed(20260921)
    width, channels, slots = 4, 384, 8
    lengths = [1, 2, 7, 0, 3, 2, 5]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cache_indices = [3, 1, 5, 4, 0, -1, 2]
    initial_state = [False, True, True, True, False, True, False]
    x = (torch.randn(offsets[-1], channels, generator=generator) * 0.25).to(dtype)
    weight = (torch.randn(width, channels, generator=generator) * 0.25).to(dtype)
    state = (torch.randn(slots, state_len, channels, generator=generator) * 0.25).to(dtype)
    expected_output, expected_state = _reference(x, weight, state, offsets, cache_indices, initial_state)

    weight_npu = weight.to("npu")
    state_npu = state.to("npu")
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor(offsets, dtype=torch.int32, device="npu"),
        # Kimi can receive a two-dimensional block table and uses its first column.
        cache_indices=torch.tensor(cache_indices, dtype=torch.int64, device="npu").unsqueeze(1),
        initial_state_mode=torch.tensor(initial_state, dtype=torch.bool, device="npu"),
    )
    output = AscendKimiGatedDeltaNetAttention._run_causal_conv1d(
        x.to("npu"), weight_npu, state_npu, metadata, run_mode=0
    )
    rtol, atol = (1e-2, 2e-3) if dtype == torch.bfloat16 else (3e-3, 3e-4)
    torch.testing.assert_close(output.cpu(), expected_output, rtol=rtol, atol=atol)
    # Exact copying must preserve null/unused slots and speculative-only tail rows.
    torch.testing.assert_close(state_npu.cpu(), expected_state, rtol=0, atol=0)

    decode_indices = [5, 3, 1, 2]
    decode_offsets = list(range(len(decode_indices) + 1))
    decode_x = (torch.randn(len(decode_indices), channels, generator=generator) * 0.25).to(dtype)
    expected_decode, expected_final_state = _reference(
        decode_x,
        weight,
        expected_state,
        decode_offsets,
        decode_indices,
        [True] * len(decode_indices),
    )
    decode_metadata = SimpleNamespace(
        query_start_loc=torch.tensor(decode_offsets, dtype=torch.int32, device="npu"),
        cache_indices=torch.tensor(decode_indices, dtype=torch.int32, device="npu"),
    )
    decode_output = AscendKimiGatedDeltaNetAttention._run_causal_conv1d(
        decode_x.to("npu"), weight_npu, state_npu, decode_metadata, run_mode=1
    )
    torch.testing.assert_close(decode_output.cpu(), expected_decode, rtol=rtol, atol=atol)
    torch.testing.assert_close(state_npu.cpu(), expected_final_state, rtol=0, atol=0)
