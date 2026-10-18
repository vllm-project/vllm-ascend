# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the production A5 KDA adapter against a CPU token recurrence."""

from itertools import accumulate

import pytest
import torch
import torch.nn.functional as functional
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.kda import KDA_CHUNK_SIZE, run_chunk_kda

from .test_kimi_kda_ascendc_npu import _l2norm, _naive_kda

_HEADS = 6
_HEAD_DIM = 128


def _reference(inputs, cu_seqlens, lower_bound):
    q, k = (_l2norm(inputs[name]) for name in ("q", "k"))
    gate_input = inputs["raw_gate"].float() + inputs["dt_bias"].view(1, 1, _HEADS, _HEAD_DIM)
    exp_a = inputs["a_log"].exp().view(1, 1, _HEADS, 1)
    gate = (
        lower_bound * torch.sigmoid(exp_a * gate_input)
        if lower_bound is not None
        else -exp_a * functional.softplus(gate_input)
    )
    output = torch.zeros_like(inputs["v"])
    final_state = torch.empty_like(inputs["initial_state"])
    for sequence, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        output[:, start:end], state_kv = _naive_kda(
            q[:, start:end],
            k[:, start:end],
            inputs["v"][:, start:end],
            gate[:, start:end],
            inputs["beta"][:, start:end],
            inputs["initial_state"][sequence : sequence + 1].transpose(-1, -2).contiguous(),
        )
        final_state[sequence : sequence + 1] = state_kv.transpose(-1, -2)
    return output, final_state


@pytest.mark.parametrize("lengths", [(129,), (513,), (127, 129, 65)], ids=["129", "513", "mixed"])
@pytest.mark.parametrize("lower_bound", [-5.0, None], ids=["safe_gate", "softplus_gate"])
@pytest.mark.parametrize("has_initial_state", [False, True], ids=["zero_state", "nonzero_state"])
@torch.inference_mode()
def test_a5_chunk_kda_padding_matches_reference_and_is_deterministic(lengths, lower_bound, has_initial_state):
    if not is_950():
        pytest.skip("requires Ascend A5")

    cu_seqlens = (0, *accumulate(lengths))
    # The host descriptors exclude the model runner's physical token padding.
    physical_padding = 7 if lengths == (129,) else 0
    shape = (1, sum(lengths) + physical_padding, _HEADS, _HEAD_DIM)
    generator = torch.Generator().manual_seed(20261007 + sum(lengths))
    inputs = {
        "q": torch.randn(shape, generator=generator).to(torch.bfloat16),
        "k": torch.randn(shape, generator=generator).to(torch.bfloat16),
        "v": (torch.randn(shape, generator=generator) * 0.2).to(torch.bfloat16),
        "raw_gate": (torch.randn(shape, generator=generator) * 2.0).to(torch.bfloat16),
        "beta": torch.randn(shape[:-1], generator=generator).sigmoid(),
        "a_log": torch.empty(_HEADS).uniform_(-0.5, 0.8, generator=generator),
        "dt_bias": torch.empty(_HEADS * _HEAD_DIM).uniform_(-7.5, -1.5, generator=generator),
        "initial_state": torch.zeros(len(lengths), _HEADS, _HEAD_DIM, _HEAD_DIM),
    }
    if has_initial_state:
        inputs["initial_state"].normal_(std=0.02, generator=generator)
    chunk_indices = tuple(
        value
        for sequence, length in enumerate(lengths)
        for chunk in range((length + KDA_CHUNK_SIZE - 1) // KDA_CHUNK_SIZE)
        for value in (sequence, chunk)
    )
    expected = _reference(inputs, cu_seqlens, lower_bound)
    # Preallocate both independent input sets before either launch, then verify
    # the adapter did not mutate any inputs, including the caller's VK cache.
    runs = [{name: value.to("npu") for name, value in inputs.items()} for _ in range(2)]
    snapshots = []
    for npu_inputs in runs:
        result = run_chunk_kda(
            **npu_inputs,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
        torch.npu.synchronize()
        snapshots.append(tuple(value.cpu().contiguous() for value in result))
        for name, value in npu_inputs.items():
            torch.testing.assert_close(value.cpu(), inputs[name], rtol=0, atol=0, msg=f"mutated {name}")

    for index, name in enumerate(("output", "final_state")):
        first, second = (snapshot[index] for snapshot in snapshots)
        assert first.shape == expected[index].shape, name
        assert first.dtype == expected[index].dtype, name
        assert torch.isfinite(first).all() and torch.isfinite(second).all(), f"nonfinite {name}"
        assert torch.equal(first.view(torch.uint8), second.view(torch.uint8)), f"nondeterministic {name}"
        # BF16 chunk intermediates differ from the FP32 token recurrence.
        # Keep an absolute allowance for values near zero without masking the
        # tail corruption that the exact repeat comparison guards above.
        torch.testing.assert_close(
            first,
            expected[index],
            rtol=3e-2,
            atol=5e-4 if index == 0 else 3e-3,
            msg=name,
        )
    if physical_padding:
        assert torch.count_nonzero(snapshots[0][0][:, cu_seqlens[-1] :]) == 0
