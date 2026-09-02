# Adapt from https://github.com/vllm-project/vllm/blob/main/tests/v1/spec_decode/test_rejection_sampler_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

VOCAB_SIZE = 4096

pytest.importorskip("triton")
if not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("NPU required for MRV2 rejection sampler tests", allow_module_level=True)


def _build_rejection_sample_inputs(
    target_logits_1d: torch.Tensor,
    draft_logits_1d: torch.Tensor,
    num_speculative_steps: int,
    temperature: float,
    num_trials: int,
) -> dict:
    """Build rejection_sample kwargs from a fixed target and draft distribution.

    target_logits_1d must already have temperature applied (the sampler applies
    sampling params before verification), whereas draft_logits_1d must not:
    rejection_sample divides the draft logits by the temperature on load.
    """
    device = target_logits_1d.device
    vocab_size = target_logits_1d.shape[0]
    K = num_speculative_steps
    num_logits = num_trials * (K + 1)

    target_logits = target_logits_1d.unsqueeze(0).expand(num_logits, -1).contiguous()
    draft_logits = draft_logits_1d.view(1, 1, vocab_size).expand(num_trials, K, -1).contiguous()

    scaled_draft_logits_1d = draft_logits_1d.float()
    if temperature > 0:
        scaled_draft_logits_1d = scaled_draft_logits_1d / temperature
    draft_probs = torch.softmax(scaled_draft_logits_1d, dim=0)
    # Sample on CPU: torch.multinomial is not reliably available on NPU.
    draft_tokens = torch.multinomial(draft_probs.cpu().expand(num_trials, -1), K, replacement=True).to(device)
    draft_sampled_2d = torch.zeros(num_trials, K + 1, dtype=torch.int64, device=device)
    draft_sampled_2d[:, 1:] = draft_tokens
    draft_sampled = draft_sampled_2d.reshape(-1)

    cu_num_logits = torch.arange(num_trials + 1, dtype=torch.int32, device=device) * (K + 1)
    pos = torch.arange(num_logits, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(num_trials, dtype=torch.int32, device=device)
    expanded_idx_mapping = torch.arange(num_trials, dtype=torch.int32, device=device).repeat_interleave(K + 1)
    expanded_local_pos = torch.arange(K + 1, dtype=torch.int32, device=device).repeat(num_trials)
    temp_tensor = torch.full((num_trials,), temperature, dtype=torch.float32, device=device)
    seed = torch.arange(num_trials, dtype=torch.int64, device=device)

    return dict(
        target_logits=target_logits,
        draft_logits=draft_logits,
        draft_sampled=draft_sampled,
        cu_num_logits=cu_num_logits,
        pos=pos,
        idx_mapping=idx_mapping,
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=expanded_local_pos,
        temperature=temp_tensor,
        seed=seed,
    )


@pytest.mark.parametrize(
    "num_speculative_steps,temperature,unconditional_rates",
    [
        (3, 1.0, [0.9, 0.5, 0.2]),
        (3, 0.0, [0.9, 0.5, 0.2]),
        (3, 1.0, [1.0, 1.0, 1.0]),
        (3, 0.0, [1.0, 1.0, 1.0]),
        (3, 1.0, [0.0, 0.0, 0.0]),
        (3, 0.0, [0.0, 0.0, 0.0]),
        (1, 1.0, [0.7]),
        (1, 0.0, [0.7]),
    ],
)
@torch.inference_mode()
def test_synthetic_rejection_sample(
    num_speculative_steps: int,
    temperature: float,
    unconditional_rates: list[float],
):
    """
    Verify that synthetic rejection sampling produces the expected
    per-position acceptance rates. The unconditional rate at position i
    is P(all draft steps 0..i accepted) = product(conditional_rates[0:i+1]).
    This is approximately mean(num accepted >= i + 1) over many trials.
    """
    torch.manual_seed(42)
    device = "npu"
    num_trials = 10 * VOCAB_SIZE
    deviation_tol = 1e-2

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)

    if temperature > 0:
        target_logits_1d /= temperature

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=temperature,
        num_trials=num_trials,
    )

    conditional_rates = unconditional_to_conditional_rates(unconditional_rates)
    synthetic_conditional_rates = torch.tensor(conditional_rates, dtype=torch.float32, device=device)

    _, num_sampled = rejection_sample(
        **inputs,
        num_speculative_steps=num_speculative_steps,
        synthetic_conditional_rates=synthetic_conditional_rates,
    )

    # num_sampled includes the resampled/bonus token.
    num_accepted = num_sampled - 1
    for i, expected_rate in enumerate(unconditional_rates):
        observed_rate = (num_accepted >= i + 1).float().mean().item()
        assert abs(observed_rate - expected_rate) < deviation_tol, (
            f"Step {i}: observed rate {observed_rate:.4f} deviates from "
            f"expected rate {expected_rate:.4f} by more than {deviation_tol}."
        )

    gc.collect()
    torch.npu.empty_cache()
