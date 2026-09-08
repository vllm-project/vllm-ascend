# Adapt from https://github.com/vllm-project/vllm/blob/main/tests/v1/spec_decode/test_rejection_sampler_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import math

import pytest
import torch
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
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


def _assert_distribution_match(
    sampled_tokens: torch.Tensor,
    target_probs: torch.Tensor,
    device: str,
    label: str = "",
    min_expected: float = 5.0,
):
    """
    Assert sampled tokens match the target distribution via a
    chi-squared goodness-of-fit test. This is done by computing
    observed vs expected token counts (target_probs * num_samples),
    then checking that the chi-squared statistic is below a conservative
    threshold. The threshold is set at df + 10*sqrt(2*df), which
    corresponds to ~10 sigma under the chi-squared distribution's
    normal approximation, effectively disallowing false positives.

    NOTE: Tokens with expected count < min_expected are merged into
    a single "other" bin to minimize chi-squared noise.
    """
    num_samples = sampled_tokens.shape[0]
    vocab_size = target_probs.shape[0]

    observed = torch.zeros(vocab_size, device=device, dtype=torch.float32)
    observed.scatter_add_(0, sampled_tokens, torch.ones(num_samples, device=device))
    expected = target_probs * num_samples

    sufficient = expected >= min_expected
    obs_main = observed[sufficient]
    exp_main = expected[sufficient]

    obs_other = observed[~sufficient].sum().unsqueeze(0)
    exp_other = expected[~sufficient].sum().unsqueeze(0)

    if exp_other.item() >= min_expected:
        obs_all = torch.cat([obs_main, obs_other])
        exp_all = torch.cat([exp_main, exp_other])
    else:
        obs_all = obs_main
        exp_all = exp_main

    chi2 = ((obs_all - exp_all) ** 2 / exp_all).sum().item()
    df = obs_all.shape[0] - 1
    if df < 1:
        # All samples were merged into < 2 bins, which is too
        # few to evaluate.
        return

    threshold = df + 10 * math.sqrt(2 * df)
    prefix = f"[{label}] " if label else ""
    assert chi2 < threshold, (
        f"{prefix}Chi-squared test failed: chi2={chi2:.1f}, "
        f"df={df}, threshold={threshold:.1f}. "
        f"Output distribution does not match target distribution."
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
    # NPU: triton-ascend caps the flattened grid at 65535. The block-stats
    # kernel launches one program per logit row, and the resample kernel
    # launches num_reqs * cdiv(vocab, 1024) programs, so a single call cannot
    # hold upstream's 10 * VOCAB_SIZE trials. Split the trials into chunks
    # below every grid bound and aggregate the acceptance statistics —
    # rejection_sample is a pure function, so this is equivalent to one
    # large call (upstream uses 10 * VOCAB_SIZE trials in one shot).
    num_trials = 10 * VOCAB_SIZE
    TRIALS_PER_CALL = 16000
    deviation_tol = 1e-2

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)

    if temperature > 0:
        target_logits_1d /= temperature

    conditional_rates = unconditional_to_conditional_rates(unconditional_rates)
    synthetic_conditional_rates = torch.tensor(conditional_rates, dtype=torch.float32, device=device)

    num_accepted_chunks = []
    for start in range(0, num_trials, TRIALS_PER_CALL):
        chunk_trials = min(TRIALS_PER_CALL, num_trials - start)
        inputs = _build_rejection_sample_inputs(
            target_logits_1d,
            draft_logits_1d,
            num_speculative_steps,
            temperature=temperature,
            num_trials=chunk_trials,
        )
        # Synthetic acceptance is driven by u = f(seed, pos) only, so chunks
        # that restart the seed/pos sequences would replay identical draws.
        # Offset both so every chunk consumes a fresh noise stream.
        inputs["seed"] = inputs["seed"] + start
        inputs["pos"] = inputs["pos"] + start * (num_speculative_steps + 1)

        _, num_sampled = rejection_sample(
            **inputs,
            num_speculative_steps=num_speculative_steps,
            synthetic_conditional_rates=synthetic_conditional_rates,
        )
        # num_sampled includes the resampled/bonus token.
        num_accepted_chunks.append(num_sampled - 1)
        gc.collect()
        torch.npu.empty_cache()

    num_accepted = torch.cat(num_accepted_chunks)
    for i, expected_rate in enumerate(unconditional_rates):
        observed_rate = (num_accepted >= i + 1).float().mean().item()
        assert abs(observed_rate - expected_rate) < deviation_tol, (
            f"Step {i}: observed rate {observed_rate:.4f} deviates from "
            f"expected rate {expected_rate:.4f} by more than {deviation_tol}."
        )

    gc.collect()
    torch.npu.empty_cache()


# The wide-vocab stochastic tests above spread their samples too thin to
# resolve a small distributional bias, so this test runs narrow: 16 bins over
# 200K trials is ~12K per bin, which resolves a few percent.
NARROW_VOCAB_SIZE = 16
NARROW_NUM_TRIALS = 200_000


def _gumbel_drafted_tokens(
    inputs: dict,
    draft_logits_1d: torch.Tensor,
    num_trials: int,
    num_speculative_steps: int,
) -> torch.Tensor:
    """Proposals drawn with gumbel_sample, shaped like inputs["draft_sampled"].

    _build_rejection_sample_inputs draws them with torch.multinomial, which is
    independent of the resample noise by construction. Production drafts come
    from gumbel_sample keyed by pos[t * (K + 1) + i] for step i of trial t --
    the same entry _rejection_kernel and _resample_kernel read for that token --
    so the draft and the residual compete for one noise stream.

    NPU: pos is int32 (triton-ascend philox), matching the pos tensor built by
    _build_rejection_sample_inputs.
    """
    k = num_speculative_steps
    vocab_size = draft_logits_1d.shape[0]
    device = draft_logits_1d.device
    draft_tokens = gumbel_sample(
        draft_logits_1d.unsqueeze(0).expand(num_trials * k, vocab_size).float(),
        inputs["expanded_idx_mapping"].view(num_trials, k + 1)[:, :k].reshape(-1).contiguous(),
        inputs["temperature"],
        inputs["seed"],
        inputs["pos"].view(num_trials, k + 1)[:, :k].reshape(-1).contiguous(),
        apply_temperature=True,
        is_drafting=True,
    )
    draft_sampled = torch.zeros(num_trials * (k + 1), dtype=torch.int64, device=device)
    draft_sampled.view(num_trials, k + 1)[:, 1:] = draft_tokens.view(num_trials, k)
    return draft_sampled


@pytest.mark.parametrize("num_speculative_steps", [1, 3])
@torch.inference_mode()
def test_gumbel_drafted_rejection_sample_is_unbiased(num_speculative_steps: int):
    """The proposal and the residual resample must not share a noise vector.

    Draws proposals on the same (seed, pos) stream the sampler verifies and
    resamples with, then checks the output still follows the target. Conditioned
    on a proposal winning the argmax, every other token's Gumbel is truncated
    below that max -- most tightly for the tokens the draft ranked highest --
    so a shared stream makes the residual under-weight exactly those tokens.

    Runs narrow because the wide-vocab test above cannot resolve this: dropping
    `is_drafting=True` in _gumbel_drafted_tokens takes position 0 from chi2 ~12
    to ~1500 against a threshold of ~70 here, while leaving that test passing.
    """
    torch.manual_seed(42)
    device = "npu"

    # A draft that ranks tokens in exactly the opposite order rejects ~73% of
    # proposals, so most trials reach the residual resample where the bias
    # lives. The disagreement has to be constructed rather than sampled: two
    # independent randn draws land close together often enough that the signal
    # swings between chi2 ~14 and ~1900 depending on the seed. At temperature
    # 1.0 the target needs no scaling before being passed in.
    target_logits_1d = torch.randn(NARROW_VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = -target_logits_1d

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=1.0,
        num_trials=NARROW_NUM_TRIALS,
    )
    inputs["draft_sampled"] = _gumbel_drafted_tokens(inputs, draft_logits_1d, NARROW_NUM_TRIALS, num_speculative_steps)

    sampled, num_sampled = rejection_sample(**inputs, num_speculative_steps=num_speculative_steps)

    # Position 0 carries the power: every trial reaches it, while later
    # positions are only reached on acceptance, which is rare by construction.
    assert (num_sampled >= 1).all()
    target_probs = torch.softmax(target_logits_1d, dim=0)
    for pos in range(num_speculative_steps + 1):
        accepted_mask = num_sampled >= pos + 1
        _assert_distribution_match(sampled[accepted_mask, pos], target_probs, device, label=f"position {pos}")

    gc.collect()
    torch.npu.empty_cache()
