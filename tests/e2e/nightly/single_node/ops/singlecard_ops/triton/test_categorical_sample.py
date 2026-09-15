# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

DEVICE = "npu"
NUM_TRIALS = 4096


@pytest.fixture(scope="module", autouse=True)
def _npu_env():
    init_device_properties_triton()
    yield
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()


@torch.inference_mode()
def test_rejection_sample_residual_rng_is_independent_from_acceptance_rng():
    """Rejected-token resampling must use RNG independent from acceptance."""
    probs = torch.tensor([0.50, 0.25, 0.15, 0.10], dtype=torch.float32)
    num_reqs = NUM_TRIALS
    num_speculative_steps = 1
    num_logits_per_req = num_speculative_steps + 1
    num_logits = num_reqs * num_logits_per_req

    target_logits = probs.log().to(DEVICE).repeat(num_logits, 1).contiguous()
    draft_sampled = torch.zeros(num_logits, dtype=torch.int32, device=DEVICE)
    cu_num_logits = torch.arange(0, num_logits + 1, num_logits_per_req, dtype=torch.int32, device=DEVICE)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = idx_mapping.repeat_interleave(num_logits_per_req)
    expanded_local_pos = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE).repeat(num_reqs)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE) * 104729 + 17
    pos = torch.arange(num_logits, dtype=torch.int64, device=DEVICE) * 7 + 11

    sampled, _ = rejection_sample(
        target_logits,
        None,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps,
    )
    torch.npu.synchronize()

    counts = torch.bincount(sampled[:, 0].cpu(), minlength=probs.numel()).to(torch.float64)
    observed = counts / counts.sum()
    torch.testing.assert_close(observed, probs.to(torch.float64), rtol=0.0, atol=0.03)
