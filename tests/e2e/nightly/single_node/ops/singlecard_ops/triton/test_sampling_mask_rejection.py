# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend integration coverage adapted from upstream rejection-sampler tests.

Reference: vLLM tests/v1/worker/test_gpu_rejection_sampler_chunking.py,
test_replay_on_off_preserves_rejection_sampling_and_rng.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.sample import logprob
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as upstream

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.v2.sample.sampling_mask import sampling_mask_from_logits_npu
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

pytestmark = [
    pytest.mark.skipif(
        get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
        reason="Requires Ascend A2/A3/A5",
    ),
    pytest.mark.skipif(
        "rows_per_request" not in SamplingMaskTensors._fields,
        reason="Requires upstream request-major rejection sampling",
    ),
]


def _make_verifier(seed_offset):
    """Use real verification and logprob methods with a minimal request state."""
    num_reqs, speculative_steps, vocab_size = 3, 3, 8
    device = torch.device("npu")
    cu = np.arange(0, 13, speculative_steps + 1, dtype=np.int32)
    mapping = np.arange(num_reqs, dtype=np.int32)
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        cu_num_logits_np=cu,
        cu_num_logits=torch.from_numpy(cu).to(device),
        idx_mapping_np=mapping,
        idx_mapping=torch.from_numpy(mapping).to(device),
        expanded_idx_mapping=torch.from_numpy(np.repeat(mapping, 4)).to(device),
        expanded_local_pos=torch.from_numpy(np.tile(np.arange(4, dtype=np.int32), 3)).to(device),
        seq_lens_cpu_upper_bound=torch.full((num_reqs,), 64, dtype=torch.int32),
    )
    sampler = SimpleNamespace(
        logprobs_mode="processed_logprobs",
        return_sampling_mask=False,
        sampling_states=SimpleNamespace(
            top_k=SimpleNamespace(np=np.full(num_reqs, vocab_size)),
            temperature=SimpleNamespace(gpu=torch.ones(num_reqs, device=device)),
            seeds=SimpleNamespace(gpu=torch.arange(num_reqs, dtype=torch.int64, device=device) + seed_offset),
        ),
        use_fp64_gumbel=False,
        apply_sampling_params=lambda logits, *_: logits,
    )
    verifier = object.__new__(upstream.RejectionSampler)
    verifier.sampler = sampler
    verifier.num_speculative_steps = speculative_steps
    verifier.enable_adaptive_verification = False
    verifier.synthetic_conditional_rates = None
    verifier.use_block_verification = False
    verifier.fly_window_size = 0
    verifier.fly_entropy_threshold = 0.3
    verifier.fly_entropy_top_k = 3
    verifier.draft_logits_num_shards = 1
    verifier.watermark_key = None
    return verifier, batch


@pytest.mark.parametrize("max_chunk_logits", [8, 64])
@pytest.mark.parametrize("max_num_logprobs", [0, 3])
@pytest.mark.parametrize("seed_offset", [0, 123])
def test_replay_on_off_preserves_rejection_sampling_and_rng(
    monkeypatch, max_chunk_logits, max_num_logprobs, seed_offset
):
    """Mask capture must preserve actual NPU rejection sampling and RNG state.

    The requests reject immediately, reject at step two, and accept every draft.
    Cover both chunk concatenation and a single verification chunk.
    """
    init_device_properties_triton()
    monkeypatch.setattr(upstream, "rejection_sample", rejection_sample)
    monkeypatch.setattr(logprob, "compute_token_logprobs", compute_token_logprobs)
    pack = sampling_mask_from_logits_npu
    pack_calls = []

    def track_pack(cls, logits, cu_num_logits, num_sampled_tokens, max_num_kept, rows_per_request=1):
        pack_calls.append((cu_num_logits.cpu().tolist(), max_num_kept, rows_per_request))
        return pack(cls, logits, cu_num_logits, num_sampled_tokens, max_num_kept, rows_per_request)

    monkeypatch.setattr(SamplingMaskTensors, "from_logits", classmethod(track_pack))
    verifier, batch = _make_verifier(seed_offset)
    logits = torch.zeros((12, 8), device="npu")
    logits[0, 0] = -float("inf")
    logits[5, 1] = -float("inf")
    draft_logits = torch.zeros((3, 3, 8), device="npu")
    draft_sampled = torch.zeros((3, 4), dtype=torch.int64, device="npu")
    draft_sampled[:, 1:] = torch.tensor([0, 1, 2], device="npu")
    args = (logits, batch, draft_logits, draft_sampled.flatten(), torch.arange(12, dtype=torch.int32, device="npu"))

    # Explicit request seeds drive the real sampler; neither pass should consume
    # the global NPU RNG. Do not reset RNG between passes and hide consumption.
    rng_before = torch.npu.get_rng_state().clone()
    off = verifier._verify_in_chunks(*args, max_chunk_logits=max_chunk_logits, max_num_logprobs=max_num_logprobs)
    rng_after_off = torch.npu.get_rng_state().clone()
    assert pack_calls == []
    verifier.sampler.return_sampling_mask = True
    on = verifier._verify_in_chunks(*args, max_chunk_logits=max_chunk_logits, max_num_logprobs=max_num_logprobs)
    rng_after_on = torch.npu.get_rng_state().clone()

    expected_calls = [([0, 4, 8], 8, 4), ([0, 4], 8, 4)] if max_chunk_logits == 8 else [([0, 4, 8, 12], 8, 4)]
    assert pack_calls == expected_calls
    off_sampled, off_counts, off_logprobs, off_masks = off
    on_sampled, on_counts, on_logprobs, on_masks = on
    assert off_masks is None and on_masks is not None
    assert off_counts.tolist() == [1, 2, 4]
    assert torch.equal(off_counts, on_counts)
    counts = off_counts.cpu().tolist()
    for req, count in enumerate(counts):
        assert torch.equal(off_sampled[req, :count], on_sampled[req, :count])
    assert torch.equal(rng_before, rng_after_off)
    assert torch.equal(rng_after_off, rng_after_on)

    # Compare every emitted support and its corresponding token/logprob, not
    # padding in the sampler's output buffers, which is intentionally undefined.
    mapped = [int(batch.cu_num_logits_np[req]) + slot for req, count in enumerate(counts) for slot in range(count)]
    assert off_logprobs is not None and on_logprobs is not None
    for field in ("logprob_token_ids", "logprobs", "selected_token_ranks"):
        assert torch.equal(getattr(off_logprobs, field)[mapped], getattr(on_logprobs, field)[mapped])
    assert on_masks.rows_per_request == 4
    assert on_masks.token_ids.shape == (12, 0)
    assert on_masks.packed_mask.shape == (12, 1)
    masks = on_masks.to_cpu_nonblocking().tolists(on_counts.cpu().numpy()).to_nested_list()
    expected_masks = [torch.isfinite(logits[row]).nonzero().flatten().tolist() for row in mapped]
    assert masks == expected_masks
    emitted = [int(on_sampled[req, slot]) for req, count in enumerate(counts) for slot in range(count)]
    assert all(token in mask for token, mask in zip(emitted, masks))
    expected_logprobs = torch.stack([torch.log_softmax(logits[row], -1)[token] for row, token in zip(mapped, emitted)])
    torch.testing.assert_close(on_logprobs.logprobs[mapped, 0], expected_logprobs, rtol=1e-5, atol=1e-6)
