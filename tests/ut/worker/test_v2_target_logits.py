# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.forward_context import override_forward_context
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler, get_max_chunk_logits

from vllm_ascend.ascend_forward_context import WRITABLE_TARGET_LOGITS_PTR
from vllm_ascend.worker.v2.sample.target_logits import reuse_writable_target_logits, wants_fp32_target_logits


@pytest.mark.parametrize("verifying", [False, True])
@pytest.mark.parametrize(
    "needs_processing,logprobs_mode,logprobs_dims,expected",
    [
        (False, "raw_logprobs", None, False),
        (True, "raw_logprobs", None, True),
        (True, "raw_logprobs", (0, 0), False),
        (True, "raw_logits", (0, 2), False),
        (True, "processed_logprobs", (2, 0), True),
        (True, "processed_logits", (2, 0), True),
    ],
)
def test_wants_fp32_target_logits(verifying, needs_processing, logprobs_mode, logprobs_dims, expected):
    sampler = object.__new__(Sampler)
    sampler.needs_logits_processing = np.array([False, needs_processing])
    sampler.logprobs_mode = logprobs_mode
    sampler.get_logprobs_dims = Mock(return_value=logprobs_dims)
    rejection_sampler = object.__new__(RejectionSampler) if verifying else None
    input_batch = SimpleNamespace(
        num_draft_tokens=int(verifying),
        logits_indices=torch.arange(1 + int(verifying)),
        idx_mapping_np=np.array([1]),
    )
    vocab_size = 129280

    assert wants_fp32_target_logits(sampler, rejection_sampler, input_batch, vocab_size) is expected
    if needs_processing and logprobs_mode.startswith("raw_"):
        sampler.get_logprobs_dims.assert_called_once_with(input_batch.idx_mapping_np, include_token_ids=not verifying)
    if verifying:
        input_batch.logits_indices = torch.arange(get_max_chunk_logits(vocab_size) + 1)
        assert not wants_fp32_target_logits(sampler, rejection_sampler, input_batch, vocab_size)


def test_sampler_reuses_only_marked_storage():
    original = Mock(side_effect=lambda _, logits, *args: logits.float().clone())
    patched = reuse_writable_target_logits(original)
    sampler = SimpleNamespace(
        logits_processors=[],
        thinking_budget_state=Mock(),
        sampling_states=Mock(),
    )
    sampler.sampling_states.apply_temperature.side_effect = lambda logits, *_: logits.mul_(0.5)
    logits = torch.tensor([[2.0, 6.0]])
    other = logits.clone()
    request_indices = torch.zeros(1, dtype=torch.int32)
    indices_np = request_indices.numpy()
    args = (request_indices, request_indices, indices_np, request_indices, request_indices, request_indices, indices_np)
    context = SimpleNamespace(additional_kwargs={WRITABLE_TARGET_LOGITS_PTR: logits.data_ptr()})

    with override_forward_context(context):
        assert patched(sampler, other, *args, skip_top_k_top_p=True) is not other
        # RejectionSampler passes a view of the single verification chunk.
        chunk = logits[:1]
        assert patched(sampler, chunk, *args, skip_top_k_top_p=True) is chunk

    torch.testing.assert_close(logits, torch.tensor([[1.0, 3.0]]))
    torch.testing.assert_close(other, torch.tensor([[2.0, 6.0]]))
    assert patched(sampler, logits, *args, skip_top_k_top_p=True) is not logits
    assert original.call_count == 2
