# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.sample.logprob import compute_topk_logprobs


@pytest.mark.parametrize("num_logprobs", [0, 2])
@pytest.mark.parametrize("logits_mode", [False, True])
@pytest.mark.parametrize("tensor_metadata", [False, True])
def test_custom_token_scores(num_logprobs, logits_mode, tensor_metadata):
    """Mixed requests preserve requested order, expanded mapping and padding."""
    init_device_properties_triton()
    logits = torch.tensor(
        [
            [0., 1., 2., 3., 4., 5.],
            [5., 4., 3., 2., 1., 0.],
            [1., 3., 5., 0., 2., 4.],
            [4., 2., 0., 5., 3., 1.],
        ],
        device="npu",
    )
    sampled = torch.tensor([3, 1, 2, 0], device="npu")
    mapping = torch.tensor([2, 0, 1, 2], dtype=torch.int32, device="npu")
    state = SimpleNamespace(
        num_token_ids=SimpleNamespace(gpu=torch.tensor([2, 0, 1], dtype=torch.int32, device="npu")),
        token_ids=SimpleNamespace(gpu=torch.tensor([[5, 0], [0, 0], [1, 0]], dtype=torch.int32, device="npu")),
    )
    metadata = (
        torch.tensor([0, 2, 4], dtype=torch.int32, device="npu") if tensor_metadata else [0, 2, 4]
    )
    result = compute_topk_logprobs(
        logits, num_logprobs, sampled, metadata, state, mapping, 2, logits_mode
    )
    expected_ids = torch.zeros((4, 3), dtype=sampled.dtype, device="npu")
    valid = torch.zeros((4, 3), dtype=torch.bool, device="npu")
    expected_ids[:, 0] = sampled
    valid[:, 0] = True
    # Rows 0 and 3 map to request 2, row 1 to request 0, row 2 uses top-k.
    expected_ids[0, 1] = expected_ids[3, 1] = 1
    valid[0, 1] = valid[3, 1] = True
    expected_ids[1, 1:] = torch.tensor([5, 0], device="npu")
    valid[1, 1:] = True
    if num_logprobs:
        expected_ids[2, 1:] = torch.tensor([2, 5], device="npu")
        valid[2, 1:] = True
    scores = logits if logits_mode else logits.log_softmax(-1)
    expected = scores.gather(-1, expected_ids).masked_fill(~valid, -float("inf"))
    torch.testing.assert_close(result.logprob_token_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(result.logprobs, expected, rtol=1e-4, atol=1e-4)
    if tensor_metadata:
        assert result.cu_num_generated_tokens is None
        assert result.cu_num_generated_tokens_tensor is metadata
    else:
        assert result.cu_num_generated_tokens is metadata
        assert result.cu_num_generated_tokens_tensor is None
