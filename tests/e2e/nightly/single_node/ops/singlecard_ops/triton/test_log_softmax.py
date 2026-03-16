import torch
import pytest
from vllm.triton_utils import triton
from vllm_ascend.worker.v2.sample.logprob import _topk_log_softmax_kernel


@pytest.mark.parametrize("batch_size,vocab_size,num_logprobs", [
    (48, 102400, 50),
    (96, 102400, 1),
    (24, 151936, 8),
])
def test_topk_log_softmax_kernel(batch_size, vocab_size, num_logprobs):
    """Test _topk_log_softmax_kernel for computing log probabilities
    Args:
        batch_size: Number of sequences in the batch
        vocab_size: Size of the vocabulary
        num_logprobs: Number of tokens to compute log probabilities for
    """
    # ========== Setup test data ==========
    torch.manual_seed(42)

    # Generate random logits
    logits = torch.randn(batch_size, vocab_size, device='npu', dtype=torch.float32)

    # Generate token_ids for which to compute logprobs
    token_ids = torch.randint(0, vocab_size, (batch_size, num_logprobs), 
                             device='npu', dtype=torch.int64)

    # ========== Execute test ==========
    # Prepare output tensor
    triton_output = torch.empty(
        batch_size, num_logprobs,
        dtype=torch.float32,
        device='npu'
    )

    # Invoke Triton kernel
    _topk_log_softmax_kernel[(batch_size,)](
        triton_output,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,
        PADDED_TOPK=max(triton.next_power_of_2(num_logprobs), 2),
    )
    torch.npu.synchronize()

    # Compute reference values using PyTorch
    torch_logprobs = torch.log_softmax(logits, dim=-1)

    # Extract logprobs for each batch and token_id
    ref_output = torch.zeros_like(triton_output)
    for i in range(batch_size):
        for j in range(num_logprobs):
            token_id = token_ids[i, j]
            ref_output[i, j] = torch_logprobs[i, token_id]

    # ========== Verify results ==========
    assert torch.allclose(triton_output, ref_output, rtol=1e-3, atol=1e-3), \
        f"Triton output differs from PyTorch reference.\n" \
        f"Max diff: {torch.max(torch.abs(triton_output - ref_output))}\n" \
        f"Mean diff: {torch.mean(torch.abs(triton_output - ref_output))}"
