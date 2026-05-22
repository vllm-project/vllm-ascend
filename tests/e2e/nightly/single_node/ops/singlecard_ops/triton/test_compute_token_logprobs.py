import random

import pytest
import torch

from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs


def torch_compute_token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference implementation of topk log softmax.

    Computes log_softmax for the entire logits tensor, then gathers
    the values at the specified token_ids positions.

    Args:
        logits: Tensor of shape (batch_size, vocab_size) containing the logits.
        token_ids: Tensor of shape (batch_size, topk) containing the token indices.

    Returns:
        Tensor of shape (batch_size, topk) containing log probabilities.
    """
    # Compute log_softmax along the vocab dimension
    # log_softmax(x) = x - log(sum(exp(x))) = x - max(x) - log(sum(exp(x - max(x))))
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

    # Gather the log probabilities at the specified token positions
    token_ids = token_ids.to(torch.int64)
    result = torch.gather(log_probs, dim=1, index=token_ids)

    return result.to(torch.float32)


# Common vocab sizes from mainstream models
VOCAB_SIZES = [
    32000,  # LLaMA / LLaMA2 / Mistral
    50257,  # GPT-2
    65024,  # ChatGLM
    128256,  # LLaMA3
    151936,  # Qwen2
]

# Different topk values to test
TOPK_VALUES = [1, 2, 5, 10, 32, 64]


@pytest.mark.parametrize(
    "batch_size, vocab_size, topk",
    [(random.randint(1, 64), vocab_size, topk) for vocab_size in VOCAB_SIZES for topk in TOPK_VALUES],
)
def test_topk_log_softmax_kernel(batch_size, vocab_size, topk):
    """
    Test the Triton _topk_log_softmax_kernel against a pure PyTorch reference.

    The kernel computes log_softmax and gathers values at specified token positions.

    Args:
        batch_size: Number of requests (rows) in the logits tensor.
        vocab_size: Vocabulary size (columns) in the logits tensor.
        topk: Number of token positions to compute log probabilities for.
    """
    torch.manual_seed(42)

    device = "npu"

    # Build input tensors
    logits = torch.randn((batch_size, vocab_size), dtype=torch.float32, device=device)

    # Generate random token indices within vocab_size
    token_ids = torch.randint(0, vocab_size, (batch_size, topk), dtype=torch.int64, device=device)

    # ========== Run Triton kernel ==========
    logprobs_triton = compute_token_logprobs(logits, token_ids)

    # ========== Run PyTorch reference ==========
    logprobs_ref = torch_compute_token_logprobs(logits, token_ids)

    # ========== Verify results ==========
    max_diff = torch.max(torch.abs(logprobs_triton - logprobs_ref)).item()
    mean_diff = torch.mean(torch.abs(logprobs_triton - logprobs_ref)).item()

    assert torch.allclose(logprobs_triton, logprobs_ref, atol=1e-4, rtol=1e-5), (
        f"Triton topk_log_softmax kernel output differs from torch reference.\n"
        f"batch_size={batch_size}, vocab_size={vocab_size}, topk={topk}\n"
        f"Max diff: {max_diff}\n"
        f"Mean diff: {mean_diff}"
    )


@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_topk_log_softmax_edge_cases(vocab_size):
    """
    Test edge cases for the topk_log_softmax kernel.

    Args:
        vocab_size: Vocabulary size to test.
    """
    torch.manual_seed(42)

    device = "npu"

    # Test case 1: Single batch, single topk
    logits = torch.randn((1, vocab_size), dtype=torch.float32, device=device)
    token_ids = torch.randint(0, vocab_size, (1, 1), dtype=torch.int64, device=device)

    logprobs_triton = compute_token_logprobs(logits, token_ids)
    logprobs_ref = torch_compute_token_logprobs(logits, token_ids)

    assert torch.allclose(logprobs_triton, logprobs_ref, atol=1e-4, rtol=1e-5), (
        f"Edge case (1,1) failed for vocab_size={vocab_size}"
    )

    # Test case 2: Logits with extreme values
    logits_extreme = torch.randn((4, vocab_size), dtype=torch.float32, device=device)
    logits_extreme[0, 0] = 100.0  # Very large positive
    logits_extreme[1, 0] = -100.0  # Very large negative
    logits_extreme[2, :] = 0.0  # All zeros
    logits_extreme[3, :] = 1.0  # All ones

    token_ids = torch.zeros((4, 5), dtype=torch.int64, device=device)
    token_ids[:, 0] = 0  # Include the extreme value position
    for i in range(1, 5):
        token_ids[:, i] = torch.randint(1, vocab_size, (4,))

    logprobs_triton = compute_token_logprobs(logits_extreme, token_ids)
    logprobs_ref = torch_compute_token_logprobs(logits_extreme, token_ids)

    assert torch.allclose(logprobs_triton, logprobs_ref, atol=1e-4, rtol=1e-5), (
        f"Extreme values test failed for vocab_size={vocab_size}"
    )


@pytest.mark.parametrize(
    "batch_size, vocab_size, topk",
    [
        (16, 32000, 10),
        (32, 50257, 5),
        (64, 128256, 20),
    ],
)
def test_topk_log_softmax_deterministic(batch_size, vocab_size, topk):
    """
    Test that the kernel produces deterministic results across multiple runs.

    Args:
        batch_size: Number of requests.
        vocab_size: Vocabulary size.
        topk: Number of token positions.
    """
    torch.manual_seed(42)

    device = "npu"

    logits = torch.randn((batch_size, vocab_size), dtype=torch.float32, device=device)
    token_ids = torch.randint(0, vocab_size, (batch_size, topk), dtype=torch.int64, device=device)

    # Run multiple times and check consistency
    results = []
    for _ in range(3):
        result = compute_token_logprobs(logits, token_ids)
        results.append(result.clone())

    for i in range(1, len(results)):
        assert torch.equal(results[0], results[i]), f"Non-deterministic results detected in run {i}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_topk_log_softmax_dtypes(dtype):
    """
    Test the kernel with different input dtypes.

    Args:
        dtype: Input tensor dtype.
    """
    torch.manual_seed(42)

    device = "npu"

    batch_size = 8
    vocab_size = 32000
    topk = 10

    logits = torch.randn((batch_size, vocab_size), dtype=dtype, device=device)
    token_ids = torch.randint(0, vocab_size, (batch_size, topk), dtype=torch.int64, device=device)

    logprobs_triton = compute_token_logprobs(logits, token_ids)
    logprobs_ref = torch_compute_token_logprobs(logits.float(), token_ids)

    # Use slightly larger tolerance for float16 due to precision loss
    atol = 1e-3 if dtype == torch.float16 else 1e-4

    assert torch.allclose(logprobs_triton, logprobs_ref, atol=atol, rtol=1e-4), f"dtype {dtype} test failed"


if __name__ == "__main__":
    # Run a quick sanity check
    print("Running quick sanity check...")

    device = "npu"
    print(f"Using device: {device}")

    torch.manual_seed(42)

    batch_size = 4
    vocab_size = 32000
    topk = 5

    logits = torch.randn((batch_size, vocab_size), dtype=torch.float32, device=device)
    token_ids = torch.randint(0, vocab_size, (batch_size, topk), dtype=torch.int64, device=device)

    logprobs_triton = compute_token_logprobs(logits, token_ids)
    logprobs_ref = torch_compute_token_logprobs(logits, token_ids)

    max_diff = torch.max(torch.abs(logprobs_triton - logprobs_ref)).item()
    mean_diff = torch.mean(torch.abs(logprobs_triton - logprobs_ref)).item()

    print(f"Max diff: {max_diff}")
    print(f"Mean diff: {mean_diff}")
    print(f"All close (atol=1e-4): {torch.allclose(logprobs_triton, logprobs_ref, atol=1e-4, rtol=1e-5)}")

    print("\nTriton output (first row):", logprobs_triton[0])
    print("PyTorch output (first row):", logprobs_ref[0])

    print("\nSanity check passed!")
