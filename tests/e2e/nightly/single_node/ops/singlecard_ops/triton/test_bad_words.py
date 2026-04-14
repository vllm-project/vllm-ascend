# SPDX-License-Identifier: Apache-2.0
# Test vllm_ascend.worker.v2.sample.bad_words.apply_bad_words (Triton-Ascend).
# Requires NPU and Triton-Ascend.

import pytest
import torch

from vllm_ascend.worker.v2.sample.bad_words import apply_bad_words

# Test cases for different input shapes
BAD_WORDS_TEST_CASES = [
    pytest.param(512, 50257, 16, 3, 2, id="small-case"),
    pytest.param(1024, 50257, 32, 5, 3, id="medium-case"),
    pytest.param(2048, 50257, 64, 8, 4, id="large-case"),
]


def create_test_data(num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length, device):
    """Create test data for testing"""
    # Create logits
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=device)

    # Create expanded_idx_mapping (map each token to a request)
    expanded_idx_mapping = torch.randint(0, num_requests, (num_tokens,), dtype=torch.int32, device=device)

    # Create bad_word_token_ids and bad_word_offsets
    MAX_BAD_WORDS_TOTAL_TOKENS = 1024
    MAX_NUM_BAD_WORDS = 128
    bad_word_token_ids = torch.zeros((num_requests, MAX_BAD_WORDS_TOTAL_TOKENS), dtype=torch.int32, device=device)
    bad_word_offsets = torch.zeros((num_requests, MAX_NUM_BAD_WORDS + 1), dtype=torch.int32, device=device)
    num_bad_words = torch.zeros(num_requests, dtype=torch.int32, device=device)

    # Fill bad words data
    for req_idx in range(num_requests):
        offset = 0
        actual_bad_words = 0
        for bw_idx in range(num_bad_words_per_req):
            # Check if adding this bad word would exceed the token limit
            if offset + bad_word_length > MAX_BAD_WORDS_TOTAL_TOKENS:
                break
            # Create a bad word with specific tokens
            bad_word = torch.tensor([100 + req_idx * 10 + bw_idx] * bad_word_length, dtype=torch.int32, device=device)
            bad_word_token_ids[req_idx, offset:offset+bad_word_length] = bad_word
            bad_word_offsets[req_idx, bw_idx] = offset
            offset += bad_word_length
            actual_bad_words += 1
        bad_word_offsets[req_idx, actual_bad_words] = offset
        num_bad_words[req_idx] = actual_bad_words

    # Create all_token_ids with some matching bad words
    max_seq_len = 1024
    all_token_ids = torch.randint(0, vocab_size, (num_requests, max_seq_len), dtype=torch.int32, device=device)

    # Create prompt_len and total_len
    prompt_len = torch.tensor([50] * num_requests, dtype=torch.int32, device=device)
    total_len = torch.tensor([max_seq_len] * num_requests, dtype=torch.int32, device=device)

    # Create input_ids with the same bad words, so they can be detected
    input_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32, device=device)
    # For each token, set input_ids to match the bad word for its request
    for token_idx in range(num_tokens):
        req_idx = expanded_idx_mapping[token_idx].item()
        if num_bad_words[req_idx] > 0:
            # Set input_ids to match the first bad word
            bad_word = bad_word_token_ids[req_idx, :bad_word_length]
            # For each position in the bad word, set input_ids accordingly
            for i in range(bad_word_length):
                if token_idx - i >= 0:
                    input_ids[token_idx - i] = bad_word[bad_word_length - 1 - i]

    # Create expanded_local_pos - set to bad_word_length - 1 so that effective_len = output_len + (bad_word_length - 1)
    # This ensures that we're checking the current token as the end of a bad word
    expanded_local_pos = torch.full((num_tokens,), bad_word_length - 1, dtype=torch.int32, device=device)

    return (
        logits, expanded_idx_mapping, bad_word_token_ids, bad_word_offsets, num_bad_words,
        all_token_ids, prompt_len, total_len, input_ids, expanded_local_pos
    )


@pytest.mark.parametrize("num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length", BAD_WORDS_TEST_CASES)
@torch.inference_mode()
def test_apply_bad_words_different_shapes(num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length, device="npu"):
    """Test apply_bad_words with different input shapes"""
    test_data = create_test_data(
        num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length, device
    )

    # Make a copy of logits to compare
    logits_before = test_data[0].clone()
    logits_after = test_data[0].clone()

    # Apply bad words
    apply_bad_words(
        logits_after, *test_data[1:], num_bad_words_per_req
    )

    # Verify that logits were modified
    assert not torch.allclose(logits_before, logits_after), "Logits should be modified when bad words are present"
    print(f"Test passed: tokens={num_tokens}, requests={num_requests}")


@torch.inference_mode()
def test_apply_bad_words_no_bad_words(device="npu"):
    """Test apply_bad_words with no bad words"""
    num_tokens = 1024
    vocab_size = 50257
    num_requests = 32
    num_bad_words_per_req = 0
    bad_word_length = 3

    test_data = create_test_data(
        num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length, device
    )

    # Make a copy of logits to compare
    logits_before = test_data[0].clone()
    logits_after = test_data[0].clone()

    # Apply bad words
    apply_bad_words(
        logits_after, *test_data[1:], num_bad_words_per_req
    )

    # Verify that logits were not modified
    assert torch.allclose(logits_before, logits_after), "Logits should not be modified when no bad words are present"
    print("No bad words test passed")


@torch.inference_mode()
def test_apply_bad_words_edge_cases(device="npu"):
    """Test apply_bad_words with edge cases"""
    # Test with maximum bad words
    num_tokens = 1024
    vocab_size = 50257
    num_requests = 16
    num_bad_words_per_req = 128  # Maximum allowed
    bad_word_length = 2

    print("\nTesting edge case: maximum bad words")
    test_data = create_test_data(
        num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length, device
    )

    # Make a copy of logits to compare
    logits_before = test_data[0].clone()
    logits_after = test_data[0].clone()

    # Apply bad words
    apply_bad_words(
        logits_after, *test_data[1:], num_bad_words_per_req
    )

    # Verify that logits were modified
    assert not torch.allclose(logits_before, logits_after), "Logits should be modified when maximum bad words are present"
    print("Maximum bad words test passed")


@torch.inference_mode()
def test_apply_bad_words_token_limit(device="npu"):
    """Test apply_bad_words with token limit cases"""
    num_tokens = 1024
    vocab_size = 50257
    num_requests = 16

    # Test case 1: Total tokens within limit
    print("\nTesting case: total tokens within limit")
    num_bad_words_per_req = 32
    bad_word_length = 32  # 32 * 32 = 1024 tokens (exactly at limit)

    test_data = create_test_data(
        num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length, device
    )

    # Make a copy of logits to compare
    logits_before = test_data[0].clone()
    logits_after = test_data[0].clone()

    # Apply bad words
    apply_bad_words(
        logits_after, *test_data[1:], num_bad_words_per_req
    )

    # Verify that logits were modified
    assert not torch.allclose(logits_before, logits_after), "Logits should be modified when total tokens are within limit"
    print("Total tokens within limit test passed")

    # Test case 2: Total tokens exceeding limit (this should still work but only process up to limit)
    print("\nTesting case: total tokens exceeding limit")
    num_bad_words_per_req = 33
    bad_word_length = 32  # 33 * 32 = 1056 tokens (exceeding limit)

    test_data = create_test_data(
        num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length, device
    )

    # Make a copy of logits to compare
    logits_before = test_data[0].clone()
    logits_after = test_data[0].clone()

    # Apply bad words
    apply_bad_words(
        logits_after, *test_data[1:], num_bad_words_per_req
    )

    # Verify that logits were modified (even though we exceed the limit)
    assert not torch.allclose(logits_before, logits_after), "Logits should be modified when total tokens exceed limit"
    print("Total tokens exceeding limit test passed")
