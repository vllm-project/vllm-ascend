import numpy as np
import torch
import unittest

# Import the functions to test
from vllm_ascend.worker.v2.sample.bad_words import apply_bad_words, get_npu_vectorcore_num

class TestBadWords(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('npu')

    def test_get_npu_vectorcore_num(self):
        """Test get_npu_vectorcore_num function"""
        vectorcore_num = get_npu_vectorcore_num()
        self.assertGreater(vectorcore_num, 0, "Vector core number should be positive")
        print(f"NPU vector core number: {vectorcore_num}")

    def create_test_data(self, num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length):
        """Create test data for testing"""
        # Create logits
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=self.device)

        # Create expanded_idx_mapping (map each token to a request)
        expanded_idx_mapping = torch.randint(0, num_requests, (num_tokens,), dtype=torch.int32, device=self.device)

        # Create bad_word_token_ids and bad_word_offsets
        MAX_BAD_WORDS_TOTAL_TOKENS = 1024
        bad_word_token_ids = torch.zeros((num_requests, MAX_BAD_WORDS_TOTAL_TOKENS), dtype=torch.int32, device=self.device)
        bad_word_offsets = torch.zeros((num_requests, num_bad_words_per_req + 1), dtype=torch.int32, device=self.device)
        num_bad_words = torch.zeros(num_requests, dtype=torch.int32, device=self.device)

        # Fill bad words data
        for req_idx in range(num_requests):
            offset = 0
            for bw_idx in range(num_bad_words_per_req):
                # Create a bad word with specific tokens
                bad_word = torch.tensor([100 + req_idx * 10 + bw_idx] * bad_word_length, dtype=torch.int32, device=self.device)
                bad_word_token_ids[req_idx, offset:offset+bad_word_length] = bad_word
                bad_word_offsets[req_idx, bw_idx] = offset
                offset += bad_word_length
            bad_word_offsets[req_idx, num_bad_words_per_req] = offset
            num_bad_words[req_idx] = num_bad_words_per_req

        # Create all_token_ids with some matching bad words
        max_seq_len = 1024
        all_token_ids = torch.randint(0, vocab_size, (num_requests, max_seq_len), dtype=torch.int32, device=self.device)

        # Add some matching bad words to all_token_ids to ensure detection
        for req_idx in range(num_requests):
            if num_bad_words_per_req > 0:
                # Add the first bad word to the end of the sequence
                bad_word = bad_word_token_ids[req_idx, :bad_word_length]
                # Find a position where we can insert the bad word
                insert_pos = max_seq_len - bad_word_length
                all_token_ids[req_idx, insert_pos:insert_pos+bad_word_length] = bad_word

        # Create prompt_len and total_len
        prompt_len = torch.randint(10, 100, (num_requests,), dtype=torch.int32, device=self.device)
        # Set total_len to be large enough to include our inserted bad words
        total_len = torch.tensor([max_seq_len] * num_requests, dtype=torch.int32, device=self.device)

        # Create input_ids
        input_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32, device=self.device)

        # Create expanded_local_pos - set to a position where bad words can be detected
        expanded_local_pos = torch.full((num_tokens,), bad_word_length - 1, dtype=torch.int32, device=self.device)

        return (
            logits, expanded_idx_mapping, bad_word_token_ids, bad_word_offsets, num_bad_words,
            all_token_ids, prompt_len, total_len, input_ids, expanded_local_pos
        )

    def test_apply_bad_words_basic(self):
        """Test basic apply_bad_words functionality"""
        num_tokens = 1024
        vocab_size = 50257
        num_requests = 32
        num_bad_words_per_req = 5
        bad_word_length = 3

        test_data = self.create_test_data(
            num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length
        )

        # Make a copy of logits to compare
        logits_before = test_data[0].clone()
        logits_after = test_data[0].clone()

        # Apply bad words
        apply_bad_words(
            logits_after, *test_data[1:], num_bad_words_per_req
        )

        # Verify that logits were modified
        self.assertFalse(torch.allclose(logits_before, logits_after))
        print("Basic apply_bad_words test passed")

    def test_apply_bad_words_different_shapes(self):
        """Test apply_bad_words with different input shapes"""
        test_cases = [
            (512, 50257, 16, 3, 2),    # Small test case
            (1024, 50257, 32, 5, 3),   # Medium test case
            (2048, 50257, 64, 8, 4),   # Large test case
        ]

        for i, (num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length) in enumerate(test_cases):
            print(f"\nTesting case {i+1}: tokens={num_tokens}, requests={num_requests}")

            test_data = self.create_test_data(
                num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length
            )

            # Make a copy of logits to compare
            logits_before = test_data[0].clone()
            logits_after = test_data[0].clone()

            # Apply bad words
            apply_bad_words(
                logits_after, *test_data[1:], num_bad_words_per_req
            )

            # Verify that logits were modified
            self.assertFalse(torch.allclose(logits_before, logits_after))
            print(f"Case {i+1} passed")

    def test_apply_bad_words_no_bad_words(self):
        """Test apply_bad_words with no bad words"""
        num_tokens = 1024
        vocab_size = 50257
        num_requests = 32
        num_bad_words_per_req = 0
        bad_word_length = 3

        test_data = self.create_test_data(
            num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length
        )

        # Make a copy of logits to compare
        logits_before = test_data[0].clone()
        logits_after = test_data[0].clone()

        # Apply bad words
        apply_bad_words(
            logits_after, *test_data[1:], num_bad_words_per_req
        )

        # Verify that logits were not modified
        self.assertTrue(torch.allclose(logits_before, logits_after))
        print("No bad words test passed")


    def test_apply_bad_words_edge_cases(self):
        """Test apply_bad_words with edge cases"""
        # Test with maximum bad words
        num_tokens = 1024
        vocab_size = 50257
        num_requests = 16
        num_bad_words_per_req = 128  # Maximum allowed
        bad_word_length = 2

        print("\nTesting edge case: maximum bad words")
        test_data = self.create_test_data(
            num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length
        )

        # Make a copy of logits to compare
        logits_before = test_data[0].clone()
        logits_after = test_data[0].clone()

        # Apply bad words
        apply_bad_words(
            logits_after, *test_data[1:], num_bad_words_per_req
        )

        # Verify that logits were modified
        self.assertFalse(torch.allclose(logits_before, logits_after))
        print("Maximum bad words test passed")

    def test_apply_bad_words_token_limit(self):
        """Test apply_bad_words with token limit cases"""
        num_tokens = 1024
        vocab_size = 50257
        num_requests = 16

        # Test case 1: Total tokens within limit
        print("\nTesting case: total tokens within limit")
        num_bad_words_per_req = 32
        bad_word_length = 32  # 32 * 32 = 1024 tokens (exactly at limit)

        test_data = self.create_test_data(
            num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length
        )

        # Make a copy of logits to compare
        logits_before = test_data[0].clone()
        logits_after = test_data[0].clone()

        # Apply bad words
        apply_bad_words(
            logits_after, *test_data[1:], num_bad_words_per_req
        )

        # Verify that logits were modified
        self.assertFalse(torch.allclose(logits_before, logits_after))
        print("Total tokens within limit test passed")

        # Test case 2: Total tokens exceeding limit (this should still work but only process up to limit)
        print("\nTesting case: total tokens exceeding limit")
        num_bad_words_per_req = 33
        bad_word_length = 32  # 33 * 32 = 1056 tokens (exceeding limit)

        test_data = self.create_test_data(
            num_tokens, vocab_size, num_requests, num_bad_words_per_req, bad_word_length
        )

        # Make a copy of logits to compare
        logits_before = test_data[0].clone()
        logits_after = test_data[0].clone()

        # Apply bad words
        apply_bad_words(
            logits_after, *test_data[1:], num_bad_words_per_req
        )

        # Verify that logits were modified (even though we exceed the limit)
        self.assertFalse(torch.allclose(logits_before, logits_after))
        print("Total tokens exceeding limit test passed")

if __name__ == '__main__':
    unittest.main()
