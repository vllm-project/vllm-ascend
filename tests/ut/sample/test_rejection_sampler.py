from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.rejection_sampler import (
    expand_batch_to_tokens,
    expand_pytorch,
    rejection_greedy_sample_pytorch,
    rejection_random_sample_block_verify_pytorch,
    rejection_random_sample_pytorch,
    sample_recovered_tokens_blockwise_pytorch,
    sample_recovered_tokens_pytorch,
)

# Global constants
PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = 0.0
MAX_SPEC_LEN = 8  # Used as MAX_NUM_TOKENS in expand_batch_to_tokens


def mock_pin_memory(original_func):
    def func_wo_pin_memory(*args, **kwargs):
        if kwargs.get("pin_memory", False):
            kwargs["pin_memory"] = False
        return original_func(*args, **kwargs)

    return func_wo_pin_memory


class TestAscendRejectionSampler(TestBase):
    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_greedy_sample_pytorch(self):
        """Test greedy rejection sampling: stop when draft doesn't match, otherwise append bonus token"""
        batch_size = 2
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 4])
        num_draft_tokens = [2, 2]
        draft_token_ids = torch.tensor([10, 11, 20, 21])
        target_argmax = torch.tensor([10, 99, 20, 22])
        bonus_token_ids = torch.tensor([[100], [200]])

        is_greedy = torch.tensor([True, True])

        rejection_greedy_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            num_draft_tokens,
            max_spec_len,
            is_greedy,
        )

        assert output_token_ids[0, 0].item() == 10
        assert output_token_ids[0, 1].item() == 99
        assert output_token_ids[1, 0].item() == 20
        assert output_token_ids[1, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_pytorch(self):
        """Test random rejection sampling: accept based on uniform probability"""
        batch_size = 2
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 1])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.0, 0.4],  # vocab_size=4
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.5, 0.0, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.8, 0.0, 0.2],
                [0.2, 0.1, 0.3, 0.4],
                [0.9, 0.1, 0.0, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([1, 2, 3])
        uniform_probs = torch.tensor([0.7, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 4

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 0
        assert output_token_ids[0, 2].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_pytorch_rejects_placeholder(self):
        batch_size = 1
        max_spec_len = 1
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1])
        draft_token_ids = torch.tensor([PLACEHOLDER_TOKEN_ID])
        target_probs = torch.tensor([[0.0, 0.0, 1.0]])
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([2])
        uniform_probs = torch.tensor([0.0])
        is_greedy = torch.tensor([False])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            None,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size=3,
            IS_NGRAM=True,
        )

        assert output_token_ids.tolist() == [[2, PLACEHOLDER_TOKEN_ID]]

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_pytorch_rejects_all_placeholder_mtp3(self):
        batch_size = 1
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([3])
        draft_token_ids = torch.tensor([PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID])
        # Placeholder draft tokens must reject regardless of target probability.
        # The recovered token is passed in after recovery sampling.
        target_probs = torch.zeros((max_spec_len, 3))
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([2, 1, 0])
        uniform_probs = torch.tensor([0.0, 0.0, 0.0])
        is_greedy = torch.tensor([False])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            None,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size=3,
            IS_NGRAM=True,
        )

        assert output_token_ids.tolist() == [[2, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]]

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_sample_recovered_tokens_pytorch_keeps_placeholder_distribution(self):
        output_token_ids = torch.empty(1, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1])
        draft_token_ids = torch.tensor([PLACEHOLDER_TOKEN_ID])
        target_probs = torch.tensor([[0.1, 0.2, 0.7]])
        q = torch.ones((1, 3), dtype=torch.float32)

        sample_recovered_tokens_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            None,
            target_probs,
            q,
            vocab_size=3,
            IS_NGRAM=True,
        )

        assert output_token_ids.tolist() == [2]

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_expand_pytorch(self):
        """Test expand_pytorch functionality"""
        input_ptr = torch.tensor([10, 20, 30], dtype=torch.int32)
        cu_num_tokens_ptr = torch.tensor([2, 5, 7])
        output_ptr = torch.empty(7, dtype=torch.int32)

        expand_pytorch(
            output_ptr,
            input_ptr,
            cu_num_tokens_ptr,
            replace_from=0,
            replace_to=0,
            MAX_NUM_TOKENS=MAX_SPEC_LEN,
        )

        expected = torch.tensor([10, 10, 20, 20, 20, 30, 30])
        assert torch.equal(output_ptr, expected)

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_expand_batch_to_tokens(self):
        """Test expand_batch_to_tokens wrapper"""
        x = torch.tensor([10, 20, 30])
        cu_num_tokens = torch.tensor([2, 5, 7])
        num_tokens = 7
        # Test PyTorch path
        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
            patch("vllm_ascend.sample.rejection_sampler.expand_pytorch") as mock_pytorch,
        ):
            expand_batch_to_tokens(x, cu_num_tokens, num_tokens)
            mock_pytorch.assert_called_once()
            args = mock_pytorch.call_args[0]
            assert (args[1] == x).all()
            assert (args[2] == cu_num_tokens).all()

        # Test Triton kernel path
        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", True),
            patch("vllm_ascend.sample.rejection_sampler.expand_triton") as mock_triton,
        ):
            expand_batch_to_tokens(x, cu_num_tokens, num_tokens)
            mock_triton.assert_called_once()
            call_args = mock_triton.call_args[0]
            assert (call_args[2] == x).all()
            assert (call_args[3] == cu_num_tokens).all()

        # Run actual function
        with patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False):
            result = expand_batch_to_tokens(x, cu_num_tokens, num_tokens)
            expected = torch.tensor([10, 10, 20, 20, 20, 30, 30])
            assert torch.equal(result, expected)

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_sample_recovered_tokens_pytorch_ngram(self):
        """Test recovered token sampling under n-gram mode"""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 2])
        draft_probs = None
        target_probs = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.3, 0.3, 0.4],
            ]
        )
        q = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.4, 0.1],
            ]
        )
        vocab_size = 3

        sample_recovered_tokens_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=True,
        )

        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 1

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_reduce_sample_recovered_tokens_pytorch_ngram(self):
        """Test recovered token sampling under n-gram mode"""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 2])
        draft_probs = None
        target_probs = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.3, 0.3, 0.4],
            ]
        )
        q = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.4, 0.1],
            ]
        )
        vocab_size = 3
        target_indices = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
            ]
        )
        enable_reduce_sampling = True
        sample_recovered_tokens_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=True,
            target_indices=target_indices,
            enable_reduce_sampling=enable_reduce_sampling,
        )

        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 1

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_reduce_sample_block_verify_pytorch(self):
        """Test random rejection sampling for block verify: accept based on uniform probability"""
        batch_size = 2
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 1])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.0, 0.4, 0.0],
                [0.1, 0.2, 0.3, 0.4, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.8, 0.0, 0.2],
                [0.2, 0.1, 0.3, 0.4],
                [0.9, 0.1, 0.0, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([1, 2, 3])
        uniform_probs = torch.tensor([0.7, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 5
        target_indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ]
        )
        enable_reduce_sampling = True
        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            target_indices=target_indices,
            enable_reduce_sampling=enable_reduce_sampling,
        )

        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 0
        assert output_token_ids[0, 2].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_reduce_sample_recovered_tokens_blockwise_pytorch_ngram(self):
        """Test recovered token sampling for blockwise speculative decoding with n-gram."""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 2])
        draft_probs = None
        target_probs = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.3, 0.3, 0.4],
            ]
        )
        q = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.5, 0.4, 0.1],
            ]
        )
        vocab_size = 3
        target_indices = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
            ]
        )
        enable_reduce_sampling = True
        sample_recovered_tokens_blockwise_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=True,
            target_indices=target_indices,
            enable_reduce_sampling=enable_reduce_sampling,
        )

        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 1

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_reduce_sample_recovered_tokens_blockwise_pytorch(self):
        """Test recovered token sampling for blockwise speculative decoding."""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = torch.tensor(
            [
                [0.6, 0.1, 0.3],
                [0.2, 0.7, 0.1],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.3, 0.6, 0.1],
            ]
        )
        q = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.1, 0.8, 0.1],
            ]
        )
        vocab_size = 3
        target_indices = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
            ]
        )
        enable_reduce_sampling = True
        sample_recovered_tokens_blockwise_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=False,
            target_indices=target_indices,
            enable_reduce_sampling=enable_reduce_sampling,
        )
        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 0

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_block_verify_pytorch_standard(self):
        """Test block verify without reduce_sampling: standard full-vocab path."""
        batch_size = 2
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 3])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.0, 0.4],
                [0.2, 0.0, 0.3, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.8, 0.0, 0.2],
                [0.1, 0.0, 0.3, 0.6],
                [0.0, 0.0, 0.9, 0.1],
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([99, 88, 77])
        uniform_probs = torch.tensor([0.7, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 4

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 0
        assert output_token_ids[0, 2].item() == 100
        assert output_token_ids[1, 0].item() == 2
        assert output_token_ids[1, 1].item() == 200


class TestEntropyVerify(TestBase):
    """Test ENTROPY_VERIFY mode in rejection sampling.

    Entropy verify modifies the acceptance threshold based on the entropy
    of the original target distribution:
    - High entropy (uncertain) → lower effective threshold → more accepting
    - Low entropy (certain)   → higher effective threshold → stricter
    """

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_standard_high_entropy_accepts_more(self):
        """High entropy (nearly uniform) lowers the effective threshold, causing
        acceptance of a draft token that would be rejected without entropy verify.

        Setup:
        - ratio = target_prob / draft_prob = 0.4 / 0.5 = 0.8
        - uniform = 0.85
        - Without entropy verify: 0.8 < 0.85 -> REJECT
        - High-entropy ori_target_probs = [0.4, 0.35, 0.25], entropy ~= 1.08
        - threshold = min(exp(-1.08 * 0.4), 0.95) ~= 0.649
        - modified_uniform = 0.649 * 0.85 ~= 0.552
        - With entropy verify: 0.8 >= 0.552 -> ACCEPT
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1])
        draft_token_ids = torch.tensor([0])
        draft_probs = torch.tensor([[0.5, 0.3, 0.2]])
        target_probs = torch.tensor([[0.4, 0.35, 0.25]])
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99])
        uniform_probs = torch.tensor([0.85])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        # High-entropy distribution (nearly uniform) -> low threshold
        ori_target_probs = torch.tensor([[0.4, 0.35, 0.25]])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # Draft token accepted, bonus token placed at position 1
        assert output_token_ids[0, 0].item() == 0
        assert output_token_ids[0, 1].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_standard_low_entropy_stricter(self):
        """Low entropy (peaked distribution) keeps threshold at POSTERIOR_THRESHOLD,
        causing rejection of the same draft token that high-entropy would accept.

        Uses the same ratio/uniform as test_entropy_verify_standard_high_entropy_accepts_more
        to isolate the entropy effect:
        - ratio = 0.4 / 0.5 = 0.8, uniform = 0.85
        - Low-entropy ori_target_probs = [0.98, 0.01, 0.01], entropy ~= 0.112
        - threshold = min(exp(-0.112 * 0.4), 0.95) = min(0.956, 0.95) = 0.95
        - modified_uniform = 0.95 * 0.85 ~= 0.808
        - With entropy verify: 0.8 < 0.808 -> REJECT (stricter than high-entropy case)
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1])
        draft_token_ids = torch.tensor([0])
        draft_probs = torch.tensor([[0.5, 0.3, 0.2]])
        target_probs = torch.tensor([[0.4, 0.35, 0.25]])
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99])
        uniform_probs = torch.tensor([0.85])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        # Low-entropy distribution (peaked at 0.98) -> threshold capped at POSTERIOR_THRESHOLD
        ori_target_probs = torch.tensor([[0.98, 0.01, 0.01]])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # Draft token rejected, recovered token placed at position 0
        assert output_token_ids[0, 0].item() == 99
        assert output_token_ids[0, 1].item() == PLACEHOLDER_TOKEN_ID
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_block_verify(self):
        """Entropy verify with block verify mode.

        Verifies cumulative-product acceptance with entropy-adjusted threshold
        across two requests. All ori_target_probs rows are valid distributions
        summing to 1.0. cu_num_draft_tokens is cumulative: req0 ends at index 2,
        req1 ends at index 3 (1 draft token).
        """
        batch_size = 2
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        # Cumulative: req0 has 2 tokens (ends at 2), req1 has 1 token (ends at 3)
        cu_num_draft_tokens = torch.tensor([2, 3])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.6, 0.4, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([99, 88, 77])
        uniform_probs = torch.tensor([0.7, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 4

        # Fixed: all rows are valid probability distributions summing to 1.0.
        # Previously row 1 was [0.88, 0.9, 0.01, 0.01] which summed to 1.8.
        ori_target_probs = torch.tensor(
            [
                [0.8, 0.18, 0.01, 0.01],
                [0.08, 0.9, 0.01, 0.01],
                [0.9, 0.08, 0.01, 0.01],
            ]
        )

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # Req0: both draft tokens rejected (pi < modified_cum_uniform), recovered at pos 0
        assert output_token_ids[0, 0].item() == 99
        assert output_token_ids[0, 1].item() == PLACEHOLDER_TOKEN_ID
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID
        # Req1: draft token has zero draft_prob (0.0), rejected, recovered at pos 0
        assert output_token_ids[1, 0].item() == 77
        assert output_token_ids[1, 1].item() == PLACEHOLDER_TOKEN_ID
        assert output_token_ids[1, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_ngram(self):
        """ENTROPY_VERIFY with IS_NGRAM: draft_probs=None, draft_token_probs=1.0.

        In NGRAM mode, acceptance depends on target_prob alone (since
        draft_prob=1.0). Entropy verify lowers the threshold for high-entropy
        tokens, making acceptance easier when the target distribution is
        uncertain.
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = None
        target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.2],
                [0.1, 0.1, 0.8],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        ori_target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.2],
                [0.1, 0.1, 0.8],
            ]
        )

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=True,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        assert output_token_ids[0, 0].item() == 0
        assert output_token_ids[0, 1].item() == 88

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_block_verify_ngram(self):
        """ENTROPY_VERIFY + IS_NGRAM + block_verify combined.

        Tests the interaction of all three modes: NGRAM (draft_probs=None),
        block verify (cumulative acceptance), and entropy-based threshold
        adjustment.
        """
        batch_size = 1
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = None
        target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.2, 0.0],
                [0.1, 0.1, 0.8, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 4

        ori_target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.2, 0.0],
                [0.1, 0.1, 0.8, 0.0],
            ]
        )

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=True,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        assert output_token_ids[0, 0].item() == 0
        assert output_token_ids[0, 1].item() == 88

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_no_ori_probs_fallback(self):
        """When ori_target_probs is None, fallback to target_probs for entropy.

        Both target_probs rows are high-entropy (~1.098), so the threshold is
        lowered and both draft tokens are accepted:
        - Token 0: ratio=0.33/0.4=0.825 >= modified_uniform~=0.451 -> ACCEPT
        - Token 1: ratio=0.34/0.8=0.425 >= modified_uniform~=0.387 -> ACCEPT
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.6, 0.4, 0.0],
                [0.8, 0.2, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.35, 0.33, 0.32],
                [0.34, 0.34, 0.32],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=None,
        )

        # Both draft tokens accepted (high entropy lowers threshold)
        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 0
        assert output_token_ids[0, 2].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_with_reduce_sample(self):
        """ENTROPY_VERIFY + enable_reduce_sampling combined (non-block-verify).

        Reduce sampling stores target_probs over a selected candidate set
        (target_indices maps candidates back to global vocab). The draft token's
        target probability is found by searching the candidate list.

        Setup:
        - draft_token=0, draft_prob=0.5
        - Candidates=[0, 1], target_probs=[0.4, 0.35] (selected vocab)
        - ori_target_probs=[0.4, 0.35, 0.25] (full dist for entropy)
        - ratio=0.4/0.5=0.8, uniform=0.85
        - Without entropy verify: 0.8 < 0.85 -> REJECT
        - High entropy ~= 1.08, threshold ~= 0.649
        - modified_uniform = 0.649 * 0.85 ~= 0.552
        - With entropy verify: 0.8 >= 0.552 -> ACCEPT
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1])
        draft_token_ids = torch.tensor([0])
        draft_probs = torch.tensor([[0.5, 0.3, 0.2]])
        # Reduced candidate set: only tokens 0 and 1 are candidates
        target_indices = torch.tensor([[0, 1]])
        target_probs = torch.tensor([[0.4, 0.35]])
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99])
        uniform_probs = torch.tensor([0.85])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        # Full distribution for entropy computation
        ori_target_probs = torch.tensor([[0.4, 0.35, 0.25]])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            target_indices=target_indices,
            enable_reduce_sampling=True,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # Draft accepted due to entropy-lowered threshold, bonus at pos 1
        assert output_token_ids[0, 0].item() == 0
        assert output_token_ids[0, 1].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_with_reduce_sample_block_verify(self):
        """ENTROPY_VERIFY + enable_reduce_sampling + block_verify combined.

        Three-way combination: cumulative-product acceptance (block verify),
        candidate search (reduce sample), and entropy-adjusted threshold.
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.4, 0.4, 0.2],
            ]
        )
        # Reduced candidate sets per token
        target_indices = torch.tensor(
            [
                [0, 1],
                [0, 1],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.4, 0.35],
                [0.3, 0.45],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.85, 0.8])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        # Full distributions for entropy (high entropy -> low threshold)
        ori_target_probs = torch.tensor(
            [
                [0.4, 0.35, 0.25],
                [0.3, 0.45, 0.25],
            ]
        )

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            target_indices=target_indices,
            enable_reduce_sampling=True,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # Token 0: draft=0, target_prob=0.4 (candidate 0), pi=0.4/0.5=0.8
        #   entropy~=1.08, threshold~=0.649, cum_uniform=0.85
        #   modified_cum_uniform=0.649*0.85~=0.552, 0.8>=0.552 -> legal
        # Token 1: draft=1, target_prob=0.45 (candidate 1), pi=0.45/0.4=1.0 (clamped)
        #   cum_pi=0.8*1.0=0.8, entropy~=1.07, threshold~=0.653
        #   cum_uniform=0.85*0.8=0.68, modified_cum_uniform=0.653*0.68~=0.444
        #   0.8>=0.444 -> legal
        # Both accepted -> bonus at pos 2
        assert output_token_ids[0, 0].item() == 0
        assert output_token_ids[0, 1].item() == 1
        assert output_token_ids[0, 2].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_comparison_with_without(self):
        """Compare entropy verify ON vs OFF on the same borderline input.

        Uses a ratio that falls between the entropy-modified threshold and the
        raw uniform, so entropy verify flips rejection to acceptance.
        """
        # Same setup for both runs: ratio=0.8, uniform=0.85
        # - Without entropy verify: 0.8 < 0.85 -> REJECT
        # - With entropy verify (high entropy): 0.8 >= 0.552 -> ACCEPT

        def run(entropy_verify: bool):
            output_token_ids = torch.full((1, 3), PLACEHOLDER_TOKEN_ID)
            rejection_random_sample_pytorch(
                output_token_ids,
                cu_num_draft_tokens=torch.tensor([1]),
                draft_token_ids=torch.tensor([0]),
                draft_probs=torch.tensor([[0.5, 0.3, 0.2]]),
                target_probs=torch.tensor([[0.4, 0.35, 0.25]]),
                bonus_token_ids=torch.tensor([[100]]),
                recovered_token_ids=torch.tensor([99]),
                uniform_probs=torch.tensor([0.85]),
                is_greedy=torch.tensor([False]),
                max_spec_len=2,
                vocab_size=3,
                IS_NGRAM=False,
                ENTROPY_VERIFY=entropy_verify,
                POSTERIOR_THRESHOLD=0.95,
                POSTERIOR_ALPHA=0.4,
                EPSILON=1e-10,
                ori_target_probs=torch.tensor([[0.4, 0.35, 0.25]]),
            )
            return output_token_ids

        out_off = run(entropy_verify=False)
        out_on = run(entropy_verify=True)

        # Without entropy verify: rejected -> recovered token at pos 0
        assert out_off[0, 0].item() == 99
        assert out_off[0, 1].item() == PLACEHOLDER_TOKEN_ID
        # With entropy verify: accepted -> draft token at pos 0, bonus at pos 1
        assert out_on[0, 0].item() == 0
        assert out_on[0, 1].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_boundary_zero_entropy(self):
        """Boundary: near-zero entropy (one-hot distribution).

        With a one-hot ori_target_probs, entropy -> 0, exp(-0)=1.0,
        threshold = min(1.0, POSTERIOR_THRESHOLD) = POSTERIOR_THRESHOLD.
        The effective threshold equals the raw uniform * POSTERIOR_THRESHOLD,
        which is the strictest setting.
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1])
        draft_token_ids = torch.tensor([0])
        draft_probs = torch.tensor([[0.5, 0.3, 0.2]])
        target_probs = torch.tensor([[0.4, 0.35, 0.25]])
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99])
        uniform_probs = torch.tensor([0.85])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        # Near one-hot distribution -> entropy -> 0
        ori_target_probs = torch.tensor([[1.0, 0.0, 0.0]])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # threshold = min(exp(0), 0.95) = 0.95
        # modified_uniform = 0.95 * 0.85 = 0.8075
        # ratio = 0.8 < 0.8075 -> REJECT
        assert output_token_ids[0, 0].item() == 99
        assert output_token_ids[0, 1].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_boundary_alpha_zero(self):
        """Boundary: POSTERIOR_ALPHA=0 disables entropy adjustment.

        exp(-entropy * 0) = exp(0) = 1.0, so threshold = min(1.0, POSTERIOR_THRESHOLD)
        = POSTERIOR_THRESHOLD for all tokens regardless of entropy.
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1])
        draft_token_ids = torch.tensor([0])
        draft_probs = torch.tensor([[0.5, 0.3, 0.2]])
        target_probs = torch.tensor([[0.4, 0.35, 0.25]])
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99])
        uniform_probs = torch.tensor([0.85])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        ori_target_probs = torch.tensor([[0.4, 0.35, 0.25]])

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.0,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # alpha=0 -> threshold = min(1.0, 0.95) = 0.95 for all tokens
        # modified_uniform = 0.95 * 0.85 = 0.8075
        # ratio = 0.8 < 0.8075 -> REJECT (same as zero-entropy boundary)
        assert output_token_ids[0, 0].item() == 99
        assert output_token_ids[0, 1].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_boundary_all_accepted(self):
        """Boundary: all draft tokens accepted when ratio dominates uniform.

        High target/draft ratio with low uniform ensures acceptance regardless
        of entropy. Verifies the bonus token is appended after all accepted.
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.4, 0.4, 0.2],
            ]
        )
        # target_prob >> draft_prob -> ratio = 1.0 (clamped)
        target_probs = torch.tensor(
            [
                [0.9, 0.05, 0.05],
                [0.8, 0.1, 0.1],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.1, 0.1])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        # Moderate-low entropy distributions for entropy computation
        ori_target_probs = torch.tensor(
            [
                [0.9, 0.05, 0.05],
                [0.8, 0.1, 0.1],
            ]
        )

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # Token 0: ratio=0.9/0.5=1.8, threshold~=0.854, modified_uniform=0.854*0.1~=0.085 -> ACCEPT
        # Token 1: ratio=0.1/0.4=0.25, threshold~=0.775, modified_uniform=0.775*0.1~=0.078 -> ACCEPT
        assert output_token_ids[0, 0].item() == 0
        assert output_token_ids[0, 1].item() == 1
        assert output_token_ids[0, 2].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_boundary_all_rejected(self):
        """Boundary: all draft tokens rejected when ratio is tiny.

        target_prob near 0 with non-trivial uniform ensures rejection regardless
        of entropy. Verifies recovered token at pos 0 and placeholders after.
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.4, 0.4, 0.2],
            ]
        )
        # target_prob near 0 -> ratio near 0 -> always rejected
        target_probs = torch.tensor(
            [
                [0.01, 0.01, 0.98],
                [0.01, 0.01, 0.98],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.1, 0.1])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        ori_target_probs = torch.tensor(
            [
                [0.4, 0.35, 0.25],
                [0.4, 0.35, 0.25],
            ]
        )

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # Token 0: ratio=0.01/0.5=0.02, modified_uniform=0.649*0.1=0.065 -> REJECT
        assert output_token_ids[0, 0].item() == 99
        assert output_token_ids[0, 1].item() == PLACEHOLDER_TOKEN_ID
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID
