from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler

from tests.ut.base import TestBase
from vllm_ascend.sample.rejection_sampler import (
    AscendRejectionSampler,
    apply_sampling_constraints,
    expand_batch_to_tokens,
    expand_pytorch,
    greedy_sample,
    rejection_greedy_sample_pytorch,
    rejection_greedy_sample_spec_len_1_pytorch,
    rejection_random_sample_block_verify_pytorch,
    rejection_random_sample_pytorch,
    rejection_sample,
    sample_recovered_tokens,
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
        """High entropy (uniform-like) makes acceptance easier via lower threshold."""
        batch_size = 2
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 1])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.6, 0.4, 0.0],
                [0.2, 0.8, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.8, 0.2, 0.0],
                [0.1, 0.9, 0.0],
                [0.9, 0.1, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([99, 88, 77])
        uniform_probs = torch.tensor([0.7, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 3

        ori_target_probs = torch.tensor(
            [
                [0.8, 0.19, 0.01],
                [0.09, 0.9, 0.01],
                [0.9, 0.09, 0.01],
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

        assert output_token_ids[0, 0].item() == 99
        assert output_token_ids[0, 1].item() == -1
        assert output_token_ids[0, 2].item() == -1

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_standard_low_entropy_stricter(self):
        """Low entropy (peaked distribution) keeps threshold near POSTERIOR_THRESHOLD."""
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
                [0.8, 0.2, 0.0],
                [0.1, 0.9, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        ori_target_probs = torch.tensor(
            [
                [0.8, 0.19, 0.01],
                [0.09, 0.9, 0.01],
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

        assert output_token_ids[0, 0].item() == 99
        assert output_token_ids[0, 1].item() == -1
        assert output_token_ids[0, 2].item() == -1

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_block_verify(self):
        """Entropy verify with block verify mode."""
        batch_size = 2
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 1])
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

        ori_target_probs = torch.tensor(
            [
                [0.8, 0.18, 0.01, 0.01],
                [0.88, 0.9, 0.01, 0.01],
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

        assert output_token_ids[0, 0].item() == 99
        assert output_token_ids[0, 1].item() == -1
        assert output_token_ids[0, 2].item() == -1

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
        """When ori_target_probs is None, fallback to target_probs for entropy."""
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

        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() in (0, 88)
        assert output_token_ids[0, 2].item() == 100


def _pin_memory_stack() -> ExitStack:
    stack = ExitStack()
    stack.enter_context(patch("torch.arange", new=mock_pin_memory(torch.arange)))
    stack.enter_context(patch("torch.ones", new=mock_pin_memory(torch.ones)))
    stack.enter_context(patch("torch.full", new=mock_pin_memory(torch.full)))
    stack.enter_context(patch("torch.tensor", new=mock_pin_memory(torch.tensor)))
    return stack


def _ascend_cfg(reduce_sample=False, block_verify=False, entropy_verify=False):
    cfg = MagicMock()
    cfg.enable_reduce_sample = reduce_sample
    cfg.rejection_sampler_config.enable_block_verify = block_verify
    cfg.rejection_sampler_config.enable_entropy_verify = entropy_verify
    cfg.rejection_sampler_config.posterior_threshold = 0.95
    cfg.rejection_sampler_config.posterior_alpha = 0.4
    return cfg


def _tp_group():
    group = MagicMock()
    group.rank_in_group = 0
    group.all_gather.side_effect = lambda tensor, dim=-1: tensor
    return group


def _replace(obj, **kwargs):
    data = dict(vars(obj))
    data.update(kwargs)
    return SimpleNamespace(**data)


def test_ascend_rejection_sampler_methods():
    logits = torch.tensor([[0.1, 0.8, 0.1]])
    with (
        patch.object(RejectionSampler, "__init__", lambda self, *args, **kwargs: None),
        patch("vllm_ascend.sample.rejection_sampler.get_ascend_config", return_value=_ascend_cfg()),
    ):
        sampler = AscendRejectionSampler(MagicMock(), None, None)
    sampler.prepare_sampling(4)
    assert sampler.top_k == 4
    sampler.prepare_sampling(None)
    assert sampler.top_k is None

    assert AscendRejectionSampler.apply_penalties(logits, SimpleNamespace(no_penalties=True), None, None, []) is logits
    with (
        patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
        patch.object(Sampler, "apply_penalties", return_value=logits) as fallback,
    ):
        assert AscendRejectionSampler.apply_penalties(
            logits, SimpleNamespace(no_penalties=False), None, None, []
        ) is logits
        fallback.assert_called_once()
    with (
        patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", True),
        patch("vllm_ascend.sample.rejection_sampler.apply_all_penalties", return_value=logits) as apply_penalties,
    ):
        meta = SimpleNamespace(
            no_penalties=False,
            prompt_token_ids=torch.tensor([[1, 2]]),
            presence_penalties=torch.zeros(1),
            frequency_penalties=torch.zeros(1),
            repetition_penalties=torch.ones(1),
        )
        assert AscendRejectionSampler.apply_penalties(logits, meta, None, torch.tensor([0]), [[3]]) is logits
        apply_penalties.assert_called_once()

    rs = AscendRejectionSampler.__new__(AscendRejectionSampler)
    rs._combine_outputs_with_spec_tokens = lambda output_token_ids, spec_token_ids: output_token_ids
    min_tokens = MagicMock(spec=MinTokensLogitsProcessor)
    min_tokens.apply_with_spec_decode.side_effect = lambda values, num_draft: values
    holder = MagicMock()
    holder.has_tracked_requests.return_value = True
    holder.apply_to_logits.side_effect = lambda values, **kwargs: values
    sampling_metadata = SimpleNamespace(
        no_penalties=False,
        bad_words_token_ids={0: [[2]]},
        output_token_ids=[[1]],
        spec_token_ids=[[2]],
        allowed_token_ids_mask=torch.zeros(1, 3, dtype=torch.bool),
        prompt_token_ids=torch.tensor([[1, 2]]),
        presence_penalties=torch.zeros(1),
        frequency_penalties=torch.zeros(1),
        repetition_penalties=torch.ones(1),
        logitsprocs=SimpleNamespace(non_argmax_invariant=[min_tokens]),
        thinking_budget_state_holder=holder,
    )
    spec_metadata = SimpleNamespace(num_draft_tokens=[1], cu_num_draft_tokens=torch.tensor([1]))
    with (
        patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
        patch.object(Sampler, "apply_penalties", return_value=logits),
        patch("vllm_ascend.sample.rejection_sampler.apply_bad_words_with_drafts"),
        patch("vllm_ascend.sample.rejection_sampler.expand_batch_to_tokens", return_value=torch.tensor([0])),
    ):
        processed = rs.apply_logits_processors(logits.clone(), sampling_metadata, spec_metadata)
    assert processed.shape == logits.shape
    min_tokens.apply_with_spec_decode.assert_called_once()
    holder.apply_to_logits.assert_called_once()

    rs.sampler = MagicMock()
    rs.sampler.logprobs_mode = "raw_logprobs"
    rs.sampler.return_value = SimpleNamespace(
        sampled_token_ids=torch.tensor([[9]], dtype=torch.int32),
        logprobs_tensors=SimpleNamespace(logprobs=torch.zeros(1, 3)),
    )
    rs.top_k = None
    rs.synthetic_mode = False
    rs.synthetic_conditional_rates = None
    rs.is_processed_logprobs_mode = False
    rs.apply_logits_processors = lambda values, *args, **kwargs: values
    rs._get_logprobs_tensors = MagicMock(return_value="lp")
    forward_logits = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
    forward_meta = SimpleNamespace(
        max_spec_len=1,
        bonus_logits_indices=torch.tensor([1]),
        target_logits_indices=torch.tensor([0]),
        draft_token_ids=torch.tensor([1]),
        num_draft_tokens=[1],
        cu_num_draft_tokens=torch.tensor([1]),
    )
    forward_sampling = SimpleNamespace(max_num_logprobs=1)
    with (
        patch("vllm_ascend.sample.rejection_sampler.replace", _replace),
        patch(
            "vllm_ascend.sample.rejection_sampler.apply_sampling_constraints",
            return_value=(forward_logits[:1], None),
        ),
        patch(
            "vllm_ascend.sample.rejection_sampler.rejection_sample",
            return_value=torch.tensor([[1, 9]], dtype=torch.int32),
        ),
    ):
        output = rs.forward(forward_meta, None, forward_logits.clone(), forward_sampling)
    assert output.logprobs_tensors == "lp"

    rs.is_processed_logprobs_mode = True
    with (
        patch("vllm_ascend.sample.rejection_sampler.replace", _replace),
        patch(
            "vllm_ascend.sample.rejection_sampler.apply_sampling_constraints",
            return_value=(forward_logits[:1], None),
        ),
        patch(
            "vllm_ascend.sample.rejection_sampler.rejection_sample",
            return_value=torch.tensor([[1, 9]], dtype=torch.int32),
        ),
    ):
        processed_output = rs.forward(
            forward_meta, None, forward_logits.clone(), SimpleNamespace(max_num_logprobs=None)
        )
    assert processed_output.logprobs_tensors is None

    debug_meta = SimpleNamespace(
        max_spec_len=1,
        num_draft_tokens=[1],
        cu_num_draft_tokens=torch.tensor([1]),
    )
    with patch("vllm_ascend.sample.rejection_sampler.logger") as logger:
        logger.isEnabledFor.return_value = True
        rs._log_rejection_sampler_entry(forward_logits, debug_meta)
        debug_meta.num_draft_tokens = torch.tensor([1])
        rs._log_rejection_sampler_exit(torch.tensor([[1, PLACEHOLDER_TOKEN_ID]]), debug_meta)
        logger.debug.assert_called()


def test_rejection_sample_constraint_and_helper_paths():
    with _pin_memory_stack():
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        greedy_meta = SimpleNamespace(
            all_greedy=True, all_random=False, temperature=torch.tensor([0.0]), generators={}
        )
        random_meta = SimpleNamespace(
            all_greedy=False,
            all_random=True,
            temperature=torch.tensor([1.0]),
            generators={0: torch.Generator().manual_seed(0)},
        )
        mixed_meta = SimpleNamespace(
            all_greedy=False,
            all_random=False,
            temperature=torch.tensor([GREEDY_TEMPERATURE]),
            generators={},
        )
        uniform = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        indices = torch.tensor([[0, 1], [0, 1]])
        draft_probs = torch.tensor([[0.2, 0.8, 0.0], [0.7, 0.3, 0.0]])
        local_draft_probs = draft_probs[:, :2].contiguous()

        def run_sample(**kwargs):
            defaults = dict(
                draft_token_ids=torch.tensor([1, 0], dtype=torch.int32),
                num_draft_tokens=[2],
                max_spec_len=2,
                cu_num_draft_tokens=torch.tensor([2]),
                draft_probs=None,
                target_logits_or_tuple=logits.clone(),
                bonus_token_ids=torch.tensor([[5]], dtype=torch.int32),
                sampling_metadata=greedy_meta,
                synthetic_mode=False,
                synthetic_conditional_rates=None,
                ori_target_logits=None,
            )
            defaults.update(kwargs)
            return rejection_sample(**defaults)

        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
            patch("vllm_ascend.sample.rejection_sampler.generate_uniform_probs", return_value=uniform[:2]),
            patch("vllm_ascend.sample.rejection_sampler.get_tp_group", return_value=_tp_group()),
            patch("vllm_ascend.sample.rejection_sampler.get_ascend_config", return_value=_ascend_cfg()),
        ):
            spec1 = run_sample(
                draft_token_ids=torch.tensor([1], dtype=torch.int32),
                num_draft_tokens=[1],
                max_spec_len=1,
                cu_num_draft_tokens=torch.tensor([1]),
                target_logits_or_tuple=torch.tensor([[0.1, 0.9]]),
            )
            assert spec1.shape == (1, 2)
            empty = run_sample(
                draft_token_ids=torch.tensor([], dtype=torch.int32),
                num_draft_tokens=[0],
                max_spec_len=0,
                cu_num_draft_tokens=torch.tensor([0]),
                target_logits_or_tuple=torch.empty(0, 2),
            )
            assert empty.shape == (1, 1)
            greedy = run_sample()
            assert greedy.shape == (1, 3)

        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
            patch("vllm_ascend.sample.rejection_sampler.generate_uniform_probs", return_value=uniform[:2]),
            patch("vllm_ascend.sample.rejection_sampler.get_tp_group", return_value=_tp_group()),
            patch(
                "vllm_ascend.sample.rejection_sampler.get_ascend_config",
                return_value=_ascend_cfg(reduce_sample=True),
            ),
        ):
            assert run_sample().shape[0] == 1
            assert run_sample(
                sampling_metadata=mixed_meta,
                target_logits_or_tuple=(logits.clone(), indices.clone()),
                draft_probs=draft_probs.clone(),
            ).shape[0] == 1

        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
            patch("vllm_ascend.sample.rejection_sampler.generate_uniform_probs", return_value=uniform[:2]),
            patch(
                "vllm_ascend.sample.rejection_sampler.get_ascend_config",
                return_value=_ascend_cfg(entropy_verify=True),
            ),
        ):
            assert run_sample(
                sampling_metadata=random_meta,
                target_logits_or_tuple=(logits.clone(), indices.clone()),
                draft_probs=draft_probs.clone(),
                ori_target_logits=logits.clone(),
            ).shape[0] == 1
            assert run_sample(
                sampling_metadata=random_meta,
                target_logits_or_tuple=logits.clone(),
                draft_probs=local_draft_probs.clone(),
            ).shape[0] == 1

        block_logits = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        block_ids = torch.tensor([1, 0, 1], dtype=torch.int32)
        block_probs = torch.tensor([[0.2, 0.8, 0.0], [0.6, 0.4, 0.0], [0.3, 0.7, 0.0]])
        local_block_probs = block_probs[:, :2].contiguous()
        block_idx = torch.tensor([[0, 1], [0, 1], [0, 1]])
        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
            patch("vllm_ascend.sample.rejection_sampler.generate_uniform_probs", return_value=uniform),
            patch(
                "vllm_ascend.sample.rejection_sampler.get_ascend_config",
                return_value=_ascend_cfg(block_verify=True, entropy_verify=True),
            ),
        ):
            assert run_sample(
                draft_token_ids=block_ids,
                num_draft_tokens=[3],
                max_spec_len=3,
                cu_num_draft_tokens=torch.tensor([3]),
                sampling_metadata=random_meta,
                target_logits_or_tuple=(block_logits.clone(), block_idx.clone()),
                draft_probs=block_probs.clone(),
                ori_target_logits=block_logits.clone(),
            ).shape == (1, 4)
            assert run_sample(
                draft_token_ids=block_ids,
                num_draft_tokens=[3],
                max_spec_len=3,
                cu_num_draft_tokens=torch.tensor([3]),
                sampling_metadata=random_meta,
                target_logits_or_tuple=block_logits.clone(),
                draft_probs=local_block_probs.clone(),
            ).shape == (1, 4)
            with pytest.raises(ValueError, match="synthetic_mode"):
                run_sample(
                    draft_token_ids=block_ids,
                    num_draft_tokens=[3],
                    max_spec_len=3,
                    cu_num_draft_tokens=torch.tensor([3]),
                    sampling_metadata=random_meta,
                    synthetic_mode=True,
                    synthetic_conditional_rates=torch.tensor([0.5, 0.5, 0.5]),
                )

        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
            patch("vllm_ascend.sample.rejection_sampler.generate_uniform_probs", return_value=uniform[:1]),
            patch("vllm_ascend.sample.rejection_sampler.get_ascend_config", return_value=_ascend_cfg()),
        ):
            synthetic = run_sample(
                draft_token_ids=torch.tensor([1], dtype=torch.int32),
                num_draft_tokens=[1],
                max_spec_len=1,
                cu_num_draft_tokens=torch.tensor([1]),
                target_logits_or_tuple=torch.tensor([[0.1, 0.9]]),
                synthetic_mode=True,
                synthetic_conditional_rates=torch.tensor([0.9]),
            )
            assert synthetic.shape == (1, 2)

        constraint_logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
        cu = torch.tensor([2])
        assert apply_sampling_constraints(constraint_logits, cu, SimpleNamespace(all_greedy=True), None)[1] is None
        non_greedy = SimpleNamespace(
            all_greedy=False,
            temperature=torch.tensor([1.0]),
            top_k=torch.tensor([2]),
            top_p=torch.tensor([0.9]),
        )
        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
            patch(
                "vllm_ascend.sample.rejection_sampler.apply_top_k_top_p",
                return_value=(constraint_logits, indices),
            ) as apply_fn,
            patch(
                "vllm_ascend.sample.rejection_sampler.get_ascend_config",
                return_value=_ascend_cfg(reduce_sample=True),
            ),
        ):
            apply_sampling_constraints(constraint_logits.clone(), cu, non_greedy, 2)
            apply_fn.assert_called()
        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
            patch("vllm_ascend.sample.rejection_sampler.apply_top_k_top_p", return_value=constraint_logits),
            patch("vllm_ascend.sample.rejection_sampler.get_ascend_config", return_value=_ascend_cfg()),
        ):
            apply_sampling_constraints(constraint_logits.clone(), cu, non_greedy, None)

        with patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False):
            recovered = sample_recovered_tokens(
                1,
                [1],
                torch.tensor([1]),
                torch.tensor([0]),
                None,
                torch.tensor([[0.2, 0.8]]),
                random_meta,
                torch.device("cpu"),
            )
            assert recovered.numel() == 1
            blocked = sample_recovered_tokens(
                1,
                [1],
                torch.tensor([1]),
                torch.tensor([0]),
                None,
                torch.tensor([[0.2, 0.8]]),
                SimpleNamespace(generators={}),
                torch.device("cpu"),
                use_block_verify=True,
            )
            assert blocked.numel() == 1

            spec_out = torch.full((1, 2), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
            rejection_greedy_sample_spec_len_1_pytorch(
                spec_out,
                torch.tensor([1]),
                torch.tensor([0]),
                torch.tensor([[9]]),
                uniform_probs=torch.tensor([0.1]),
                synthetic_conditional_rates=torch.tensor([0.9]),
                synthetic_mode=True,
            )
            assert spec_out[0, 0].item() == 1

            syn_out = torch.full((1, 3), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
            rejection_greedy_sample_pytorch(
                syn_out,
                torch.tensor([2]),
                torch.tensor([1, 0]),
                torch.tensor([1, 1]),
                torch.tensor([[9]]),
                [2],
                2,
                uniform_probs=torch.tensor([0.1, 0.9]),
                synthetic_conditional_rates=torch.tensor([0.5, 0.5]),
                synthetic_mode=True,
            )
            assert syn_out.shape == (1, 3)

            expand_pytorch(torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.long), 0, 0, MAX_SPEC_LEN)
            sample_recovered_tokens_pytorch(
                torch.empty(0, dtype=torch.int32),
                torch.tensor([0]),
                torch.empty(0, dtype=torch.int32),
                None,
                torch.empty(0, 2),
                torch.empty(1, 2),
                2,
            )
            sample_recovered_tokens_blockwise_pytorch(
                torch.empty(0, dtype=torch.int32),
                torch.tensor([0]),
                torch.empty(0, dtype=torch.int32),
                None,
                torch.empty(0, 2),
                torch.empty(1, 2),
                2,
            )

            recovered_out = torch.empty(2, dtype=torch.int32)
            sample_recovered_tokens_pytorch(
                recovered_out,
                torch.tensor([2]),
                torch.tensor([0, 1]),
                draft_probs.clone(),
                torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
                torch.ones((1, 2)),
                2,
                IS_NGRAM=False,
                target_indices=indices.clone(),
                enable_reduce_sampling=True,
            )
            sample_recovered_tokens_pytorch(
                recovered_out,
                torch.tensor([2]),
                torch.tensor([0, 1]),
                draft_probs[:, :2].clone(),
                torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
                torch.ones((1, 2)),
                2,
                IS_NGRAM=False,
            )
            sample_recovered_tokens_blockwise_pytorch(
                recovered_out,
                torch.tensor([2]),
                torch.tensor([0, 1]),
                draft_probs[:, :2].clone(),
                torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
                torch.ones((1, 2)),
                2,
                IS_NGRAM=False,
            )
            sample_recovered_tokens_blockwise_pytorch(
                recovered_out,
                torch.tensor([2]),
                torch.tensor([PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]),
                None,
                torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
                torch.ones((1, 2)),
                2,
                IS_NGRAM=True,
            )

            rand_out = torch.full((1, 3), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
            rejection_random_sample_pytorch(
                rand_out,
                torch.tensor([2]),
                torch.tensor([1, 0]),
                draft_probs.clone(),
                torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
                torch.tensor([[9]]),
                torch.tensor([0, 1]),
                torch.tensor([0.1, 0.2]),
                torch.tensor([False]),
                2,
                2,
                IS_NGRAM=False,
                target_indices=indices.clone(),
                enable_reduce_sampling=True,
                synthetic_mode=True,
                synthetic_conditional_rates=torch.tensor([0.9, 0.1]),
            )
            assert rand_out.shape == (1, 3)

    with patch("vllm_ascend.sample.rejection_sampler.get_tp_group", return_value=_tp_group()):
        assert greedy_sample(torch.tensor([[1.0, 3.0, 2.0]])).tolist() == [1]
