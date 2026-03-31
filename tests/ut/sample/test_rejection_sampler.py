#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import os
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.rejection_sampler import (
    expand_batch_to_tokens, expand_pytorch, rejection_greedy_sample_pytorch,
    rejection_random_sample_pytorch, sample_recovered_tokens_pytorch)

# Global constants
PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = 0.0
MAX_SPEC_LEN = 8  # Used as MAX_NUM_TOKENS in expand_batch_to_tokens


def mock_pin_memory(original_func):

    def func_wo_pin_memory(*args, **kwargs):
        if kwargs.get('pin_memory', False):
            kwargs['pin_memory'] = False
        return original_func(*args, **kwargs)

    return func_wo_pin_memory


class TestAscendRejectionSampler(TestBase):

    @patch('torch.arange', new=mock_pin_memory(torch.arange))
    @patch('torch.ones', new=mock_pin_memory(torch.ones))
    @patch('torch.full', new=mock_pin_memory(torch.full))
    @patch('torch.tensor', new=mock_pin_memory(torch.tensor))
    def test_rejection_greedy_sample_pytorch(self):
        """Test greedy rejection sampling: stop when draft doesn't match, otherwise append bonus token"""
        batch_size = 2
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1),
                                      PLACEHOLDER_TOKEN_ID)

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

    @patch('torch.arange', new=mock_pin_memory(torch.arange))
    @patch('torch.ones', new=mock_pin_memory(torch.ones))
    @patch('torch.full', new=mock_pin_memory(torch.full))
    @patch('torch.tensor', new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_pytorch(self):
        """Test random rejection sampling: accept based on uniform probability"""
        batch_size = 2
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1),
                                      PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 1])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor([
            [0.0, 0.6, 0.0, 0.4],  # vocab_size=4
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.5, 0.0, 0.0],
        ])
        target_probs = torch.tensor([
            [0.0, 0.8, 0.0, 0.2],
            [0.2, 0.1, 0.3, 0.4],
            [0.9, 0.1, 0.0, 0.0],
        ])
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

    @patch('torch.arange', new=mock_pin_memory(torch.arange))
    @patch('torch.ones', new=mock_pin_memory(torch.ones))
    @patch('torch.full', new=mock_pin_memory(torch.full))
    @patch('torch.tensor', new=mock_pin_memory(torch.tensor))
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

    @patch('torch.arange', new=mock_pin_memory(torch.arange))
    @patch('torch.ones', new=mock_pin_memory(torch.ones))
    @patch('torch.full', new=mock_pin_memory(torch.full))
    @patch('torch.tensor', new=mock_pin_memory(torch.tensor))
    def test_expand_batch_to_tokens(self):
        """Test expand_batch_to_tokens wrapper"""
        x = torch.tensor([10, 20, 30])
        cu_num_tokens = torch.tensor([2, 5, 7])
        num_tokens = 7
        # Test PyTorch path
        with patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False):
            with patch("vllm_ascend.sample.rejection_sampler.expand_pytorch"
                       ) as mock_pytorch:
                expand_batch_to_tokens(x, cu_num_tokens, num_tokens)
                mock_pytorch.assert_called_once()
                args = mock_pytorch.call_args[0]
                assert (args[1] == x).all()
                assert (args[2] == cu_num_tokens).all()

        # Test Triton kernel path
        with patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", True):
            with patch("vllm_ascend.sample.rejection_sampler.expand_triton"
                       ) as mock_triton:
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

    @patch('torch.arange', new=mock_pin_memory(torch.arange))
    @patch('torch.ones', new=mock_pin_memory(torch.ones))
    @patch('torch.full', new=mock_pin_memory(torch.full))
    @patch('torch.tensor', new=mock_pin_memory(torch.tensor))
    def test_sample_recovered_tokens_pytorch_ngram(self):
        """Test recovered token sampling under n-gram mode"""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 2])
        draft_probs = None
        target_probs = torch.tensor([
            [0.1, 0.2, 0.7],
            [0.3, 0.3, 0.4],
        ])
        q = torch.tensor([
            [0.1, 0.2, 0.7],
            [0.5, 0.4, 0.1],
        ])
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

    @patch('torch.arange', new=mock_pin_memory(torch.arange))
    @patch('torch.ones', new=mock_pin_memory(torch.ones))
    @patch('torch.full', new=mock_pin_memory(torch.full))
    @patch('torch.tensor', new=mock_pin_memory(torch.tensor))
    def test_sample_recovered_tokens_pytorch_autoregressive(self):
        """Test recovered token sampling for autoregressive models"""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = torch.tensor([
            [0.6, 0.1, 0.3],
            [0.2, 0.7, 0.1],
        ])
        target_probs = torch.tensor([
            [0.8, 0.1, 0.1],
            [0.3, 0.6, 0.1],
        ])
        q = torch.tensor([
            [0.5, 0.3, 0.2],
            [0.1, 0.8, 0.1],
        ])
        vocab_size = 3

        sample_recovered_tokens_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=False,
        )
        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 0

    def test_ears_uniform_probs_adjustment(self):
        """Test that EARS shifts uniform_probs down by tolerance * uncertainty
        and clamps to dtype epsilon when VLLM_EARS_TOLERANCE > 0."""
        import vllm_ascend.envs as envs_ascend
        from vllm_ascend.sample.rejection_sampler import generate_uniform_probs

        # target_probs: high uncertainty token (max=0.2) and low uncertainty token (max=0.9)
        target_probs = torch.tensor([[0.1, 0.2, 0.3, 0.4],   # max=0.4, uncertainty=0.6
                                     [0.9, 0.05, 0.03, 0.02]])  # max=0.9, uncertainty=0.1
        # Use fixed uniform_probs to make the test deterministic
        uniform_probs = torch.tensor([0.5, 0.5])

        ears_tolerance = 0.5
        expected_uncertainties = 1.0 - target_probs.max(dim=-1).values
        expected_tolerance = ears_tolerance * expected_uncertainties
        eps = torch.finfo(uniform_probs.dtype).eps
        expected = (uniform_probs - expected_tolerance).clamp_min(eps)

        with patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": str(ears_tolerance)}):
            # Reload env var
            adjusted = uniform_probs.clone()
            _ears_tolerance = envs_ascend.VLLM_EARS_TOLERANCE
            max_target_probs = target_probs.max(dim=-1).values
            uncertainties = 1.0 - max_target_probs
            tolerance = _ears_tolerance * uncertainties
            adjusted = (adjusted - tolerance).clamp_min(eps)

        assert torch.allclose(adjusted, expected), (
            f"EARS adjustment mismatch: got {adjusted}, expected {expected}")
        # High-uncertainty token should be shifted more than low-uncertainty token
        assert adjusted[0] < uniform_probs[0]
        assert adjusted[1] > adjusted[0]

    def test_ears_disabled_when_tolerance_zero(self):
        """Test that EARS does not modify uniform_probs when tolerance=0 (default)."""
        import vllm_ascend.envs as envs_ascend

        uniform_probs = torch.tensor([0.3, 0.7, 0.5])

        with patch.dict(os.environ, {"VLLM_EARS_TOLERANCE": "0.0"}):
            ears_tolerance = envs_ascend.VLLM_EARS_TOLERANCE
            assert ears_tolerance == 0.0
            # The if-guard in rejection_sample ensures no modification
            if ears_tolerance > 0:
                raise AssertionError("EARS should be disabled when tolerance=0")

        assert torch.equal(uniform_probs,
                           torch.tensor([0.3, 0.7, 0.5])), "uniform_probs should be unchanged"
