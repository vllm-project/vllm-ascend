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
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.outputs import SamplerOutput

from tests.ut.base import TestBase
from vllm_ascend.sample.rejection_sampler import (
    RejectionSampler,
    _should_use_cpu_strict_sampling,
    expand_batch_to_tokens,
    expand_pytorch,
    rejection_greedy_sample_pytorch,
    rejection_random_sample_pytorch,
    sample_recovered_tokens_pytorch,
    strict_rejection_sample_tensor,
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
    def test_cpu_strict_sampling_is_skipped_with_seeded_runtime_state(self):
        sampling_metadata = SimpleNamespace(
            all_greedy=False,
            max_num_logprobs=None,
            seeds=torch.tensor([3], dtype=torch.int64),
        )
        logits = SimpleNamespace(
            device=SimpleNamespace(type="npu"),
        )

        assert not _should_use_cpu_strict_sampling(
            None,
            logits,
            sampling_metadata,
            torch.tensor([10], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
        )

    def test_cpu_strict_sampling_remains_available_without_seeded_runtime_state(
        self,
    ):
        sampling_metadata = SimpleNamespace(
            all_greedy=False,
            max_num_logprobs=None,
            seeds=None,
        )
        logits = SimpleNamespace(
            device=SimpleNamespace(type="npu"),
        )

        assert _should_use_cpu_strict_sampling(
            None,
            logits,
            sampling_metadata,
            torch.tensor([10], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
        )

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_forward_no_draft_probs_uses_strict_verifier(self):
        class FakeSampler:
            logprobs_mode = "raw_logprobs"

            def __call__(self, **_kwargs):
                return SamplerOutput(
                    sampled_token_ids=torch.tensor([[100], [200]], dtype=torch.int32),
                    logprobs_tensors=None,
                )

        sampler = RejectionSampler(FakeSampler())
        metadata = SimpleNamespace(
            max_spec_len=1,
            bonus_logits_indices=torch.tensor([1, 3], dtype=torch.int64),
            target_logits_indices=torch.tensor([0, 2], dtype=torch.int64),
            draft_token_ids=torch.tensor([11, 22], dtype=torch.int32),
            num_draft_tokens=[1, 1],
            cu_num_draft_tokens=torch.tensor([1, 2], dtype=torch.int32),
        )
        sampling_metadata = SimpleNamespace(
            temperature=torch.ones(2, dtype=torch.float32),
            all_greedy=False,
            all_random=True,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=None,
            presence_penalties=None,
            repetition_penalties=None,
            output_token_ids=[[], []],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=SimpleNamespace(
                argmax_invariant=[],
                non_argmax_invariant=[],
            ),
            spec_token_ids=[[], []],
            seeds=torch.tensor([3, 5], dtype=torch.int64),
        )
        sampling_metadata._ascend_positions = torch.arange(4, dtype=torch.int32)
        sampling_metadata._ascend_idx_mapping = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        logits = torch.tensor(
            [
                [0.0, 4.0, 1.0],
                [5.0, 0.0, 1.0],
                [0.0, 2.0, 6.0],
                [1.0, 7.0, 0.0],
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor([[1, -1], [2, 200]], dtype=torch.int32)

        with (
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False),
            patch("vllm_ascend.sample.rejection_sampler.rejection_sample") as mock_random_rejection,
            patch(
                "vllm_ascend.sample.rejection_sampler.strict_rejection_sample_tensor",
                return_value=expected,
            ) as mock_strict,
        ):
            output = sampler(
                metadata,
                None,
                logits,
                sampling_metadata,
            )

        mock_random_rejection.assert_not_called()
        mock_strict.assert_called_once()
        target_token_ids = mock_strict.call_args.args[3]
        assert torch.equal(target_token_ids, torch.tensor([1, 2], dtype=torch.int32))
        assert torch.equal(output.sampled_token_ids, expected)

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_forward_no_draft_probs_cpu_strict_sample_skips_sampler(self):
        class FakeSampler:
            logprobs_mode = "raw_logprobs"

            def __call__(self, **_kwargs):
                raise AssertionError("CPU strict path must not call sampler")

            @staticmethod
            def apply_logits_processors(logits, sampling_metadata, predict_bonus_token):
                return logits

        sampler = RejectionSampler(FakeSampler())
        metadata = SimpleNamespace(
            max_spec_len=1,
            bonus_logits_indices=torch.tensor([1], dtype=torch.int64),
            target_logits_indices=torch.tensor([0], dtype=torch.int64),
            draft_token_ids=torch.tensor([1], dtype=torch.int32),
            num_draft_tokens=[1],
            cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32),
        )
        sampling_metadata = SimpleNamespace(
            temperature=torch.ones(1, dtype=torch.float32),
            all_greedy=False,
            all_random=True,
            top_p=None,
            top_k=torch.tensor([1], dtype=torch.int32),
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=None,
            presence_penalties=None,
            repetition_penalties=None,
            output_token_ids=[[]],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=SimpleNamespace(
                argmax_invariant=[],
                non_argmax_invariant=[],
            ),
            spec_token_ids=[[]],
            seeds=torch.tensor([3], dtype=torch.int64),
        )
        sampling_metadata._ascend_positions = torch.tensor([10, 11], dtype=torch.int32)
        sampling_metadata._ascend_idx_mapping = torch.tensor([0, 0], dtype=torch.int32)
        logits = torch.tensor(
            [
                [0.0, 8.0, 1.0],
                [0.0, 1.0, 9.0],
            ],
            dtype=torch.float32,
        )

        with patch(
            "vllm_ascend.sample.rejection_sampler._should_use_cpu_strict_sampling",
            return_value=True,
        ):
            output = sampler(
                metadata,
                None,
                logits,
                sampling_metadata,
            )

        expected = torch.tensor([[1, 2]], dtype=torch.int32)
        assert torch.equal(output.sampled_token_ids.cpu(), expected)

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
    def test_strict_rejection_sample_tensor_supports_mtp_1_2_3(self):
        draft_token_ids = torch.tensor([10, 20, 21, 30, 31, 32])
        target_token_ids = torch.tensor([10, 20, 99, 30, 31, 32])
        bonus_token_ids = torch.tensor([100, 200, 300])
        cu_num_draft_tokens = torch.tensor([1, 3, 6], dtype=torch.int32)

        output_token_ids = strict_rejection_sample_tensor(
            draft_token_ids,
            cu_num_draft_tokens,
            max_spec_len=3,
            target_token_ids=target_token_ids,
            bonus_token_ids=bonus_token_ids,
        )

        expected = torch.tensor(
            [
                [10, 100, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
                [20, 99, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
                [30, 31, 32, 300],
            ],
            dtype=torch.int32,
        )
        assert torch.equal(output_token_ids, expected)

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
    def test_sample_recovered_tokens_pytorch_autoregressive(self):
        """Test recovered token sampling for autoregressive models"""
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
