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
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.rejection_sampler import (
    _should_use_fused_rejection_sampler,
    clear_sampling_metadata_draft_logits,
    expand_batch_to_tokens,
    expand_pytorch,
    rejection_greedy_sample_pytorch,
    rejection_random_sample_block_verify_pytorch,
    rejection_random_sample_pytorch,
    rejection_sample,
    sample_recovered_tokens_blockwise_pytorch,
    sample_recovered_tokens_pytorch,
    set_sampling_metadata_draft_logits,
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
    def _sampling_metadata(
        self,
        *,
        all_greedy=False,
        all_random=True,
        temperature=None,
        generators=None,
    ):
        return SimpleNamespace(
            all_greedy=all_greedy,
            all_random=all_random,
            temperature=torch.tensor([1.0]) if temperature is None else temperature,
            generators={} if generators is None else generators,
            top_k=None,
            top_p=None,
        )

    def test_fused_rejection_sampler_guard(self):
        draft_logits = torch.ones((1, 2, 4), dtype=torch.float32)
        draft_probs = torch.ones((2, 4), dtype=torch.float32)
        npu_device = SimpleNamespace(type="npu")
        cpu_device = torch.device("cpu")

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", True),
        ):
            assert _should_use_fused_rejection_sampler(
                draft_logits,
                None,
                self._sampling_metadata(all_random=True),
                [2],
                2,
                npu_device,
            )
            assert _should_use_fused_rejection_sampler(
                None,
                draft_probs,
                self._sampling_metadata(all_random=True),
                [2],
                2,
                npu_device,
            )
            assert not _should_use_fused_rejection_sampler(
                None,
                draft_probs,
                self._sampling_metadata(all_random=True),
                [2],
                2,
                cpu_device,
            )
            assert not _should_use_fused_rejection_sampler(
                None,
                draft_probs,
                self._sampling_metadata(all_random=False),
                [2],
                2,
                npu_device,
            )
            assert not _should_use_fused_rejection_sampler(
                None,
                None,
                self._sampling_metadata(all_random=True),
                [2],
                2,
                npu_device,
            )
            assert not _should_use_fused_rejection_sampler(
                None,
                draft_probs,
                self._sampling_metadata(all_random=True),
                [0],
                2,
                npu_device,
            )
            assert not _should_use_fused_rejection_sampler(
                None,
                draft_probs,
                self._sampling_metadata(all_random=True),
                [3],
                2,
                npu_device,
            )

        with (
            patch.dict(os.environ, {"VLLM_ASCEND_USE_FUSED_REJECTION": "1"}),
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", True),
        ):
            assert _should_use_fused_rejection_sampler(
                None,
                draft_probs,
                self._sampling_metadata(all_random=True),
                [2],
                2,
                npu_device,
            )

        with (
            patch.dict(os.environ, {"VLLM_ASCEND_USE_FUSED_REJECTION": "0"}),
            patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", True),
        ):
            assert not _should_use_fused_rejection_sampler(
                None,
                draft_probs,
                self._sampling_metadata(all_random=True),
                [2],
                2,
                npu_device,
            )

    def test_rejection_sample_uses_fused_path_when_supported(self):
        draft_token_ids = torch.tensor([1, 2], dtype=torch.int64)
        num_draft_tokens = [2]
        max_spec_len = 2
        cu_num_draft_tokens = torch.tensor([2], dtype=torch.int32)
        draft_probs = torch.full((2, 4), 0.25, dtype=torch.float32)
        target_logits = torch.zeros((2, 4), dtype=torch.float32)
        bonus_token_ids = torch.tensor([[3]], dtype=torch.int64)
        metadata = self._sampling_metadata()
        expected = torch.tensor([[1, 3, PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
        uniform_resample = torch.full((1, 4), 0.5, dtype=torch.float32)

        with (
            patch("vllm_ascend.sample.rejection_sampler._should_use_fused_rejection_sampler", return_value=True),
            patch(
                "vllm_ascend.sample.rejection_sampler.generate_uniform_probs",
                return_value=torch.tensor([0.1, 0.2], dtype=torch.float32),
            ),
            patch(
                "vllm_ascend.sample.rejection_sampler.generate_uniform_resample",
                return_value=uniform_resample,
            ) as mock_uniform_resample,
            patch(
                "vllm_ascend.sample.rejection_sampler.fused_rejection_sample_from_probs",
                return_value=expected,
            ) as mock_fused,
        ):
            actual = rejection_sample(
                draft_token_ids,
                num_draft_tokens,
                max_spec_len,
                cu_num_draft_tokens,
                draft_probs,
                target_logits,
                bonus_token_ids,
                metadata,
            )

        assert torch.equal(actual, expected)
        mock_uniform_resample.assert_called_once_with(1, 4, metadata.generators, target_logits.device)
        mock_fused.assert_called_once()
        assert torch.equal(mock_fused.call_args.args[0], draft_token_ids)
        assert mock_fused.call_args.args[1] == num_draft_tokens
        assert mock_fused.call_args.args[2] == max_spec_len
        assert torch.equal(mock_fused.call_args.args[7], torch.tensor([0.1, 0.2], dtype=torch.float32))
        assert torch.equal(mock_fused.call_args.args[8], uniform_resample)

    def test_rejection_sample_uses_fused_logits_path_when_supported(self):
        draft_token_ids = torch.tensor([1, 2], dtype=torch.int64)
        num_draft_tokens = [2]
        max_spec_len = 2
        cu_num_draft_tokens = torch.tensor([2], dtype=torch.int32)
        draft_logits = torch.full((1, 2, 4), 0.25, dtype=torch.float32)
        target_logits = torch.zeros((2, 4), dtype=torch.float32)
        bonus_token_ids = torch.tensor([[3]], dtype=torch.int64)
        metadata = self._sampling_metadata()
        expected = torch.tensor([[1, 3, PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
        uniform_resample = torch.full((1, 4), 0.5, dtype=torch.float32)

        set_sampling_metadata_draft_logits(metadata, draft_logits)
        try:
            with (
                patch("vllm_ascend.sample.rejection_sampler._should_use_fused_rejection_sampler", return_value=True),
                patch(
                    "vllm_ascend.sample.rejection_sampler.generate_uniform_probs",
                    return_value=torch.tensor([0.1, 0.2], dtype=torch.float32),
                ),
                patch(
                    "vllm_ascend.sample.rejection_sampler.generate_uniform_resample",
                    return_value=uniform_resample,
                ) as mock_uniform_resample,
                patch(
                    "vllm_ascend.sample.rejection_sampler.fused_rejection_sample_from_logits",
                    return_value=expected,
                ) as mock_fused,
            ):
                actual = rejection_sample(
                    draft_token_ids,
                    num_draft_tokens,
                    max_spec_len,
                    cu_num_draft_tokens,
                    None,
                    target_logits,
                    bonus_token_ids,
                    metadata,
                )
        finally:
            clear_sampling_metadata_draft_logits(metadata)

        assert torch.equal(actual, expected)
        mock_uniform_resample.assert_called_once_with(1, 4, metadata.generators, target_logits.device)
        mock_fused.assert_called_once()
        assert torch.equal(mock_fused.call_args.args[0], draft_token_ids)
        assert mock_fused.call_args.args[1] == num_draft_tokens
        assert mock_fused.call_args.args[2] == max_spec_len
        assert torch.equal(mock_fused.call_args.args[4], draft_logits)

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
    def test_rejection_random_sample_block_verify_pytorch(self):
        """Test random rejection sampling for block verify: accept based on uniform probability"""
        batch_size = 2
        max_spec_len = 3
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 1])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.0, 0.4],
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

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_sample_recovered_tokens_blockwise_pytorch_ngram(self):
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

        sample_recovered_tokens_blockwise_pytorch(
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
    def test_sample_recovered_tokens_blockwise_pytorch(self):
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

        sample_recovered_tokens_blockwise_pytorch(
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
