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

import pytest
import torch
from vllm.triton_utils import triton

from vllm_ascend.sample.rejection_sampler import (
    expand_kernel, rejection_greedy_sample_kernel,
    rejection_random_sample_kernel, sample_recovered_tokens_kernel)

# Global constants
PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = 0.0
MAX_SPEC_LEN = 8  # Used as MAX_NUM_TOKENS in expand_batch_to_tokens
# Test cases parameters
BATCH_SIZES = [1, 4, 16, 128]
VOCAB_SIZES = [
    151936,  # Qwen3-32B Qwen3-235B
    129280,  # Deepseek R1
]
SPEC_LENS = [1, 2, 3, 4, 5]
DTYPES = [torch.bfloat16, torch.float16]
DEVICE = f"npu:{0}"


@torch.inference_mode()
def test_rejection_greedy_sample():
    """Test greedy rejection sampling: stop when draft doesn't match, otherwise append bonus token"""
    global DEVICE
    batch_size = 2
    max_spec_len = 2
    output_token_ids = torch.full((batch_size, max_spec_len + 1),
                                  PLACEHOLDER_TOKEN_ID,
                                  dtype=torch.int32,
                                  device=DEVICE)

    cu_num_draft_tokens = torch.tensor([2, 4],
                                       dtype=torch.int32,
                                       device=DEVICE)
    draft_token_ids = torch.tensor([10, 11, 20, 21],
                                   dtype=torch.int32,
                                   device=DEVICE)
    target_argmax = torch.tensor([10, 99, 20, 22],
                                 dtype=torch.int32,
                                 device=DEVICE)
    bonus_token_ids = torch.tensor([[100], [200]],
                                   dtype=torch.int32,
                                   device=DEVICE)

    is_greedy = torch.tensor([True, True], dtype=torch.bool, device=DEVICE)

    rejection_greedy_sample_kernel[(batch_size, )](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
    )

    assert output_token_ids[0, 0].item() == 10
    assert output_token_ids[0, 1].item() == 99
    assert output_token_ids[1, 0].item() == 20
    assert output_token_ids[1, 2].item() == PLACEHOLDER_TOKEN_ID


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_rejection_random_sample(dtype):
    """Test random rejection sampling: accept based on uniform probability"""
    global DEVICE
    batch_size = 2
    max_spec_len = 3
    output_token_ids = torch.full((batch_size, max_spec_len + 1),
                                  PLACEHOLDER_TOKEN_ID,
                                  dtype=torch.int32,
                                  device=DEVICE)

    cu_num_draft_tokens = torch.tensor([2, 1],
                                       dtype=torch.int32,
                                       device=DEVICE)
    draft_token_ids = torch.tensor([1, 0, 2], dtype=torch.int32, device=DEVICE)
    draft_probs = torch.tensor(
        [
            [0.0, 0.6, 0.0, 0.4],  # vocab_size=4
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=dtype,
        device=DEVICE)
    target_probs = torch.tensor([
        [0.0, 0.8, 0.0, 0.2],
        [0.2, 0.1, 0.3, 0.4],
        [0.9, 0.1, 0.0, 0.0],
    ],
                                dtype=dtype,
                                device=DEVICE)
    bonus_token_ids = torch.tensor([[100], [200]],
                                   dtype=torch.int32,
                                   device=DEVICE)
    recovered_token_ids = torch.tensor([1, 2, 3],
                                       dtype=torch.int32,
                                       device=DEVICE)
    uniform_probs = torch.tensor([0.7, 0.6, 0.5],
                                 dtype=torch.int32,
                                 device=DEVICE)
    is_greedy = torch.tensor([False, False], dtype=torch.bool, device=DEVICE)
    vocab_size = 4

    rejection_random_sample_kernel[(batch_size, )](
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
        NO_DRAFT_PROBS=False,
    )

    assert output_token_ids[0, 0].item() == 1
    assert output_token_ids[0, 1].item() == 0
    assert output_token_ids[0, 2].item() == 100


@torch.inference_mode()
def test_expand_pytorch():
    """Test expand_pytorch functionality"""
    global DEVICE
    input_ptr = torch.tensor([10, 20, 30], dtype=torch.int32, device=DEVICE)
    cu_num_tokens_ptr = torch.tensor([2, 5, 7],
                                     dtype=torch.int32,
                                     device=DEVICE)
    output_ptr = torch.empty(7, dtype=torch.int32, device=DEVICE)
    batch_size = input_ptr.shape[0]

    expand_kernel[(batch_size, )](
        output_ptr,
        input_ptr,
        cu_num_tokens_ptr,
        replace_from=0,
        replace_to=0,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,
    )

    expected = torch.tensor([10, 10, 20, 20, 20, 30, 30],
                            dtype=torch.int32,
                            device=DEVICE)
    assert torch.equal(output_ptr, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_sample_recovered_tokens_pytorch_ngram(dtype):
    """Test recovered token sampling under n-gram mode"""
    global DEVICE
    output_token_ids = torch.empty(2, dtype=torch.int32, device=DEVICE)
    cu_num_draft_tokens = torch.tensor([1, 2],
                                       dtype=torch.int32,
                                       device=DEVICE)
    draft_token_ids = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    draft_probs = None
    target_probs = torch.tensor([
        [0.1, 0.2, 0.7],
        [0.3, 0.3, 0.4],
    ],
                                dtype=dtype,
                                device=DEVICE)
    q = torch.tensor([
        [0.1, 0.2, 0.7],
        [0.5, 0.4, 0.1],
    ],
                     dtype=dtype,
                     device=DEVICE)
    vocab_size = 3
    max_spec_len = 1
    batch_size = cu_num_draft_tokens.shape[0]

    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        NO_DRAFT_PROBS=True,
        SUB_BLOCK=4 * 1024,
        multibuffer=False,
    )

    assert output_token_ids[0].item() == 0
    assert output_token_ids[1].item() == 1


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_sample_recovered_tokens_pytorch_autoregressive(dtype):
    """Test recovered token sampling for autoregressive models"""
    global DEVICE
    output_token_ids = torch.empty(2, dtype=torch.int32, device=DEVICE)
    cu_num_draft_tokens = torch.tensor([1, 2],
                                       dtype=torch.int32,
                                       device=DEVICE)
    draft_token_ids = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    draft_probs = torch.tensor([
        [0.6, 0.1, 0.3],
        [0.2, 0.7, 0.1],
    ],
                               dtype=dtype,
                               device=DEVICE)
    target_probs = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.3, 0.6, 0.1],
    ],
                                dtype=dtype,
                                device=DEVICE)
    q = torch.tensor([
        [0.5, 0.3, 0.2],
        [0.1, 0.8, 0.1],
    ],
                     dtype=dtype,
                     device=DEVICE)
    vocab_size = 3
    max_spec_len = 1
    batch_size = cu_num_draft_tokens.shape[0]

    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        NO_DRAFT_PROBS=False,
        SUB_BLOCK=4 * 1024,
        multibuffer=False,
    )

    assert output_token_ids[0].item() == 0
    assert output_token_ids[1].item() == 0
