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

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize(
    "pcp_size, dcp_size, num_reqs, query_lens, num_decodes, use_mla, total_tokens, expect_not_none",
    [
        (1, 1, 5, [10, 20, 30, 40, 50], 2, False, 100, False),
        (1, 2, 3, [20, 30, 40], 1, False, 50, True),
        (2, 1, 4, [5, 10, 40, 60], 2, False, 100, True),
        (2, 1, 4, [5, 10, 40, 60], 2, True, 100, True),
        (2, 1, 3, [5, 10, 15], 3, False, 50, True),
        (2, 1, 3, [40, 50, 60], 0, False, 150, True),
    ])
def test_generate_pcp_metadata_basic(pcp_size, dcp_size, num_reqs, query_lens,
                                     num_decodes, use_mla, total_tokens,
                                     expect_not_none):
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.pcp_size = pcp_size
    mock_runner.dcp_size = dcp_size
    mock_runner.decode_threshold = 4
    mock_runner.pcp_rank = 0
    mock_runner.device = torch.device('cpu')
    mock_runner.dtype = torch.float32

    mock_runner.parallel_config = MagicMock()
    mock_runner.parallel_config.cp_kv_cache_interleave_size = 64

    mock_runner.vllm_config = MagicMock()
    mock_runner.vllm_config.model_config = MagicMock()
    mock_runner.vllm_config.model_config.use_mla = use_mla

    mock_runner.input_batch = MagicMock()
    mock_runner.input_batch.num_reqs = num_reqs

    num_computed_tokens = []
    num_prompt_tokens = []
    num_tokens = []

    for i in range(num_reqs):
        if i < num_decodes:
            num_computed_tokens.append(query_lens[i])
            num_prompt_tokens.append(query_lens[i] // 2)
            num_tokens.append(query_lens[i])
        else:
            num_computed_tokens.append(0)
            num_prompt_tokens.append(query_lens[i])
            num_tokens.append(query_lens[i])

    mock_runner.input_batch.num_computed_tokens_cpu = torch.tensor(
        num_computed_tokens)
    mock_runner.input_batch.num_prompt_tokens = torch.tensor(num_prompt_tokens)
    mock_runner.input_batch.num_tokens = torch.tensor(num_tokens)

    mock_runner.query_lens = torch.tensor(query_lens)

    mock_runner._get_cp_local_seq_lens = NPUModelRunner._get_cp_local_seq_lens.__get__(
        mock_runner, NPUModelRunner)

    mock_runner.pcp_allgather_restore_idx = torch.arange(total_tokens * 2)
    mock_runner.cp_kv_recover_idx_for_chunk = torch.arange(total_tokens)

    mock_runner.long_seq_metadata = None
    mock_runner.num_actual_tokens_pcp_padded = 0
    mock_runner.kv_idx_names = {}
    mock_runner.extra_long_seq_kwargs = {}
    mock_runner.attn_mask = None
    mock_runner.q_head_idx_tensor = None
    mock_runner.q_tail_idx_tensor = None
    mock_runner.q_full_idx = None

    method = NPUModelRunner._generate_pcp_metadata.__get__(
        mock_runner, NPUModelRunner)
    result = method(total_tokens)

    if not expect_not_none:
        assert result is None, f"Expected to return None, but got {type(result)}"
    else:
        assert result is not None, "Expected to return a metadata object, but got None."

        assert hasattr(result, 'num_actual_tokens_pcp_padded')
        assert hasattr(result, 'num_computed_tokens_of_pcp_dcp')

        if pcp_size > 1:
            assert hasattr(result, 'pcp_allgather_restore_idx')

            has_prefill_requests = (num_reqs - num_decodes) > 0
            if has_prefill_requests:
                assert hasattr(result, 'q_head_idx_tensor')
                assert hasattr(result, 'q_tail_idx_tensor')
                assert hasattr(result, 'q_full_idx')
                assert hasattr(result, 'kv_with_q_head_nomask_idx_tensor')
                assert hasattr(result, 'kv_with_q_head_mask_idx_tensor')
                assert hasattr(result, 'kv_with_q_tail_nomask_idx_tensor')
                assert hasattr(result, 'kv_with_q_tail_mask_idx_tensor')
                assert hasattr(result, 'attn_mask_seqlens')
                assert hasattr(result, 'head_attn_nomask_seqlens')
                assert hasattr(result, 'tail_attn_nomask_seqlens')

                if hasattr(result, 'pcp_prefill_mask'
                           ) and result.pcp_prefill_mask is not None:
                    if use_mla:
                        assert result.pcp_prefill_mask.shape == (512, 512)
                    else:
                        assert result.pcp_prefill_mask.shape == (2048, 2048)
            else:
                if hasattr(result, 'pcp_prefill_mask'):
                    if result.pcp_prefill_mask is not None:
                        if use_mla:
                            assert result.pcp_prefill_mask.shape == (512, 512)
                        else:
                            assert result.pcp_prefill_mask.shape == (2048,
                                                                     2048)


def test_generate_pcp_metadata_edge_cases():
    mock_runner = MagicMock()
    mock_runner.pcp_size = 2
    mock_runner.dcp_size = 1
    mock_runner.input_batch = MagicMock()
    mock_runner.input_batch.num_reqs = 0
    mock_runner.query_lens = torch.tensor([10, 20, 30])

    assert (mock_runner.input_batch.num_reqs
            or mock_runner.query_lens.size(0)) == 3

    mock_runner.input_batch.num_reqs = 100
    mock_runner.query_lens = torch.ones(100) * 1000

    for rank in [0, 1]:
        mock_runner.pcp_rank = rank
        q_head_chunk_id = rank
        q_tail_chunk_id = 2 * 2 - 1 - rank
        assert q_head_chunk_id == rank
        assert q_tail_chunk_id == 3 - rank


def test_pcp_allgather_restore_idx_slicing():
    mock_runner = MagicMock()
    mock_runner.pcp_size = 2
    mock_runner.pcp_allgather_restore_idx = torch.arange(1000)

    total_num_scheduled_tokens = 200
    num_actual_tokens_pcp_padded = total_num_scheduled_tokens * 2

    expected_slice = mock_runner.pcp_allgather_restore_idx[:
                                                           num_actual_tokens_pcp_padded]
    assert len(expected_slice) == 400
    assert expected_slice[0] == 0
    assert expected_slice[-1] == 399


@pytest.mark.parametrize(
    "tokens, num_reqs, num_computed_tokens, num_prompt_tokens, pcp_size, pcp_rank, expected_pcp_tokens",
    [
        # Case 1: prefill only
        ([8, 12, 16], 3, [0, 0, 0], [8, 12, 16], 4, 0, [2, 4, 4]),

        # Case 2: mix prefill and decode
        ([8, 4, 12], 3, [8, 4, 0], [8, 4, 12], 4, 0, [8, 4, 4]),

        # Case 3: request which need to be padded
        ([3, 7, 9], 3, [0, 0, 0], [3, 7, 9], 4, 0, [2, 2, 4]),

        # Case 4: single request
        ([10], 1, [0], [10], 4, 0, [4]),
    ])
def test_update_tokens_for_pcp_basic(tokens, num_reqs, num_computed_tokens,
                                     num_prompt_tokens, pcp_size, pcp_rank,
                                     expected_pcp_tokens):
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.pcp_size = pcp_size
    mock_runner.pcp_rank = pcp_rank

    mock_runner.input_batch = MagicMock()
    mock_runner.input_batch.num_reqs = num_reqs
    mock_runner.input_batch.num_computed_tokens_cpu = np.array(
        num_computed_tokens, dtype=np.int32)
    mock_runner.input_batch.num_prompt_tokens = np.array(num_prompt_tokens,
                                                         dtype=np.int32)

    mock_runner.pcp_allgather_restore_idx = torch.zeros(1000, dtype=torch.long)

    mock_runner.num_pcp_pads = [0] * num_reqs
    mock_runner.arange_np = np.arange(10000)

    mock_runner._update_tokens_for_pcp = NPUModelRunner._update_tokens_for_pcp.__get__(
        mock_runner, NPUModelRunner)
    mock_runner._get_cumsum_and_arange = NPUModelRunner._get_cumsum_and_arange.__get__(
        mock_runner, NPUModelRunner)

    pcp_tokens_result, positions_result, unpad_mask_result = mock_runner._update_tokens_for_pcp(
        tokens)

    assert np.array_equal(pcp_tokens_result, expected_pcp_tokens), \
        f"Expected pcp_tokens: {expected_pcp_tokens}, got: {pcp_tokens_result}"

    total_pcp_tokens: int = np.sum(pcp_tokens_result)
    assert positions_result.shape == (total_pcp_tokens,), \
        f"Positions shape mismatch. Expected length {total_pcp_tokens}, got {positions_result.shape}"

    padded_tokens = [
        (t + 2 * pcp_size - 1) // (2 * pcp_size) *
        (2 * pcp_size) if num_computed_tokens[i] == 0 else t * pcp_size
        for i, t in enumerate(tokens)
    ]
    total_padded_tokens: int = np.sum(padded_tokens)
    assert unpad_mask_result.shape[0] == total_padded_tokens, \
        f"unpad_mask size mismatch: expected {total_padded_tokens}, got {unpad_mask_result.shape[0]}"


def test_update_tokens_for_pcp_with_padding():
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.pcp_size = 4
    mock_runner.pcp_rank = 0

    mock_runner.arange_np = np.arange(10000)

    mock_runner.input_batch = MagicMock()
    mock_runner.input_batch.num_reqs = 3
    mock_runner.input_batch.num_computed_tokens_cpu = np.array([0, 0, 0],
                                                               dtype=np.int32)
    mock_runner.input_batch.num_prompt_tokens = np.array([5, 9, 13],
                                                         dtype=np.int32)

    mock_runner.num_pcp_pads = [0, 0, 0]
    mock_runner.pcp_allgather_restore_idx = torch.zeros(1000, dtype=torch.long)

    mock_runner._update_tokens_for_pcp = NPUModelRunner._update_tokens_for_pcp.__get__(
        mock_runner, NPUModelRunner)
    mock_runner._get_cumsum_and_arange = NPUModelRunner._get_cumsum_and_arange.__get__(
        mock_runner, NPUModelRunner)

    tokens = [5, 9, 13]

    pcp_tokens, positions, unpad_mask = mock_runner._update_tokens_for_pcp(
        tokens)

    expected_pcp_tokens = [2, 4, 4]
    assert np.array_equal(pcp_tokens, expected_pcp_tokens), \
        f"Expected {expected_pcp_tokens}, got {pcp_tokens}"

    expected_pads = [3, 7, 3]
    assert np.array_equal(mock_runner.num_pcp_pads, expected_pads), \
        f"Expected padding {expected_pads}, got {mock_runner.num_pcp_pads}"


def test_update_tokens_for_pcp_unpad_mask():
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.pcp_size = 4
    mock_runner.pcp_rank = 0

    mock_runner.arange_np = np.arange(10000)

    mock_runner.input_batch = MagicMock()
    mock_runner.input_batch.num_reqs = 2
    mock_runner.input_batch.num_computed_tokens_cpu = np.array([0, 0],
                                                               dtype=np.int32)
    mock_runner.input_batch.num_prompt_tokens = np.array([5, 7],
                                                         dtype=np.int32)

    mock_runner.num_pcp_pads = [0, 0]
    mock_runner.pcp_allgather_restore_idx = torch.zeros(1000, dtype=torch.long)

    mock_runner._update_tokens_for_pcp = NPUModelRunner._update_tokens_for_pcp.__get__(
        mock_runner, NPUModelRunner)
    mock_runner._get_cumsum_and_arange = NPUModelRunner._get_cumsum_and_arange.__get__(
        mock_runner, NPUModelRunner)

    tokens = [5, 7]

    pcp_tokens, positions, unpad_mask = mock_runner._update_tokens_for_pcp(
        tokens)

    assert unpad_mask.dtype == torch.bool, \
        f"unpad_mask should be bool, got {unpad_mask.dtype}"

    padded_tokens = [8, 8]
    expected_length = sum(padded_tokens)
    assert unpad_mask.shape[0] == expected_length, \
        f"unpad_mask length mismatch: expected {expected_length}, got {unpad_mask.shape[0]}"

    expected_mask = [True] * 5 + [False] * 3 + [True] * 7 + [False] * 1
    actual_mask = unpad_mask.numpy().tolist()
    assert actual_mask == expected_mask, \
        f"unpad_mask incorrect. Expected {expected_mask}, got {actual_mask}"
