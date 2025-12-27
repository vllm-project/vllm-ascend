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
    mock_runner.speculative_config = None

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

    mock_runner._get_cp_local_seq_lens.side_effect = NPUModelRunner._get_cp_local_seq_lens.__get__(
        mock_runner, NPUModelRunner)
    mock_runner._list_to_tensor.side_effect = NPUModelRunner._list_to_tensor.__get__(
        mock_runner, NPUModelRunner)
    mock_runner._split_nomask_idx_tensor_list.side_effect = NPUModelRunner._split_nomask_idx_tensor_list.__get__(
        mock_runner, NPUModelRunner)
    mock_runner._split_multi_batch_kv_idx.side_effect = NPUModelRunner._split_multi_batch_kv_idx.__get__(
        mock_runner, NPUModelRunner)

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

    result = NPUModelRunner._generate_pcp_metadata(mock_runner, total_tokens)

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
    "tokens, num_reqs, num_computed_tokens, num_prompt_tokens," \
    "pcp_size, pcp_rank, decode_threshold, expected_pcp_tokens",
    [
        # Case 1: prefill only
        ([8, 12, 16], 3, [0, 0, 0], [8, 12, 16], 4, 0, 1, [2, 4, 4]),

        # Case 2: mix prefill and decode (with spec decode)
        ([8, 4, 12], 3, [8, 4, 0], [8, 4, 12], 4, 0, 8, [8, 4, 4]),

        # Case 3: request which need to be padded
        ([3, 7, 9], 3, [0, 0, 0], [3, 7, 9], 4, 0, 1, [2, 2, 4]),

        # Case 4: single request
        ([10], 1, [0], [10], 4, 0, 1, [4]),
    ])
def test_update_tokens_for_pcp_basic(tokens, num_reqs, num_computed_tokens,
                                     num_prompt_tokens, pcp_size, pcp_rank,
                                     decode_threshold, expected_pcp_tokens):
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
    mock_runner.decode_threshold = decode_threshold

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
    mock_runner.decode_threshold = 1

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
    mock_runner.decode_threshold = 1

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


# yapf: disable
@pytest.mark.parametrize(
    "seq_lens, pcp_world_size, dcp_world_size, cp_kv_cache_interleave_size, target",
    [
        # without pcp and dcp
        (torch.tensor([1, 2, 128, 129]), 1, 1, 1,
        torch.tensor([[[1]], [[2]], [[128]], [[129]]])),
        # pcp
        (torch.tensor([1, 2, 128, 129]), 2, 1, 1,
        torch.tensor([[[1], [0]], [[1], [1]], [[64], [64]], [[65], [64]]])),
        # dcp
        (torch.tensor([1, 2, 128, 129]), 1, 2, 1,
        torch.tensor([[[1, 0]], [[1, 1]], [[64, 64]], [[65, 64]]])),
        # pcp + dcp
        (torch.tensor([1, 2, 128, 129]), 2, 2, 1,
        torch.tensor([[[1, 0], [0, 0]], [[1, 1], [0, 0]],
                     [[32, 32], [32, 32]], [[33, 32], [32, 32]]])),
        # specify interleave_size
        (torch.tensor([1, 2, 128, 129]), 2, 1, 2,
        torch.tensor([[[1], [0]], [[2], [0]], [[64], [64]], [[65], [64]]])),
        (torch.tensor([1, 2, 128, 129]), 2, 1, 128,
        torch.tensor([[[1], [0]], [[2], [0]], [[128], [0]], [[128], [1]]])),
        (torch.tensor([1, 2, 128, 129, 256, 257]), 2, 2, 128,
        torch.tensor([[[1, 0], [0, 0]], [[2, 0], [0, 0]],
                     [[128, 0], [0, 0]], [[128, 1], [0, 0]],
                     [[128, 128], [0, 0]], [[128, 128], [1, 0]]])),
    ]
)
# yapf: enable
def test_get_cp_local_seq_lens(
    seq_lens,
    pcp_world_size,
    dcp_world_size,
    cp_kv_cache_interleave_size,
    target,
):
    mock_runner = MagicMock(spec=NPUModelRunner)
    ret = NPUModelRunner._get_cp_local_seq_lens(mock_runner, seq_lens,
                                                pcp_world_size, dcp_world_size,
                                                cp_kv_cache_interleave_size)
    assert torch.equal(ret, target)


@pytest.fixture
def pcp_mtp_mock_runner():
    # set up pcp & mtp related buffers
    max_num_reqs = 4
    max_model_len = 4096
    max_num_tokens = 4096
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.device = 'cpu'
    mock_runner.pin_memory = False

    # Init model_runner pcp_mtp related buffers
    mock_runner.query_start_loc_pcp_full = NPUModelRunner._make_buffer(
        mock_runner, max_num_reqs + 1, dtype=torch.int32)

    positions_buff = torch.zeros(max_num_tokens,
                                 dtype=torch.int64,
                                 device="cpu")
    mock_runner.positions_pcp_full = positions_buff
    mock_runner.positions_pcp_full_np = positions_buff.numpy()

    mock_runner.input_ids_pcp_full = NPUModelRunner._make_buffer(
        mock_runner, max_num_tokens, dtype=torch.int32)
    mock_runner.query_lens_pcp_full = NPUModelRunner._make_buffer(
        mock_runner, max_num_reqs, dtype=torch.int32)
    mock_runner.decode_threshold = 1

    mock_runner.arange_np = np.arange(max_model_len)
    mock_runner.input_batch = MagicMock()
    mock_runner.input_batch.num_computed_tokens_cpu = \
        np.zeros(max_num_reqs, dtype=np.int32)
    token_ids_cpu_tensor = torch.zeros(
        (max_num_reqs, max_model_len),
        device="cpu",
        dtype=torch.int32,
    )
    mock_runner.input_batch.token_ids_cpu_tensor = token_ids_cpu_tensor
    mock_runner.input_batch.token_ids_cpu = token_ids_cpu_tensor.numpy()
    return mock_runner


# yapf: disable
@pytest.mark.parametrize(
    "req_ids, num_computed_tokens," \
    "token_ids_tensor_list," \
    "num_reqs, total_num_scheduled_tokens, num_scheduled_tokens," \
    "target_input_ids_pcp_full, target_query_start_loc_pcp_full",
    [
        # prefill
        (
            ['0'], np.array([0]),
            [torch.tensor([0, 671, 6102, 294, 8760, 344])],
            1, 6, {'0': 6},
            torch.tensor([0, 671, 6102, 294, 8760, 344]),
            torch.tensor([0, 6])
        ),
        # decode
        (
            ['0'], np.array([6]),
            [torch.tensor([0, 671, 6102, 294, 8760, 344, 88907, 0])],
            1, 2, {'0': 2},
            torch.tensor([88907, 0]),
            torch.tensor([0, 2])
        ),
        # decode + prefill
        (
            ['0', '1'], np.array([6, 0]),
            [
                torch.tensor([0, 671, 6102, 294, 8760, 344, 88907, 0]),
                torch.tensor([0, 19923, 14, 1026, 2329, 344, 9807, 14, 342, 1030]),
            ],
            2, 12, {'0': 2, '1': 10},
            torch.tensor([88907, 0, 0, 19923, 14, 1026, 2329, 344, 9807, 14, 342, 1030]),
            torch.tensor([0, 2, 12])
        ),
        # decodes + prefills
        (
            ['0', '1', '2', '3'], np.array([6, 8, 0, 0]),
            [
                torch.tensor([0, 671, 6102, 294, 8760, 344, 88907, 0]),
                torch.tensor([0, 19923, 14, 1026, 2329, 344, 9807, 14, 342, 0]),
                torch.tensor([0, 671, 8749, 294, 3702, 4106, 344, 88907]),
                torch.tensor([0, 671, 5335, 1469, 7539, 305, 6397]),
            ],
            4, 19, {'0': 2, '1': 2, '2': 8, '3': 7},
            torch.tensor([88907, 0, 342, 0, 0, 671, 8749, 294, 3702, 4106, 344, 88907,
                          0, 671, 5335, 1469, 7539, 305, 6397]),
            torch.tensor([0, 2, 4, 12, 19])
        ),
    ])
# yapf: enable
def test_generate_pcp_mtp_input(
    pcp_mtp_mock_runner,
    req_ids,
    num_computed_tokens,
    token_ids_tensor_list,
    num_reqs,
    total_num_scheduled_tokens,
    num_scheduled_tokens,
    target_input_ids_pcp_full,
    target_query_start_loc_pcp_full,
):
    mock_runner = pcp_mtp_mock_runner
    token_ids_cpu_tensor = mock_runner.input_batch.token_ids_cpu_tensor

    # Set input_batch
    mock_runner.input_batch.req_ids = req_ids
    mock_runner.input_batch.num_computed_tokens_cpu[:num_computed_tokens.
                                                    size] = num_computed_tokens
    for i, token_ids_tensor in enumerate(token_ids_tensor_list):
        token_ids_cpu_tensor[i][:token_ids_tensor.size(0)] = token_ids_tensor

    NPUModelRunner._generate_pcp_mtp_input(mock_runner, num_reqs,
                                           total_num_scheduled_tokens,
                                           num_scheduled_tokens)
    assert torch.equal(
        mock_runner.input_ids_pcp_full.cpu[:total_num_scheduled_tokens],
        target_input_ids_pcp_full)
    assert torch.equal(mock_runner.query_start_loc_pcp_full.cpu[:num_reqs + 1],
                       target_query_start_loc_pcp_full)


@pytest.mark.parametrize(
    "pcp_rank, split_with_q_head_nomask_idx_reqs, split_kv_with_q_tail_nomask_idx_reqs,"
    "head_attn_nomask_seqlens, chunk_seqlens,"
    "target_split_q_head, target_split_q_tail, target_head_seqlens, target_tail_seqlens",
    [
        # case1: pcp_rank=0
        (0, [[10, 20, 30]], [[40, 50, 60]],
         torch.tensor([[64], [0]], dtype=torch.int32), [64], [
             torch.tensor([1, 2, 3], dtype=torch.int32)
         ], [torch.tensor([40, 50, 60], dtype=torch.int32)], [
             torch.tensor([[64], [0]], dtype=torch.int32)
         ], [torch.tensor([[64], [3]], dtype=torch.int32)]),
        # case2: pcp_rank=1
        (1, [[1, 2], [3, 4, 5]], [[6, 7], [8, 9, 10]],
         torch.tensor([[128, 128], [128, 128]], dtype=torch.int32), [128, 128],
         [torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)], [
             torch.tensor([6, 7, 8, 9, 10], dtype=torch.int32)
         ], [torch.tensor([[128, 128], [2, 3]], dtype=torch.int32)
             ], [torch.tensor([[128, 128], [2, 3]], dtype=torch.int32)]),
        # case3: pcp_rank=2
        (2, [[11, 12, 13, 14], [15, 16]], [[17, 18, 19], [20, 21, 22, 23]],
         torch.tensor([[256, 256], [512, 512]], dtype=torch.int32), [256, 256],
         [torch.tensor([11, 12, 13, 14, 15, 16], dtype=torch.int32)], [
             torch.tensor([17, 18, 19, 20, 21, 22, 23], dtype=torch.int32)
         ], [torch.tensor([[256, 256], [4, 2]], dtype=torch.int32)
             ], [torch.tensor([[256, 256], [3, 4]], dtype=torch.int32)]),
        # case4: empty input
        (
            0,
            [],
            [],
            torch.tensor([], dtype=torch.int32).reshape(2, 0),
            [],
            [],
            [],
            [],
            [],
        ),
        # case5: single element input
        (
            0,
            [[10]],
            [[40]],
            torch.tensor([[64], [0]], dtype=torch.int32),
            [64],
            [torch.tensor([1, 2, 3], dtype=torch.int32)],
            [torch.tensor([40], dtype=torch.int32)],
            [torch.tensor([[64], [0]], dtype=torch.int32)],
            [torch.tensor([[64], [1]], dtype=torch.int32)],
        ),
        # case6: pcp_rank=3
        (
            3,
            [[1, 2], [3, 4, 5]],
            [[6, 7], [8, 9, 10]],
            torch.tensor([[128, 128], [128, 128]], dtype=torch.int32),
            [128, 128],
            [torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)],
            [torch.tensor([6, 7, 8, 9, 10], dtype=torch.int32)],
            [torch.tensor([[128, 128], [2, 3]], dtype=torch.int32)],
            [torch.tensor([[128, 128], [2, 3]], dtype=torch.int32)],
        ),
    ])
def test_split_nomask_idx_tensor_list(
        pcp_rank, split_with_q_head_nomask_idx_reqs,
        split_kv_with_q_tail_nomask_idx_reqs, head_attn_nomask_seqlens,
        chunk_seqlens, target_split_q_head, target_split_q_tail,
        target_head_seqlens, target_tail_seqlens):
    # Mock input data
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.device = "cpu"
    mock_runner.pcp_rank = 0
    mock_runner.kv_idx_names = {
        "kv_with_q_head_nomask_idx_tensor":
        torch.tensor([1, 2, 3], dtype=torch.int32)
    }

    mock_runner.pcp_rank = pcp_rank

    # Mock output
    mock_runner._split_multi_batch_kv_idx.side_effect = NPUModelRunner._split_multi_batch_kv_idx.__get__(
        mock_runner, NPUModelRunner)
    mock_runner._list_to_tensor.side_effect = NPUModelRunner._list_to_tensor.__get__(
        mock_runner, NPUModelRunner)

    # Call the method under test
    result = NPUModelRunner._split_nomask_idx_tensor_list(
        mock_runner,
        split_with_q_head_nomask_idx_reqs=split_with_q_head_nomask_idx_reqs,
        split_kv_with_q_tail_nomask_idx_reqs=
        split_kv_with_q_tail_nomask_idx_reqs,
        head_attn_nomask_seqlens=head_attn_nomask_seqlens,
        chunk_seqlens=chunk_seqlens)
    split_q_head, split_q_tail, head_seqlens, tail_seqlens = result

    # Assert the method call
    assert len(split_q_head) == len(target_split_q_head)
    for res, target in zip(split_q_head, target_split_q_head):
        assert torch.equal(res, target)

    assert len(split_q_tail) == len(target_split_q_tail)
    for res, target in zip(split_q_tail, target_split_q_tail):
        assert torch.equal(res, target)

    assert len(head_seqlens) == len(target_head_seqlens)
    for res, target in zip(head_seqlens, target_head_seqlens):
        if isinstance(target, torch.Tensor):
            assert torch.equal(res, target)
        else:
            assert res == target

    assert len(tail_seqlens) == len(target_tail_seqlens)
    for res, target in zip(tail_seqlens, target_tail_seqlens):
        if isinstance(target, torch.Tensor):
            assert torch.equal(res, target)
        else:
            assert res == target


@pytest.mark.parametrize(
    "kv_nomask_idx_multi_batch, split_size, expected_merged_idx, expected_merged_len",
    [
        # case1: multiple batches + split size greater than batch length
        (
            [[0, 1, 2, 3, 4], [5, 6, 7]],
            2,
            # expected  merged_split_kv_idx_3d
            [[0, 1, 5, 6], [2, 3, 7], [4]],
            # expected merged_split_kv_len_2d
            [[2, 2], [2, 1], [1, 0]],
        ),
        # case2: single batch + split size greater than batch length
        (
            [[0, 1, 2]],
            5,
            [[0, 1, 2]],
            [[3]],
        ),
        # case3: split size equals maximum batch length
        (
            [[0, 1, 2, 3], [5, 6]],
            4,
            [[0, 1, 2, 3, 5, 6]],
            [[4, 2]],
        ),
        # case4: Split size is 1 (minimum granularity split)
        (
            [[0, 1], [2]],
            1,
            [[0, 2], [1]],
            [[1, 1], [1, 0]],
        ),
        # case6: the batch contains an empty list
        (
            [[], [0, 1], [2]],
            1,
            [[0, 2], [1]],
            [[0, 1, 1], [0, 1, 0]],
        ),
        # case: empty input
        (
            [],
            2,
            [],
            [],
        ),
    ])
def test_split_multi_batch_kv_idx(
    kv_nomask_idx_multi_batch,
    split_size,
    expected_merged_idx,
    expected_merged_len,
):
    # Mock input data
    model_runner = MagicMock(spec=NPUModelRunner)

    # Call the method under test
    result = NPUModelRunner._split_multi_batch_kv_idx(
        self=model_runner,
        kv_nomask_idx_multi_batch=kv_nomask_idx_multi_batch,
        split_size=split_size)

    merged_split_kv_idx_3d, merged_split_kv_len_2d = result

    # Assert the method call
    assert len(merged_split_kv_idx_3d) == len(expected_merged_idx)

    for t, (actual_seg, expected_seg) in enumerate(
            zip(merged_split_kv_idx_3d, expected_merged_idx)):
        assert actual_seg == expected_seg

    assert len(merged_split_kv_len_2d) == len(expected_merged_len)

    for t, (actual_len, expected_len) in enumerate(
            zip(merged_split_kv_len_2d, expected_merged_len)):
        assert actual_len == expected_len
