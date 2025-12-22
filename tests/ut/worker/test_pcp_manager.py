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

from vllm_ascend.worker.pcp_utils import PCPManager


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
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.use_mla = use_mla
    vllm_config.parallel_config.cp_kv_cache_interleave_size = 64
    vllm_config.speculative_config.num_speculative_tokens = 0

    pcp_manager = PCPManager(pcp_world_size=pcp_size,
                             pcp_rank=0,
                             dcp_world_size=dcp_size,
                             dcp_rank=0,
                             max_buffer_num_tokens=10000,
                             max_num_reqs=1000,
                             device="cpu",
                             vllm_config=vllm_config,
                             pin_memory=False)
    input_batch = MagicMock()
    input_batch.num_reqs = num_reqs

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

    input_batch.num_computed_tokens_cpu = torch.tensor(num_computed_tokens)
    input_batch.num_prompt_tokens = torch.tensor(num_prompt_tokens)
    input_batch.num_tokens = torch.tensor(num_tokens)

    query_lens = torch.tensor(query_lens)
    result = pcp_manager.generate_pcp_metadata(total_tokens, query_lens, None,
                                               input_batch)

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


@pytest.mark.parametrize(
    "tokens, num_reqs, num_computed_tokens, num_prompt_tokens, pcp_size, pcp_rank, expected_pcp_tokens",
    [
        # Case 1: prefill only
        ([8, 12, 16], 3, [0, 0, 0], [8, 12, 16], 4, 0, [2, 4, 4]),

        # # Case 2: mix prefill and decode
        ([8, 4, 12], 3, [8, 4, 0], [8, 0, 12], 4, 0, [2, 2, 4]),

        # # Case 3: request which need to be padded
        ([3, 7, 9], 3, [0, 0, 0], [3, 7, 9], 4, 0, [2, 2, 4]),

        # Case 4: single request
        ([10], 1, [0], [10], 4, 0, [4]),
    ])
def test_update_tokens_for_pcp_basic(tokens, num_reqs, num_computed_tokens,
                                     num_prompt_tokens, pcp_size, pcp_rank,
                                     expected_pcp_tokens):
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.speculative_config.num_speculative_tokens = 0

    pcp_manager = PCPManager(pcp_world_size=pcp_size,
                             pcp_rank=0,
                             dcp_world_size=1,
                             dcp_rank=0,
                             max_buffer_num_tokens=10000,
                             max_num_reqs=1000,
                             device="cpu",
                             vllm_config=vllm_config,
                             pin_memory=False)
    input_batch = MagicMock()
    input_batch.num_reqs = num_reqs
    input_batch.num_computed_tokens_cpu = np.array(num_computed_tokens,
                                                   dtype=np.int32)
    input_batch.num_prompt_tokens = np.array(num_prompt_tokens, dtype=np.int32)
    arange_np = np.arange(10000)
    pcp_tokens_result, positions_result = pcp_manager.update_tokens_for_pcp(
        np.array(tokens), arange_np, num_reqs, 1)

    assert np.array_equal(pcp_tokens_result, expected_pcp_tokens), \
        f"Expected pcp_tokens: {expected_pcp_tokens}, got: {pcp_tokens_result}"

    total_pcp_tokens: int = np.sum(pcp_tokens_result)
    assert positions_result.shape == (total_pcp_tokens,), \
        f"Positions shape mismatch. Expected length {total_pcp_tokens}, got {positions_result.shape}"


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
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.speculative_config.num_speculative_tokens = 0
    pcp_manager = PCPManager(pcp_world_size=pcp_world_size,
                             pcp_rank=0,
                             dcp_world_size=dcp_world_size,
                             dcp_rank=0,
                             max_buffer_num_tokens=10000,
                             max_num_reqs=1000,
                             device="cpu",
                             vllm_config=vllm_config,
                             pin_memory=False)
    ret = pcp_manager._get_cp_local_seq_lens(seq_lens, pcp_world_size,
                                             dcp_world_size,
                                             cp_kv_cache_interleave_size)
    assert torch.equal(ret, target)


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
    req_ids,
    num_computed_tokens,
    token_ids_tensor_list,
    num_reqs,
    total_num_scheduled_tokens,
    num_scheduled_tokens,
    target_input_ids_pcp_full,
    target_query_start_loc_pcp_full,
):
    max_num_reqs = 4
    max_model_len = 4096
    max_num_tokens = 4096
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.speculative_config.num_speculative_tokens = 1
    vllm_config.scheduler_config.max_num_seqs = max_num_reqs
    vllm_config.scheduler_config.max_num_batched_tokens = max_model_len
    pcp_manager = PCPManager(pcp_world_size=2,
                             pcp_rank=0,
                             dcp_world_size=1,
                             dcp_rank=0,
                             max_buffer_num_tokens=max_num_tokens,
                             max_num_reqs=max_num_reqs,
                             device="cpu",
                             vllm_config=vllm_config,
                             pin_memory=False)
    arange_np = np.arange(max_model_len)
    input_batch = MagicMock()
    input_batch.num_computed_tokens_cpu = \
        np.zeros(max_num_reqs, dtype=np.int32)
    token_ids_cpu_tensor = torch.zeros(
        (max_num_reqs, max_model_len),
        device="cpu",
        dtype=torch.int32,
    )
    input_batch.token_ids_cpu_tensor = token_ids_cpu_tensor
    input_batch.token_ids_cpu = token_ids_cpu_tensor.numpy()
    token_ids_cpu_tensor = input_batch.token_ids_cpu_tensor

    # Set input_batch
    input_batch.req_ids = req_ids
    input_batch.num_computed_tokens_cpu[:num_computed_tokens.
                                        size] = num_computed_tokens
    for i, token_ids_tensor in enumerate(token_ids_tensor_list):
        token_ids_cpu_tensor[i][:token_ids_tensor.size(0)] = token_ids_tensor

    pcp_manager.generate_pcp_mtp_input(num_reqs, total_num_scheduled_tokens,
                                       num_scheduled_tokens, input_batch,
                                       arange_np)
    assert torch.equal(
        pcp_manager.input_ids_pcp_full.cpu[:total_num_scheduled_tokens],
        target_input_ids_pcp_full)
    assert torch.equal(pcp_manager.query_start_loc_pcp_full.cpu[:num_reqs + 1],
                       target_query_start_loc_pcp_full)
