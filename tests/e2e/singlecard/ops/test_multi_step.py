# Copyright (c) China Merchants Bank Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#/

# to run this test, you need to cd to the upper package which is 'tests',
# and run with command 'pytest -s ops/test_multi_step.py'

import torch
import torch_npu  # noqa: F401

DTYPES = [torch.int32, torch.int64]
DEVICES = [f"npu:{0}"]
# Set tolerance to 0 for equals
DEFAULT_ATOL = 0
DEFAULT_RTOL = 0

# test custom ops of https://github.com/vllm-project/vllm-ascend/tree/main/csrc/kernels/advance_step.cpp


@torch.inference_mode()
def test_single_generation_multi_step() -> None:
    input_tokens_data = [2926]
    input_tokens_ascendc = torch.tensor(input_tokens_data, device='npu:0')
    input_tokens_python = torch.tensor(input_tokens_data, device='npu:0')

    sampled_token_ids_data = [[13]]
    sampled_token_ids = torch.tensor(sampled_token_ids_data, device='npu:0')

    input_positions_data = [5]
    input_positions_ascendc = torch.tensor(input_positions_data,
                                           device='npu:0')
    input_positions_python = torch.tensor(input_positions_data, device='npu:0')

    seq_lens_data = [6]
    seq_lens_ascendc = torch.tensor(seq_lens_data,
                                    device='npu:0',
                                    dtype=torch.int32)
    seq_lens_python = torch.tensor(seq_lens_data,
                                   device='npu:0',
                                   dtype=torch.int32)

    slot_mapping_data = [5]
    slot_mapping_ascendc = torch.tensor(slot_mapping_data,
                                        device='npu:0',
                                        dtype=torch.int32)
    slot_mapping_python = torch.tensor(slot_mapping_data,
                                       device='npu:0',
                                       dtype=torch.int32)

    block_tables_data = [[0]]

    block_tables = torch.tensor(block_tables_data,
                                device='npu:0',
                                dtype=torch.int32)

    torch.ops._C.advance_step_flashattn_ascendc(
        1, 1, 128, input_tokens_ascendc, sampled_token_ids,
        input_positions_ascendc, seq_lens_ascendc, slot_mapping_ascendc,
        block_tables)

    normal(1, 1, 128, input_tokens_python, sampled_token_ids,
           input_positions_python, seq_lens_python, slot_mapping_python,
           block_tables)

    # Compare the results.
    torch.testing.assert_close(input_tokens_ascendc,
                               input_tokens_python,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(input_positions_ascendc,
                               input_positions_python,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(seq_lens_ascendc,
                               seq_lens_python,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(slot_mapping_ascendc,
                               slot_mapping_python,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)


@torch.inference_mode()
def test_multi_result_generation_multi_step() -> None:
    input_tokens_data = [2926, 279, 12095, 1588]
    input_tokens_ascendc = torch.tensor(input_tokens_data, device='npu:0')
    input_tokens_python = torch.tensor(input_tokens_data, device='npu:0')

    sampled_token_ids_data = [[13], [1968], [13], [13]]
    sampled_token_ids = torch.tensor(sampled_token_ids_data, device='npu:0')

    input_positions_data = [5, 7, 5, 5]
    input_positions_ascendc = torch.tensor(input_positions_data,
                                           device='npu:0')
    input_positions_python = torch.tensor(input_positions_data, device='npu:0')

    seq_lens_data = [6, 8, 6, 6]
    seq_lens_ascendc = torch.tensor(seq_lens_data,
                                    device='npu:0',
                                    dtype=torch.int32)
    seq_lens_python = torch.tensor(seq_lens_data,
                                   device='npu:0',
                                   dtype=torch.int32)

    slot_mapping_data = [5, 135, 261, 389]
    slot_mapping_ascendc = torch.tensor(slot_mapping_data,
                                        device='npu:0',
                                        dtype=torch.int32)
    slot_mapping_python = torch.tensor(slot_mapping_data,
                                       device='npu:0',
                                       dtype=torch.int32)

    block_tables_data = [[0], [1], [2], [3]]

    block_tables = torch.tensor(block_tables_data,
                                device='npu:0',
                                dtype=torch.int32)

    torch.ops._C.advance_step_flashattn_ascendc(
        4, 4, 128, input_tokens_ascendc, sampled_token_ids,
        input_positions_ascendc, seq_lens_ascendc, slot_mapping_ascendc,
        block_tables)

    normal(4, 4, 128, input_tokens_python, sampled_token_ids,
           input_positions_python, seq_lens_python, slot_mapping_python,
           block_tables)

    # Compare the results.
    torch.testing.assert_close(input_tokens_ascendc,
                               input_tokens_python,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(input_positions_ascendc,
                               input_positions_python,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(seq_lens_ascendc,
                               seq_lens_python,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(slot_mapping_ascendc,
                               slot_mapping_python,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)


def normal(num_seqs: int, num_queries: int, block_size: int,
           input_tokens: torch.Tensor, sampled_token_ids: torch.Tensor,
           input_positions: torch.Tensor, seq_lens_tensor: torch.Tensor,
           slot_mapping: torch.Tensor, block_tables: torch.Tensor) -> None:
    sampled_token_ids_list = sampled_token_ids[:num_queries].squeeze(-1)
    input_tokens[:num_queries] = sampled_token_ids_list

    # get seq_lens and input_positions
    seq_lens = seq_lens_tensor[:num_queries]
    next_seq_lens = seq_lens + 1
    next_input_pos = next_seq_lens - 1

    # update seq_lens and input_positions
    seq_lens_tensor[:num_queries] = next_seq_lens
    input_positions[:num_queries] = next_input_pos  # type: ignore

    # get block index and offset
    block_idx = next_input_pos // block_size
    block_offset = next_input_pos % block_size

    current_block_table = block_tables.gather(
        1, block_idx.unsqueeze(-1)).squeeze(-1)
    slot_num = current_block_table * block_size + block_offset

    # update slot_mapping
    slot_mapping[:num_queries] = slot_num
