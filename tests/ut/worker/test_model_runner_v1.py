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

from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import AscendDeviceType
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


# yapf: disable
@pytest.mark.parametrize(
    "soc_version, enable_expert_parallel, world_size, pipeline_size, num_tokens, mc2_tokens_capacity, quant_type, expected_method",
    [
        # Case 1: Expert parallel is disabled, should always be 'allgather'
        (AscendDeviceType._910B, False, 8, 2, 100, 256, None, MoECommType.ALLGATHER),
        (AscendDeviceType._910_93, False, 16, 2, 500, 256, None, MoECommType.ALLGATHER),

        # Case 2: A2 SOC with w4a8_dynamic -> use alltoall when not mc2
        (AscendDeviceType._910B, True, 8, 1, 100, 256, "w4a8_dynamic", MoECommType.ALLTOALL),
        (AscendDeviceType._910B, True, 16, 1, 257, 256, "w4a8_dynamic", MoECommType.ALLTOALL),
        (AscendDeviceType._910B, True, 16, 1, 100, 256, "w4a8_dynamic", MoECommType.MC2),  # meets mc2 condition

        # Case 3: A2 SOC without w4a8_dynamic -> fallback to allgather
        (AscendDeviceType._910B, True, 8, 2, 100, 256, None, MoECommType.ALLGATHER),
        (AscendDeviceType._910B, True, 16, 2, 257, 256, None, MoECommType.ALLGATHER),

        # Case 4: A3 SOC
        (AscendDeviceType._910_93, True, 8, 2, 100, 256, None, MoECommType.MC2),
        (AscendDeviceType._910_93, True, 8, 2, 257, 256, None, MoECommType.ALLTOALL),
    ])
# yapf: enable
def test_select_moe_comm_method(soc_version, enable_expert_parallel,
                                world_size, pipeline_size, num_tokens,
                                mc2_tokens_capacity, quant_type,
                                expected_method):
    """
    Tests the _select_moe_comm_method with various configurations including quant_type.
    """
    # Mock the NPUModelRunner instance and its dependencies
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.parallel_config = MagicMock()
    mock_runner.parallel_config.enable_expert_parallel = enable_expert_parallel
    mock_runner.parallel_config.world_size_across_dp = world_size
    mock_runner.parallel_config.pipeline_parallel_size = pipeline_size
    mock_runner.mc2_tokens_capacity = mc2_tokens_capacity

    # Add vllm_config.model_config.hf_config mock with moe_quantize
    mock_hf_config = MagicMock()
    mock_hf_config.moe_quantize = quant_type
    mock_model_config = MagicMock()
    mock_model_config.hf_config = mock_hf_config
    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config = mock_model_config
    mock_runner.vllm_config = mock_vllm_config

    # Patch the helper functions
    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_device_type',
               return_value=soc_version), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True), \
         patch('vllm_ascend.worker.model_runner_v1.is_moe_model',
               return_value=True):

        # Bind the real method to the mock object
        method = NPUModelRunner._select_moe_comm_method(
            mock_runner, num_tokens)

        # Assert the result
        assert method == expected_method


def test_select_moe_comm_method_unsupported_soc():
    """
    Tests that _select_moe_comm_method raises ValueError for an unsupported SOC.
    """
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.parallel_config = MagicMock()
    mock_runner.parallel_config.enable_expert_parallel = True
    mock_runner.mc2_tokens_capacity = 256

    # Add vllm_config.model_config.hf_config mock with moe_quantize
    mock_hf_config = MagicMock()
    mock_hf_config.moe_quantize = None
    mock_model_config = MagicMock()
    mock_model_config.hf_config = mock_hf_config
    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config = mock_model_config
    mock_runner.vllm_config = mock_vllm_config

    unsupported_soc = "UnsupportedSOC"

    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_device_type',
               return_value=unsupported_soc), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True), \
         patch('vllm_ascend.worker.model_runner_v1.is_moe_model',
                  return_value=True), \
         pytest.raises(ValueError, match=f"Unsupported soc_version: {unsupported_soc}"):

        NPUModelRunner._select_moe_comm_method(mock_runner, 100)


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
def test_get_cp_local_seq_lens(
    seq_lens,
    pcp_world_size,
    dcp_world_size,
    cp_kv_cache_interleave_size,
    target,
):
    mock_runner = MagicMock(spec=NPUModelRunner)
    ret = NPUModelRunner._get_cp_local_seq_lens(
        mock_runner,
        seq_lens,
        pcp_world_size,
        dcp_world_size,
        cp_kv_cache_interleave_size
    )
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
    mock_runner = MagicMock(spec=NPUModelRunner)

    # Init model_runner pcp_mtp related buffers
    query_start_loc_buff = torch.zeros(max_num_reqs + 1,
                                  dtype=torch.int32,
                                  device="cpu",
                                  pin_memory=True)
    mock_runner.query_start_loc_pcp_full_cpu = query_start_loc_buff
    mock_runner.query_start_loc_pcp_full_np = query_start_loc_buff.numpy()
    mock_runner.query_start_loc_pcp_full = query_start_loc_buff.clone()

    positions_buff = torch.zeros(max_num_tokens,
                            dtype=torch.int64,
                            device="cpu",
                            pin_memory=True)
    mock_runner.positions_pcp_full = positions_buff
    mock_runner.positions_pcp_full_np = positions_buff.numpy()

    input_ids_buff = torch.zeros(max_num_tokens,
                            dtype=torch.int32,
                            device="cpu",
                            pin_memory=True)
    mock_runner.input_ids_pcp_full_cpu = input_ids_buff
    mock_runner.input_ids_pcp_full = input_ids_buff.clone()

    mock_runner.arange_np = np.arange(max_model_len)
    mock_runner.input_batch = MagicMock()
    mock_runner.input_batch.num_computed_tokens_cpu = \
        np.zeros(max_num_reqs, dtype=np.int32)
    token_ids_cpu_tensor = torch.zeros(
        (max_num_reqs, max_model_len),
        device="cpu",
        dtype=torch.int32,
        pin_memory=False,
    )
    mock_runner.input_batch.token_ids_cpu_tensor = token_ids_cpu_tensor
    mock_runner.input_batch.token_ids_cpu = token_ids_cpu_tensor.numpy()

    # Set input_batch
    mock_runner.input_batch.req_ids = req_ids
    mock_runner.input_batch.num_computed_tokens_cpu[
        :num_computed_tokens.size] = num_computed_tokens
    for i, token_ids_tensor in enumerate(token_ids_tensor_list):
        token_ids_cpu_tensor[i][:token_ids_tensor.size(0)] = token_ids_tensor

    NPUModelRunner._generate_pcp_mtp_input(
        mock_runner, num_reqs, total_num_scheduled_tokens, num_scheduled_tokens)
    assert torch.equal(mock_runner.input_ids_pcp_full[:total_num_scheduled_tokens],
                       target_input_ids_pcp_full)
    assert torch.equal(mock_runner.query_start_loc_pcp_full[:num_reqs + 1],
                       target_query_start_loc_pcp_full)

