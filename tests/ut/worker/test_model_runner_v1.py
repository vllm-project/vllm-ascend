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

import numpy as np
import pytest
import torch
from vllm.config import (CacheConfig, CUDAGraphMode, ModelConfig,
                         ParallelConfig, VllmConfig)
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase
from vllm_ascend.utils import AscendSocVersion
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunner(TestBase):

    def setUp(self):
        """Setup test environment"""
        # Create configuration mocks
        self.cache_config_mock = MagicMock(spec=CacheConfig)
        self.cache_config_mock.cache_dtype = "auto"

        self.model_config_mock = MagicMock(spec=ModelConfig)
        self.model_config_mock.dtype = torch.float16
        self.model_config_mock.trust_remote_code = False
        self.model_config_mock.pooler_config = None
        self.model_config_mock.get_hidden_size.return_value = 7168
        self.model_config_mock.is_multimodal_model = False
        self.model_config_mock.max_model_len = 128
        self.model_config_mock.uses_mrope = False

        self.parallel_config_mock = MagicMock(spec=ParallelConfig)
        self.parallel_config_mock.enable_sequence_parallel = 1

        self.scheduler_output_mock = MagicMock()
        self.scheduler_output_mock.num_scheduled_tokens = {
            0: 2,
            1: 2,
            2: 2,
            3: 2
        }
        self.scheduler_output_mock.max_num_batched_tokens = 10
        self.scheduler_output_mock.max_num_seqs = 4
        self.scheduler_output_mock.total_num_scheduled_tokens = 8

        self.vllm_config_mock = MagicMock(spec=VllmConfig)
        self.vllm_config_mock.cache_config = self.cache_config_mock
        self.vllm_config_mock.model_config = self.model_config_mock
        self.vllm_config_mock.parallel_config = self.parallel_config_mock
        self.vllm_config_mock.additional_config = None
        self.vllm_config_mock.load_config = None
        self.vllm_config_mock.scheduler_config = self.scheduler_output_mock
        self.vllm_config_mock.device_config = None
        self.vllm_config_mock.compilation_config = MagicMock()
        self.vllm_config_mock.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        self.vllm_config_mock.speculative_config = None
        self.vllm_config_mock.lora_config = None

        self.device = None

        self.local_rank = 0
        self.rank = 0
        self.distributed_init_method = "tcp://localhost:12345"
        self.is_driver_worker = False

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable",
           return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_attn_backend")
    @patch("vllm_ascend.worker.model_runner_v1.get_ascend_config")
    @patch(
        "vllm_ascend.worker.model_runner_v1.get_tensor_model_parallel_world_size",
        return_value=2)
    @patch(
        "vllm_ascend.worker.model_runner_v1.get_context_model_parallel_world_size",
        return_value=2)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    @patch("vllm_ascend.worker.model_runner_v1.get_tp_group")
    @patch("vllm_ascend.worker.model_runner_v1.get_cp_group")
    @patch('vllm.distributed.parallel_state._TP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.parallel_state._CP",
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def test_prepare_inputs_when_sp_cp(
        self,
        mock_cp,
        mock_tp,
        mock_get_cp_group,
        mock_get_tp_group,
        mock_get_pp_group,
        mock_cp_world_size,
        mock_tp_world_size,
        mock_get_ascend_config,
        mock_get_attn_backend,
        mock_lmhead_tp_enable,
    ):
        """Test NPURunner normal initialization"""
        # Import and create NPUWorker instance
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        mock_cp_group = MagicMock()
        mock_cp_group.rank_in_group = 0
        mock_cp_group.world_size = 2
        mock_get_cp_group.return_value = mock_cp_group
        mock_cp.rank_in_group = 0

        mock_tp_group = MagicMock()
        mock_tp_group.rank_in_group = 0
        mock_tp_group.world_size = 2
        mock_get_tp_group.return_value = mock_tp_group
        mock_tp.rank_in_group = 0
        mock_tp.world_size = 2

        mock_pp_group = MagicMock()
        mock_pp_group.rank_in_group = 0
        mock_get_pp_group.return_value = mock_pp_group

        mock_get_ascend_config.return_value = MagicMock()

        mock_get_attn_backend.return_value = AscendAttentionBackend
        mock_get_attn_backend.lmhead_tensor_parallel_size.return_value = False

        runner = NPUModelRunner(
            vllm_config=self.vllm_config_mock,
            device=self.device,
        )

        runner._update_states = MagicMock()
        runner.input_batch = MagicMock()
        runner.input_batch.num_reqs = 4
        runner.input_batch.req_ids = [0, 1, 2, 3]
        runner.input_batch.num_computed_tokens_cpu = np.array([0, 0, 0, 0])
        runner.input_batch.num_computed_tokens_of_cp_sp = np.zeros((4, 2, 2),
                                                                   dtype=int)
        runner.input_batch.token_ids_cpu_tensor = torch.zeros(
            (4, 10),
            device="cpu",
            dtype=torch.int32,
        )
        runner.input_batch.token_ids_cpu = runner.input_batch.token_ids_cpu_tensor.numpy(
        )
        runner.model = MagicMock()
        output = runner._prepare_inputs(self.scheduler_output_mock, None)
        self.assertIsNotNone(output)


# yapf: disable
@pytest.mark.parametrize(
    "soc_version, enable_expert_parallel, world_size, num_tokens, mc2_tokens_capacity, expected_method",
    [
        # Case 1: Expert parallel is disabled, should always be 'allgather'
        (AscendSocVersion.A2, False, 8, 100, 256, "allgather"),
        (AscendSocVersion.A3, False, 16, 500, 256, "allgather"),

        # Case 2: A2 SOC
        # 2.1: MC2 conditions met (tokens <= capacity, world_size >= 16)
        (AscendSocVersion.A2, True, 16, 100, 256, "mc2"),
        (AscendSocVersion.A2, True, 32, 256, 256, "mc2"),
        # 2.2: MC2 token capacity exceeded
        (AscendSocVersion.A2, True, 16, 257, 256, "allgather"),
        # 2.3: MC2 world size not met
        (AscendSocVersion.A2, True, 8, 100, 256, "allgather"),
        (AscendSocVersion.A2, True, 15, 100, 256, "allgather"),

        # Case 3: A3 SOC
        # 3.1: MC2 condition met (tokens <= capacity)
        (AscendSocVersion.A3, True, 8, 100, 256, "mc2"),
        (AscendSocVersion.A3, True, 16, 256, 256, "mc2"),
        # 3.2: MC2 token capacity exceeded
        (AscendSocVersion.A3, True, 8, 257, 256, "alltoall"),
        (AscendSocVersion.A3, True, 16, 500, 256, "alltoall"),

    ])
# yapf: enable
def test_select_moe_comm_method(soc_version, enable_expert_parallel,
                                world_size, num_tokens, mc2_tokens_capacity,
                                expected_method):
    """
    Tests the _select_moe_comm_method with various configurations.
    """
    # Mock the NPUModelRunner instance and its dependencies
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.parallel_config = MagicMock()
    mock_runner.parallel_config.enable_expert_parallel = enable_expert_parallel
    mock_runner.parallel_config.world_size_across_dp = world_size
    mock_runner.mc2_tokens_capacity = mc2_tokens_capacity

    # Patch the helper functions
    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_soc_version',
               return_value=soc_version), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True):

        # Call the method under test
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

    unsupported_soc = "UnsupportedSOC"

    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_soc_version',
               return_value=unsupported_soc), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True), \
         pytest.raises(ValueError, match=f"Unsupported soc_version: {unsupported_soc}"):

        NPUModelRunner._select_moe_comm_method(mock_runner, 100)
