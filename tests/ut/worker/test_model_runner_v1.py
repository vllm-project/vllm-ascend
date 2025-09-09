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

from vllm_ascend.utils import AscendSocVersion
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


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
    mock_runner.parallel_config.world_size = world_size
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
