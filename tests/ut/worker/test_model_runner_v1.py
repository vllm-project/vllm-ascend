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
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.platforms import current_platform

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import AscendSocVersion
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

DEVICE = current_platform.device_type
BLOCK_SIZE = 16


# yapf: disable
@pytest.mark.parametrize(
    "soc_version, enable_expert_parallel, world_size, num_tokens, mc2_tokens_capacity, quant_type, expected_method",
    [
        # Case 1: Expert parallel is disabled, should always be 'allgather'
        (AscendSocVersion.A2, False, 8, 100, 256, None, MoECommType.ALLGATHER),
        (AscendSocVersion.A3, False, 16, 500, 256, None, MoECommType.ALLGATHER),

        # Case 2: A2 SOC with w4a8_dynamic -> use alltoall when not mc2
        (AscendSocVersion.A2, True, 8, 100, 256, "w4a8_dynamic", MoECommType.ALLTOALL),
        (AscendSocVersion.A2, True, 16, 257, 256, "w4a8_dynamic", MoECommType.ALLTOALL),
        (AscendSocVersion.A2, True, 16, 100, 256, "w4a8_dynamic", MoECommType.MC2),  # meets mc2 condition

        # Case 3: A2 SOC without w4a8_dynamic -> fallback to allgather
        (AscendSocVersion.A2, True, 8, 100, 256, None, MoECommType.ALLGATHER),
        (AscendSocVersion.A2, True, 16, 257, 256, None, MoECommType.ALLGATHER),

        # Case 4: A3 SOC
        (AscendSocVersion.A3, True, 8, 100, 256, None, MoECommType.MC2),
        (AscendSocVersion.A3, True, 8, 257, 256, None, MoECommType.ALLTOALL),
    ])
# yapf: enable
def test_select_moe_comm_method(soc_version, enable_expert_parallel,
                                world_size, num_tokens, mc2_tokens_capacity,
                                quant_type, expected_method):
    """
    Tests the _select_moe_comm_method with various configurations including quant_type.
    """
    # Mock the NPUModelRunner instance and its dependencies
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.parallel_config = MagicMock()
    mock_runner.parallel_config.enable_expert_parallel = enable_expert_parallel
    mock_runner.parallel_config.world_size_across_dp = world_size
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
    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_soc_version',
               return_value=soc_version), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True), \
         patch('vllm_ascend.worker.model_runner_v1.is_moe_model',
               return_value=True):

        # Bind the real method to the mock object
        method = NPUModelRunner._select_moe_comm_method(
            mock_runner, num_tokens, False)

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

    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_soc_version',
               return_value=unsupported_soc), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True), \
         patch('vllm_ascend.worker.model_runner_v1.is_moe_model',
                  return_value=True), \
         pytest.raises(ValueError, match=f"Unsupported soc_version: {unsupported_soc}"):

        NPUModelRunner._select_moe_comm_method(mock_runner, 100, False)


def get_vllm_config():
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config

@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()
    return NPUModelRunner(vllm_config, DEVICE)

def test_update_config(model_runner):
    # Simple update
    model_runner.update_config({"load_config": {"load_format": "dummy"}})
    assert model_runner.load_config.load_format == "dummy"
    # Raise error on non-existing config
    with pytest.raises(AssertionError):
        model_runner.update_config({"do_not_exist_config": "dummy"})

def test_reload_weights_before_load_model(model_runner):
    with pytest.raises(AssertionError):
        model_runner.reload_weights()
