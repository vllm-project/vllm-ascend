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
from vllm.config import CUDAGraphMode

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import AscendSocVersion
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

# Check if CompilationLevel is available (v0.11.0+)
try:
    from vllm.config import CompilationLevel
    HAS_COMPILATION_LEVEL = True
except ImportError:
    HAS_COMPILATION_LEVEL = False

# Check if CompilationMode is available (pre-v0.11.0)
try:
    from vllm.config import CompilationMode
    HAS_COMPILATION_MODE = True
except ImportError:
    HAS_COMPILATION_MODE = False


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


def test_get_cumsum_and_arange():
    """
    Tests the _get_cumsum_and_arange method with various input arrays.
    """
    mock_runner = MagicMock(spec=NPUModelRunner)
    # Initialize arange_np with a large enough size
    mock_runner.arange_np = np.arange(100, dtype=np.int32)

    # Test case 1: [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
    num_tokens = np.array([2, 5, 3], dtype=np.int32)
    cu_num_tokens, arange = NPUModelRunner._get_cumsum_and_arange(
        mock_runner, num_tokens)
    assert np.array_equal(cu_num_tokens, np.array([2, 7, 10]))
    assert np.array_equal(arange, np.array([0, 1, 0, 1, 2, 3, 4, 0, 1, 2]))

    # Test case 2: Single element
    num_tokens = np.array([5], dtype=np.int32)
    cu_num_tokens, arange = NPUModelRunner._get_cumsum_and_arange(
        mock_runner, num_tokens)
    assert np.array_equal(cu_num_tokens, np.array([5]))
    assert np.array_equal(arange, np.array([0, 1, 2, 3, 4]))

    # Test case 3: Empty array - should raise IndexError or handle gracefully
    num_tokens = np.array([], dtype=np.int32)
    # Empty array will cause IndexError when accessing cu_num_tokens[-1]
    # This is expected behavior as the method assumes non-empty input
    with pytest.raises(IndexError):
        NPUModelRunner._get_cumsum_and_arange(mock_runner, num_tokens)


@pytest.mark.parametrize(
    "seq_lens_np, num_scheduled_tokens, num_valid_tokens, speculative_config, chunked_prefill_enabled, ascend_scheduler_enabled, expected_state",
    [
        # Case 1: PrefillNoCache - seq_lens equals num_scheduled_tokens
        (np.array([5, 10, 8]), np.array([5, 10, 8]), np.array([5, 10, 8]),
         None, False, False, AscendAttentionState.PrefillNoCache),
        # Case 2: DecodeOnly - all num_scheduled_tokens == 1
        (np.array([5, 10, 8]), np.array([1, 1, 1]), np.array(
            [1, 1, 1]), None, False, False, AscendAttentionState.DecodeOnly),
        # Case 3: SpecDecoding - all num_scheduled_tokens == 1 with deepseek_mtp
        (np.array([5, 10, 8]), np.array([1, 1, 1]), np.array(
            [1, 1, 1]), MagicMock(method='deepseek_mtp'), False, False,
         AscendAttentionState.SpecDecoding),
        # Case 4: SpecDecoding - all num_valid_tokens == 1 with deepseek_mtp
        (np.array([5, 10, 8]), np.array([2, 2, 2]), np.array(
            [1, 1, 1]), MagicMock(method='deepseek_mtp'), False, False,
         AscendAttentionState.SpecDecoding),
        # Case 5: ChunkedPrefill - all num_valid_tokens == 1 without deepseek_mtp
        (np.array([5, 10, 8]), np.array([2, 2, 2]), np.array(
            [1, 1, 1]), MagicMock(method='other'), False, False,
         AscendAttentionState.ChunkedPrefill),
        # Case 6: ChunkedPrefill - chunked_prefill_enabled
        (np.array([5, 10, 8]), np.array([3, 4, 5]), np.array([3, 4, 5]), None,
         True, False, AscendAttentionState.ChunkedPrefill),
        # Case 7: ChunkedPrefill - ascend_scheduler not enabled
        (np.array([5, 10, 8]), np.array([3, 4, 5]), np.array([3, 4, 5]), None,
         True, False, AscendAttentionState.ChunkedPrefill),
        # Case 8: PrefillCacheHit - default case
        (np.array([5, 10, 8]), np.array([3, 4, 5]), np.array([3, 4, 5]), None,
         False, True, AscendAttentionState.PrefillCacheHit),
    ])
def test_build_attn_state(seq_lens_np, num_scheduled_tokens, num_valid_tokens,
                          speculative_config, chunked_prefill_enabled,
                          ascend_scheduler_enabled, expected_state):
    """
    Tests the _build_attn_state method with various configurations.
    """
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.seq_lens_np = seq_lens_np
    mock_runner.speculative_config = speculative_config
    mock_runner.chunked_prefill_enabled = chunked_prefill_enabled

    mock_ascend_config = MagicMock()
    mock_ascend_config.ascend_scheduler_config.enabled = ascend_scheduler_enabled

    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_config',
               return_value=mock_ascend_config):
        result = NPUModelRunner._build_attn_state(mock_runner,
                                                  len(num_scheduled_tokens),
                                                  num_scheduled_tokens,
                                                  num_valid_tokens)
        assert result == expected_state


@pytest.mark.parametrize(
    "cudagraph_mode, use_piecewise_level, enforce_eager, vllm_version, expected",
    [
        # Case 1: v0.11.0 - should use aclgraph
        pytest.param(CUDAGraphMode.PIECEWISE,
                     True,
                     False,
                     "0.11.0",
                     True,
                     marks=pytest.mark.skipif(
                         not HAS_COMPILATION_LEVEL,
                         reason="CompilationLevel not available")),
        # Case 2: v0.11.0 - enforce_eager should disable aclgraph
        pytest.param(CUDAGraphMode.PIECEWISE,
                     True,
                     True,
                     "0.11.0",
                     False,
                     marks=pytest.mark.skipif(
                         not HAS_COMPILATION_LEVEL,
                         reason="CompilationLevel not available")),
        # Case 3: v0.11.0 - NONE mode should disable aclgraph
        pytest.param(CUDAGraphMode.NONE,
                     True,
                     False,
                     "0.11.0",
                     False,
                     marks=pytest.mark.skipif(
                         not HAS_COMPILATION_LEVEL,
                         reason="CompilationLevel not available")),
        # Case 4: v0.11.0 - wrong level should disable aclgraph
        pytest.param(CUDAGraphMode.PIECEWISE,
                     False,
                     False,
                     "0.11.0",
                     False,
                     marks=pytest.mark.skipif(
                         not HAS_COMPILATION_LEVEL,
                         reason="CompilationLevel not available")),
        # Case 5: non-v0.11.0 - should use aclgraph with VLLM_COMPILE
        pytest.param(CUDAGraphMode.PIECEWISE,
                     True,
                     False,
                     "0.10.0",
                     True,
                     marks=pytest.mark.skipif(
                         not HAS_COMPILATION_MODE,
                         reason="CompilationMode not available")),
        # Case 6: non-v0.11.0 - enforce_eager should disable aclgraph
        pytest.param(CUDAGraphMode.PIECEWISE,
                     True,
                     True,
                     "0.10.0",
                     False,
                     marks=pytest.mark.skipif(
                         not HAS_COMPILATION_MODE,
                         reason="CompilationMode not available")),
        # Case 7: non-v0.11.0 - NONE mode should disable aclgraph
        pytest.param(CUDAGraphMode.NONE,
                     True,
                     False,
                     "0.10.0",
                     False,
                     marks=pytest.mark.skipif(
                         not HAS_COMPILATION_MODE,
                         reason="CompilationMode not available")),
        # Case 8: non-v0.11.0 - wrong mode should disable aclgraph
        pytest.param(CUDAGraphMode.PIECEWISE,
                     False,
                     False,
                     "0.10.0",
                     False,
                     marks=pytest.mark.skipif(
                         not HAS_COMPILATION_MODE,
                         reason="CompilationMode not available")),
    ])
def test_use_aclgraph(cudagraph_mode, use_piecewise_level, enforce_eager,
                      vllm_version, expected):
    """
    Tests the _use_aclgraph method with various configurations.
    """
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_compilation_config = MagicMock()
    mock_compilation_config.cudagraph_mode = cudagraph_mode

    # Import the enum classes based on version and set the correct values
    if vllm_version == "0.11.0":
        # Set the mock to match CompilationLevel.PIECEWISE
        if use_piecewise_level:
            mock_compilation_config.level = CompilationLevel.PIECEWISE
        else:
            # Use a different value that won't match
            mock_compilation_config.level = MagicMock()
    else:
        # Set the mock to match CompilationMode.VLLM_COMPILE
        if use_piecewise_level:
            mock_compilation_config.mode = CompilationMode.VLLM_COMPILE
        else:
            # Use a different value that won't match
            mock_compilation_config.mode = MagicMock()

    mock_runner.compilation_config = mock_compilation_config
    mock_model_config = MagicMock()
    mock_model_config.enforce_eager = enforce_eager
    mock_runner.model_config = mock_model_config

    with patch('vllm_ascend.worker.model_runner_v1.vllm_version_is',
               return_value=(vllm_version == "0.11.0")):
        result = NPUModelRunner._use_aclgraph(mock_runner)
        assert result == expected


def test_sync_metadata_across_dp_single_rank():
    """
    Tests the _sync_metadata_across_dp method when dp_size == 1.
    """
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.dp_size = 1
    mock_runner.dp_rank = 0

    num_tokens = 100
    with_prefill = True

    result = NPUModelRunner._sync_metadata_across_dp(mock_runner, num_tokens,
                                                     with_prefill)
    max_tokens, num_tokens_after_padding, global_with_prefill = result

    assert max_tokens == num_tokens
    assert num_tokens_after_padding is None
    assert global_with_prefill == with_prefill


def test_get_split_computed_tokens_basic():
    """
    Tests the get_split_computed_tokens method with basic input.
    """
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.pcp_world_size = 2
    mock_runner.dcp_world_size = 2

    num_computed_tokens = np.array([10, 20], dtype=np.int32)
    request_ids = ["req1", "req2"]
    request_start_rank_dict = {}

    # Mock the functions from their actual import locations
    mock_pcp_group = MagicMock()
    mock_pcp_group.world_size = 2
    mock_dcp_group = MagicMock()
    mock_dcp_group.world_size = 2

    # Patch the functions that are conditionally imported
    # get_pcp_group is conditionally imported, so we need to patch it where it's used
    with patch('vllm_ascend.worker.model_runner_v1.prefill_context_parallel_enable',
               return_value=True), \
         patch('vllm_ascend.worker.model_runner_v1.get_pcp_group',
               return_value=mock_pcp_group, create=True), \
         patch('vllm_ascend.worker.model_runner_v1.get_dcp_group',
               return_value=mock_dcp_group):

        result = NPUModelRunner.get_split_computed_tokens(
            mock_runner,
            num_computed_tokens,
            request_ids,
            request_start_rank_dict,
            cp_kv_cache_interleave_size=1)

        # Should return a list with shape [num_requests][pcp_size][dcp_size]
        assert len(result) == 2
        assert len(result[0]) == 2  # pcp_size
        assert len(result[0][0]) == 2  # dcp_size
        # Check that tokens are distributed
        total_distributed = sum(
            sum(sum(result[i][j]) for j in range(2)) for i in range(2))
        assert total_distributed == 30  # 10 + 20
