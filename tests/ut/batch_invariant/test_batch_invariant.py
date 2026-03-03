# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# type: ignore
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

# Now import the module under test
import vllm_ascend.batch_invariant as batch_invariant


class TestBatchInvariant:
    """Complete test suite for batch_invariant.py"""

    def test_override_envs_for_invariance(self):
        """Test environment variable override"""
        # Clear environment variables
        env_vars = ["VLLM_ASCEND_ENABLE_NZ", "HCCL_DETERMINISTIC", "LCCL_DETERMINISTIC"]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

        # Call function
        batch_invariant.override_envs_for_invariance()

        # Verify environment variables
        assert os.environ["VLLM_ASCEND_ENABLE_NZ"] == "0"
        assert os.environ["HCCL_DETERMINISTIC"] == "strict"
        assert os.environ["LCCL_DETERMINISTIC"] == "1"

    @patch("vllm_ascend.batch_invariant.HAS_TRITON", False)
    @patch("vllm_ascend.batch_invariant.HAS_ASCENDC_BATCH_INVARIANT", True)
    def test_enable_batch_invariant_mode_ascendc_path(self):
        """Test enable_batch_invariant_mode with AscendC ops available"""
        # Mock dependencies
        mock_library = MagicMock()
        batch_invariant.torch.library.Library = MagicMock(return_value=mock_library)
        batch_invariant.torch.ops.batch_invariant_ops = MagicMock()

        # Call function
        batch_invariant.enable_batch_invariant_mode()

        # Verify library created
        batch_invariant.torch.library.Library.assert_called_once_with("aten", "IMPL")

        # Verify operator registrations
        assert mock_library.impl.call_count == 3
        mock_library.impl.assert_any_call(
            "aten::mm", batch_invariant.torch.ops.batch_invariant_ops.npu_mm_batch_invariant, "NPU"
        )
        mock_library.impl.assert_any_call(
            "aten::matmul", batch_invariant.torch.ops.batch_invariant_ops.npu_matmul_batch_invariant, "NPU"
        )
        mock_library.impl.assert_any_call(
            "aten::sum", batch_invariant.torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant, "NPU"
        )

        # Verify torch_npu function patching
        assert (
            batch_invariant.torch_npu.npu_fused_infer_attention_score
            == batch_invariant.torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant
        )

    @patch("vllm_ascend.batch_invariant.HAS_TRITON", True)
    @patch("vllm_ascend.batch_invariant.HAS_ASCENDC_BATCH_INVARIANT", False)
    def test_enable_batch_invariant_mode_triton_path(self):
        """Test enable_batch_invariant_mode with only Triton available"""
        # Mock dependencies
        mock_library = MagicMock()
        batch_invariant.torch.library.Library = MagicMock(return_value=mock_library)

        # Mock triton imports
        batch_invariant.addmm_batch_invariant = MagicMock()
        batch_invariant.bmm_batch_invariant = MagicMock()
        batch_invariant.mm_batch_invariant = MagicMock()
        batch_invariant.matmul_batch_invariant = MagicMock()
        batch_invariant.linear_batch_invariant = MagicMock()
        batch_invariant.softmax_batch_invariant = MagicMock()

        # Call function
        batch_invariant.enable_batch_invariant_mode()

        # Verify operator registrations
        assert mock_library.impl.call_count == 7
        mock_library.impl.assert_any_call("aten::addmm", batch_invariant.addmm_batch_invariant, "NPU")
        mock_library.impl.assert_any_call("aten::bmm", batch_invariant.bmm_batch_invariant, "NPU")
        mock_library.impl.assert_any_call("aten::mm", batch_invariant.mm_batch_invariant, "NPU")
        mock_library.impl.assert_any_call("aten::matmul", batch_invariant.matmul_batch_invariant, "NPU")
        mock_library.impl.assert_any_call("aten::linear", batch_invariant.linear_batch_invariant, "NPU")
        mock_library.impl.assert_any_call("aten::softmax", batch_invariant.softmax_batch_invariant, "NPU")
        mock_library.impl.assert_any_call("aten::_softmax", batch_invariant.softmax_batch_invariant, "NPU")

    @patch("vllm_ascend.batch_invariant.HAS_TRITON", False)
    @patch("vllm_ascend.batch_invariant.HAS_ASCENDC_BATCH_INVARIANT", False)
    def test_enable_batch_invariant_mode_no_backend(self):
        """Test enable_batch_invariant_mode with no backends available"""
        # Mock library
        mock_library = MagicMock()
        batch_invariant.torch.library.Library = MagicMock(return_value=mock_library)

        # Call function
        batch_invariant.enable_batch_invariant_mode()

        # Verify no operators registered
        mock_library.impl.assert_not_called()

    @pytest.mark.parametrize(
        "batch_invariant_enabled, has_backend, expected_logger_call",
        [(True, True, "info"), (True, False, "warning"), (False, True, None), (False, False, None)],
    )
    def test_init_batch_invariance(self, batch_invariant_enabled, has_backend, expected_logger_call):
        """Test init_batch_invariance under different conditions"""
        # Mock dependencies
        batch_invariant.vllm_is_batch_invariant = MagicMock(return_value=batch_invariant_enabled)
        batch_invariant.HAS_TRITON = has_backend
        batch_invariant.HAS_ASCENDC_BATCH_INVARIANT = has_backend
        batch_invariant.override_envs_for_invariance = MagicMock()
        batch_invariant.enable_batch_invariant_mode = MagicMock()

        # Call function
        batch_invariant.init_batch_invariance()

        # Verify function calls based on conditions
        if batch_invariant_enabled and has_backend:
            batch_invariant.override_envs_for_invariance.assert_called_once()
            batch_invariant.enable_batch_invariant_mode.assert_called_once()
        elif batch_invariant_enabled and not has_backend:
            batch_invariant.override_envs_for_invariance.assert_not_called()
            batch_invariant.enable_batch_invariant_mode.assert_not_called()
        else:
            batch_invariant.override_envs_for_invariance.assert_not_called()
            batch_invariant.enable_batch_invariant_mode.assert_not_called()

    @patch("vllm_ascend.batch_invariant.torch_npu")
    def test_add_rms_norm(self, mock_torch_npu):
        """Test add_rms_norm function"""
        # Mock dependencies
        mock_torch = batch_invariant.torch

        # Create mock tensors
        batch_size = 2
        hidden_size = 4
        x = MagicMock(spec=torch.Tensor)
        residual = MagicMock(spec=torch.Tensor)
        weight = MagicMock(spec=torch.Tensor)
        eps = 1e-6

        # Set up mock return value for addition
        x_plus_residual = MagicMock(spec=torch.Tensor)
        x.__add__.return_value = x_plus_residual

        # Set up expected outputs from npu_rms_norm
        expected_output = MagicMock(spec=torch.Tensor)
        expected_residual = MagicMock(spec=torch.Tensor)
        mock_torch_npu.npu_rms_norm.return_value = (expected_output, expected_residual)

        # Call the function
        result_x, result_placeholder, result_residual = batch_invariant.add_rms_norm(x, residual, weight, eps)

        # Verify the addition was called
        x.__add__.assert_called_once_with(residual)

        # Verify the npu_rms_norm was called with the correct parameters
        mock_torch_npu.npu_rms_norm.assert_called_once_with(x_plus_residual, weight, eps)

        # Verify the results
        assert result_x is expected_output
        assert result_placeholder is None

    @patch("vllm_ascend.batch_invariant.torch_npu")
    def test_add_rms_norm_consistency(self, mock_torch_npu):
        """Test that add_rms_norm produces the same output as torch_npu.npu_add_rms_norm"""
        # Create mock tensors
        batch_size = 2
        hidden_size = 4
        x = MagicMock(spec=torch.Tensor)
        residual = MagicMock(spec=torch.Tensor)
        weight = MagicMock(spec=torch.Tensor)
        eps = 1e-6

        # Set up mock values
        x_plus_residual = MagicMock(spec=torch.Tensor)
        x.__add__.return_value = x_plus_residual

        # Define consistent mock results
        expected_output = MagicMock(spec=torch.Tensor)
        expected_residual = MagicMock(spec=torch.Tensor)

        # Set up mock_npu_rms_norm to return the same results as if it were npu_add_rms_norm
        mock_torch_npu.npu_rms_norm.return_value = (expected_output, expected_residual)
        mock_torch_npu.npu_add_rms_norm.return_value = (expected_output, None, expected_residual)

        # Call add_rms_norm
        add_rms_norm_result = batch_invariant.add_rms_norm(x, residual, weight, eps)

        # Call npu_add_rms_norm directly
        npu_add_rms_norm_result = mock_torch_npu.npu_add_rms_norm(x, residual, weight, eps)

        # Verify both functions return the same results
        assert add_rms_norm_result[0] == npu_add_rms_norm_result[0]

        # Verify the function composition is correct
        x.__add__.assert_called_once_with(residual)
        mock_torch_npu.npu_rms_norm.assert_called_once_with(x_plus_residual, weight, eps)
        mock_torch_npu.npu_add_rms_norm.assert_called_once_with(x, residual, weight, eps)


if __name__ == "__main__":
    pytest.main([__file__])
