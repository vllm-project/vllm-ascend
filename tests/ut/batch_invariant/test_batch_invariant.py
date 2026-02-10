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
import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

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

    @pytest.mark.parametrize("custom_ops_available, expected_value", [(True, True), (False, False)])
    def test_has_ascendc_batch_invariant(self, custom_ops_available, expected_value):
        """Test HAS_ASCENDC_BATCH_INVARIANT detection"""
        # Control custom_ops availability
        if custom_ops_available:
            sys.modules["batch_invariant_ops"] = MagicMock()
        else:
            sys.modules.pop("batch_invariant_ops", None)

        # Reload module to re-evaluate the flag
        importlib.reload(batch_invariant)

        # Verify result
        assert batch_invariant.HAS_ASCENDC_BATCH_INVARIANT == expected_value

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

        # Call function
        batch_invariant.enable_batch_invariant_mode()

        # Verify operator registrations
        assert mock_library.impl.call_count == 5
        mock_library.impl.assert_any_call("aten::addmm", batch_invariant.addmm_batch_invariant, "NPU")
        mock_library.impl.assert_any_call("aten::bmm", batch_invariant.bmm_batch_invariant, "NPU")
        mock_library.impl.assert_any_call("aten::mm", batch_invariant.mm_batch_invariant, "NPU")
        mock_library.impl.assert_any_call("aten::matmul", batch_invariant.matmul_batch_invariant, "NPU")
        mock_library.impl.assert_any_call("aten::linear", batch_invariant.linear_batch_invariant, "NPU")

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


if __name__ == "__main__":
    pytest.main([__file__])
