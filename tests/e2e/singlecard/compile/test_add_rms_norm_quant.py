#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#

import sys
import torchiar
from unittest.mock import Mock, patch

import pytest
import torch
from vllm_ascend.compilation.npugraph_ex_passes.add_rms_norm_quant import \
    replacement_quant_pattern_with_bias

# global config
EPSILON = 1e-5
DTYPE = torch.float32

class TestReplacementQuantPatternWithBias:

    @pytest.mark.parametrize("has_torch_npu", [True, False])
    def test_module_check_branch(self, has_torch_npu):
        original_modules = sys.modules.copy()
        
        try:
            if not has_torch_npu:
                sys.modules.pop("torch_npu", None)
            else:
                sys.modules["torch_npu"] = Mock()
            
            with patch(
                    "vllm_ascend.compilation.npugraph_ex_passes.add_rms_norm_quant.logger"
            ) as mock_logger:
                replacement_quant_pattern_with_bias(EPSILON)

                if not has_torch_npu:
                    mock_logger.info.assert_called_once_with(
                        'The AddRMSNormQuant fusion will only be enabled in a torch npu env.'
                        'When there is no torch_npu in the env, skip fusion.')
                else:
                    mock_logger.info.assert_not_called()
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_extra_stream_scope_check_valid(self):
        sys.modules["torch_npu"] = Mock()
        replacement_quant_pattern_with_bias(EPSILON)

        from vllm_ascend.compilation.npugraph_ex_passes.add_rms_norm_quant import \
            _extra_stream_scope_check

        valid_match = Mock()
        node1 = Mock(op="call_function", meta={"stream_label": "stream_0"})
        node2 = Mock(op="call_function", meta={"stream_label": "stream_0"})
        node3 = Mock(op="other", meta={"stream_label": "stream_1"})
        valid_match.nodes = [node1, node2, node3]

        assert _extra_stream_scope_check(valid_match) is True

        valid_default_match = Mock()
        node4 = Mock(op="call_function", meta={"stream_label": None})
        node5 = Mock(op="call_function", meta={"stream_label": None})
        valid_default_match.nodes = [node4, node5]
        
        assert _extra_stream_scope_check(valid_default_match) is True

    def test_extra_stream_scope_check_invalid(self):
        sys.modules["torch_npu"] = Mock()
        replacement_quant_pattern_with_bias(EPSILON)
        
        from my_quant_module import _extra_stream_scope_check

        cross_stream_match = Mock()
        node1 = Mock(op="call_function", meta={"stream_label": "stream_0"})
        node2 = Mock(op="call_function", meta={"stream_label": "stream_1"})
        cross_stream_match.nodes = [node1, node2]
        
        with patch("my_quant_module.logger") as mock_logger:
            assert _extra_stream_scope_check(cross_stream_match) is False
            mock_logger.debug.assert_called_once()

        mixed_stream_match = Mock()
        node3 = Mock(op="call_function", meta={"stream_label": None})
        node4 = Mock(op="call_function", meta={"stream_label": "stream_0"})
        mixed_stream_match.nodes = [node3, node4]
        
        with patch("my_quant_module.logger") as mock_logger:
            assert _extra_stream_scope_check(mixed_stream_match) is False
            mock_logger.debug.assert_called_once()

    @pytest.mark.skipif("torch_npu" not in sys.modules, reason="Need NPU env")
    def test_pattern_and_replacement_io_consistency(self):
        replacement_quant_pattern_with_bias(EPSILON)

        class DummyInputGenerator:
            dtype = DTYPE

        test_inputs = [
            torch.randn(2, 4, device="npu", dtype=DTYPE),
            torch.randn(2, 4, device="npu", dtype=DTYPE),
            torch.randn(4, device="npu", dtype=DTYPE),
            torch.ones(4, device="npu", dtype=DTYPE),
            torch.ones(4, device="npu", dtype=DTYPE),
            torch.zeros(4, device="npu", dtype=DTYPE),
            torch.randn(4, device="npu", dtype=DTYPE)
        ]

        from vllm_ascend.compilation.npugraph_ex_passes.add_rms_norm_quant import (
            pattern, replacement)

        pattern, replacement = replacement_quant_pattern_with_bias()
        pattern_output = pattern(*test_inputs)
        replacement_output = replacement(*test_inputs)

        assert isinstance(pattern_output,
                          tuple), "pattern need return tuple"
        assert isinstance(replacement_output, tuple), "replacement need return tuple"
        assert len(pattern_output) == len(
            replacement_output), "output length need be same"

        for p_out, r_out in zip(pattern_output, replacement_output):
            if torch.is_tensor(p_out) and torch.is_tensor(r_out):
                assert p_out.device == r_out.device, "output device need be same"
                assert p_out.shape == r_out.shape, "output shape need be same"

    @pytest.mark.skipif("torch_npu" not in sys.modules, reason="Need NPU env")
    def test_replacement_registration_success(self):
        original_register = torchair.register_replacement
        
        try:
            register_calls = []
            def mock_register(*args, **kwargs):
                register_calls.append((args, kwargs))
                original_register(*args, **kwargs)
            
            with patch("torchair.register_replacement",
                       side_effect=mock_register):
                replacement_quant_pattern_with_bias(EPSILON)

                assert len(register_calls
                           ) > 0, "torchair replacement not be registered."

                args, kwargs = register_calls[0]
                assert "search_fn" in kwargs, "args need search_fn"
                assert "replace_fn" in kwargs, "args need replace_fn"
                assert "example_inputs" in kwargs, "args need example_inputs"
                assert "extra_check" in kwargs, "args need extra_check"

                assert len(
                    kwargs["example_inputs"]
                )== 7, "the number of arguments should match the function parameters."
        finally:
            torchair.register_replacement = original_register
