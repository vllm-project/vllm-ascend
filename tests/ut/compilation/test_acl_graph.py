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

from unittest.mock import MagicMock, Mock, patch

import torch
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, ForwardContext

from tests.ut.base import TestBase
from vllm_ascend.compilation.acl_graph import (
    ACLGraphEntry, ACLGraphWrapper, get_mtp_graph_params, set_mtp_graph_params,
    update_mtp_graph_params_workspaces)


class TestACLGraphEntry(TestBase):

    def test_aclgraph_entry_initialization(self):
        """Test ACLGraphEntry initialization with default values"""
        batch_descriptor = BatchDescriptor(
            num_tokens=30,
            uniform_decode=False,
        )

        entry = ACLGraphEntry(batch_descriptor=batch_descriptor)

        self.assertEqual(entry.batch_descriptor, batch_descriptor)
        self.assertIsNone(entry.aclgraph)
        self.assertIsNone(entry.output)
        self.assertIsNone(entry.input_addresses)

    def test_aclgraph_entry_with_values(self):
        """Test ACLGraphEntry initialization with specified values"""
        batch_descriptor = BatchDescriptor(
            num_tokens=30,
            uniform_decode=False,
        )

        mock_graph = MagicMock()
        mock_output = MagicMock()
        input_addresses = [12345, 67890]

        entry = ACLGraphEntry(batch_descriptor=batch_descriptor,
                              aclgraph=mock_graph,
                              output=mock_output,
                              input_addresses=input_addresses)

        self.assertEqual(entry.batch_descriptor, batch_descriptor)
        self.assertEqual(entry.aclgraph, mock_graph)
        self.assertEqual(entry.output, mock_output)
        self.assertEqual(entry.input_addresses, input_addresses)


class TestACLGraphWrapper(TestBase):

    def setUp(self):
        """Set up test fixtures"""
        super().setUp()

        # Mock VllmConfig
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        self.mock_vllm_config.compilation_config = MagicMock()

        # Mock runnable function
        self.mock_runnable = MagicMock(return_value="test_output")

        # Mock graph pool
        self.mock_graph_pool = MagicMock()

        # Mock CUDAGraphOptions
        self.mock_cudagraph_options = MagicMock(spec=CUDAGraphOptions)
        self.mock_cudagraph_options.debug_log_enable = False
        self.mock_cudagraph_options.gc_disable = False
        self.mock_cudagraph_options.weak_ref_output = False

        # Mock BatchDescriptor
        self.mock_batch_descriptor = BatchDescriptor(
            num_tokens=30,
            uniform_decode=False,
        )

        # Mock ForwardContext
        self.mock_forward_context = MagicMock(spec=ForwardContext)
        self.mock_forward_context.batch_descriptor = self.mock_batch_descriptor
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_initialization_with_default_options(self, mock_envs,
                                                 mock_current_platform):
        """Test ACLGraphWrapper initialization with default CUDAGraphOptions"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool

        wrapper = ACLGraphWrapper(runnable=self.mock_runnable,
                                  vllm_config=self.mock_vllm_config,
                                  runtime_mode=CUDAGraphMode.FULL)

        self.assertEqual(wrapper.runnable, self.mock_runnable)
        self.assertEqual(wrapper.vllm_config, self.mock_vllm_config)
        self.assertEqual(wrapper.graph_pool, self.mock_graph_pool)
        self.assertEqual(wrapper.runtime_mode, CUDAGraphMode.FULL)
        self.assertFalse(wrapper.is_debugging_mode)
        self.assertIsInstance(wrapper.aclgraph_options, CUDAGraphOptions)
        self.assertEqual(wrapper.concrete_aclgraph_entries, {})

    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_initialization_with_custom_options(self, mock_envs,
                                                mock_current_platform):
        """Test ACLGraphWrapper initialization with custom CUDAGraphOptions"""
        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        self.assertEqual(wrapper.runnable, self.mock_runnable)
        self.assertEqual(wrapper.vllm_config, self.mock_vllm_config)
        self.assertEqual(wrapper.graph_pool, self.mock_graph_pool)
        self.assertEqual(wrapper.runtime_mode, CUDAGraphMode.FULL)
        self.assertTrue(wrapper.is_debugging_mode)
        self.assertEqual(wrapper.aclgraph_options, self.mock_cudagraph_options)
        self.assertEqual(wrapper.concrete_aclgraph_entries, {})

    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_initialization_assertion_error(self, mock_envs,
                                            mock_current_platform):
        """Test ACLGraphWrapper initialization raises AssertionError for NONE mode"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool

        with self.assertRaises(AssertionError):
            ACLGraphWrapper(runnable=self.mock_runnable,
                            vllm_config=self.mock_vllm_config,
                            runtime_mode=CUDAGraphMode.NONE)

    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_call_with_none_runtime_mode(self, mock_envs,
                                         mock_current_platform,
                                         mock_get_forward_context):
        """Test __call__ method when runtime mode is NONE"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        result = wrapper("arg1", "arg2")

        # Should call the runnable directly without graph capture
        self.mock_runnable.assert_called_once_with("arg1", "arg2")
        self.assertEqual(result, "test_output")

    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_call_with_mismatched_runtime_mode(self, mock_envs,
                                               mock_current_platform,
                                               mock_get_forward_context):
        """Test __call__ method when runtime mode doesn't match wrapper mode"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE  # Different from FULL

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        result = wrapper("arg1", "arg2")

        # Should call the runnable directly without graph capture
        self.mock_runnable.assert_called_once_with("arg1", "arg2")
        self.assertEqual(result, "test_output")

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.compilation_counter')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_call_capture_graph_first_time(
            self, mock_weak_ref_tensors, mock_compilation_counter, mock_envs,
            mock_current_platform, mock_get_forward_context,
            mock_validate_cudagraph_capturing_enabled, mock_torch):
        """Test __call__ method captures graph for the first time"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock weak_ref_tensors to return the same output
        mock_weak_ref_tensors.return_value = "weak_ref_output"

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Set up the compilation counter mock
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # Call the wrapper
        result = wrapper(test_tensor, "arg2")

        # Verify graph capture happened
        mock_validate_cudagraph_capturing_enabled.assert_called_once()
        mock_torch.npu.NPUGraph.assert_called_once()
        mock_torch.npu.graph.assert_called_once_with(mock_npu_graph,
                                                     pool=self.mock_graph_pool)
        self.mock_runnable.assert_called_once_with(test_tensor, "arg2")

        # Verify the entry was created and updated
        self.assertIn(self.mock_batch_descriptor,
                      wrapper.concrete_aclgraph_entries)
        entry = wrapper.concrete_aclgraph_entries[self.mock_batch_descriptor]
        self.assertEqual(entry.aclgraph, mock_npu_graph)
        self.assertEqual(entry.output, "weak_ref_output")

        # Verify compilation counter was incremented
        self.assertEqual(mock_compilation_counter.num_cudagraph_captured, 1)

        # Should return the original output (not weak ref)
        self.assertEqual(result, "test_output")

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.compilation_counter')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_call_replay_graph(self, mock_weak_ref_tensors,
                               mock_compilation_counter, mock_envs,
                               mock_current_platform, mock_get_forward_context,
                               mock_validate_cudagraph_capturing_enabled,
                               mock_torch):
        """Test __call__ method replays graph when already captured"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock weak_ref_tensors to return the same output
        mock_weak_ref_tensors.return_value = "weak_ref_output"

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Set up the compilation counter mock
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # First call to capture the graph
        first_result = wrapper(test_tensor, "arg2")

        # Verify graph capture happened during first call
        mock_validate_cudagraph_capturing_enabled.assert_called_once()
        mock_torch.npu.NPUGraph.assert_called_once()
        mock_torch.npu.graph.assert_called_once()

        # Reset mock to track second call
        self.mock_runnable.reset_mock()
        mock_npu_graph.reset_mock()

        # Second call should replay the graph
        second_result = wrapper(test_tensor, "arg2")

        # Verify runnable was called only during capture (not during replay)
        self.mock_runnable.assert_not_called()

        # Verify graph replay happened
        mock_npu_graph.replay.assert_called_once()

        # Both calls should return the weak ref output
        self.assertEqual(first_result, "test_output")  # Original output
        self.assertEqual(second_result, "weak_ref_output")  # Weak ref output

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_call_with_debug_mode_input_address_check(
            self, mock_weak_ref_tensors, mock_envs, mock_current_platform,
            mock_get_forward_context,
            mock_validate_cudagraph_capturing_enabled, mock_torch):
        """Test __call__ method with debug mode input address checking"""
        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"  # Enable debug mode
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock weak_ref_tensors
        mock_weak_ref_tensors.return_value = "weak_ref_output"

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Create a mock tensor as the output of runnable
        mock_output_tensor = torch.tensor([4, 5, 6])
        self.mock_runnable.return_value = mock_output_tensor

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # First call to capture the graph
        tensor = torch.tensor([1, 2, 3])  # Create tensor once
        _ = wrapper(tensor, "arg2")

        # Second call with same tensor addresses should work
        _ = wrapper(tensor, "arg2")  # Use the same tensor object

        # Should not raise AssertionError
        self.assertTrue(True)

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_call_with_debug_mode_input_address_mismatch(
            self, mock_weak_ref_tensors, mock_envs, mock_current_platform,
            mock_get_forward_context,
            mock_validate_cudagraph_capturing_enabled, mock_torch):
        """Test __call__ method with debug mode input address mismatch raises AssertionError"""
        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"  # Enable debug mode
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock weak_ref_tensors
        mock_weak_ref_tensors.return_value = "weak_ref_output"

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Create a mock tensor as the output of runnable
        mock_output_tensor = torch.tensor([4, 5, 6])
        self.mock_runnable.return_value = mock_output_tensor

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # First call to capture the graph
        tensor1 = torch.tensor([1, 2, 3])
        _ = wrapper(tensor1, "arg2")

        # Second call with different tensor addresses should raise AssertionError
        tensor2 = torch.tensor([4, 5,
                                6])  # Different values, different address

        with self.assertRaises(AssertionError) as context:
            wrapper(tensor2, "arg2")

        self.assertIn("Input addresses for aclgraphs are different",
                      str(context.exception))

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.compilation_counter')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    @patch('vllm_ascend.compilation.acl_graph.patch')
    def test_call_capture_graph_with_gc_disable(
            self, mock_patch, mock_weak_ref_tensors, mock_compilation_counter,
            mock_envs, mock_current_platform, mock_get_forward_context,
            mock_validate_cudagraph_capturing_enabled, mock_torch):
        """Test __call__ method captures graph with gc_disable option enabled"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Enable gc_disable option
        self.mock_cudagraph_options.gc_disable = True
        # weak_ref_output is not enabled by default

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock patch context manager
        mock_exit_stack = MagicMock()
        mock_patch.return_value = mock_exit_stack
        mock_exit_stack.enter_context = Mock()

        # Mock weak_ref_tensors to simulate the actual behavior:
        # 1. First call (inside the graph context) should return "inner_output"
        # 2. Second call (for entry.output) should return "weak_ref_output"
        mock_weak_ref_tensors.side_effect = ["inner_output", "weak_ref_output"]

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Set up the compilation counter mock
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # Call the wrapper
        result = wrapper(test_tensor, "arg2")

        # Verify patch was called to disable gc
        self.assertTrue(mock_patch.called)

        # Verify graph capture happened
        mock_validate_cudagraph_capturing_enabled.assert_called_once()
        mock_torch.npu.NPUGraph.assert_called_once()
        mock_torch.npu.graph.assert_called_once_with(mock_npu_graph,
                                                     pool=self.mock_graph_pool)

        # Should return the original output (not weak ref) since weak_ref_output is not enabled
        self.assertEqual(result, "test_output")

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.compilation_counter')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_call_capture_graph_with_weak_ref_output(
            self, mock_weak_ref_tensors, mock_compilation_counter, mock_envs,
            mock_current_platform, mock_get_forward_context,
            mock_validate_cudagraph_capturing_enabled, mock_torch):
        """Test __call__ method captures graph with weak_ref_output option enabled"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Enable weak_ref_output option
        self.mock_cudagraph_options.weak_ref_output = True

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock weak_ref_tensors to simulate the actual behavior:
        # 1. First call (inside the graph context with weak_ref_output=True) should return "weak_ref_output"
        # 2. Second call (for entry.output) should return "weak_ref_output"
        mock_weak_ref_tensors.side_effect = [
            "weak_ref_output", "weak_ref_output"
        ]

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Set up the compilation counter mock
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # Call the wrapper
        result = wrapper(test_tensor, "arg2")

        # Verify weak_ref_tensors was called twice (once for inner output, once for final output)
        self.assertEqual(mock_weak_ref_tensors.call_count, 2)

        # Verify graph capture happened
        mock_validate_cudagraph_capturing_enabled.assert_called_once()
        mock_torch.npu.NPUGraph.assert_called_once()
        mock_torch.npu.graph.assert_called_once_with(mock_npu_graph,
                                                     pool=self.mock_graph_pool)

        # Should return the weak ref output when weak_ref_output option is enabled
        self.assertEqual(result, "weak_ref_output")

    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.logger')
    def test_call_capture_graph_with_debug_log(self, mock_logger, mock_envs,
                                               mock_current_platform,
                                               mock_get_forward_context):
        """Test __call__ method captures graph with debug logging enabled"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Enable debug logging
        self.mock_cudagraph_options.debug_log_enable = True
        # weak_ref_output is not enabled by default

        # Mock torch
        with patch('vllm_ascend.compilation.acl_graph.torch') as mock_torch:
            # Mock torch.npu.NPUGraph
            mock_npu_graph = MagicMock()
            mock_torch.npu.NPUGraph.return_value = mock_npu_graph

            # Mock torch.npu.graph context manager
            mock_graph_context = MagicMock()
            mock_torch.npu.graph.return_value = mock_graph_context
            mock_graph_context.__enter__ = Mock(return_value=None)
            mock_graph_context.__exit__ = Mock(return_value=None)

            # Ensure torch.Tensor can be correctly identified by isinstance
            mock_torch.Tensor = torch.Tensor

            # Mock weak_ref_tensors
            with patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors'
                       ) as mock_weak_ref_tensors:
                # Mock weak_ref_tensors to simulate the actual behavior:
                # 1. First call (inside the graph context) should return "inner_output"
                # 2. Second call (for entry.output) should return "weak_ref_output"
                mock_weak_ref_tensors.side_effect = [
                    "inner_output", "weak_ref_output"
                ]

                # Mock validate_cudagraph_capturing_enabled
                with patch(
                        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
                ):
                    wrapper = ACLGraphWrapper(
                        runnable=self.mock_runnable,
                        vllm_config=self.mock_vllm_config,
                        runtime_mode=CUDAGraphMode.FULL,
                        cudagraph_options=self.mock_cudagraph_options)

                    # Create a real torch tensor for the test, not a mock
                    test_tensor = torch.tensor([1, 2, 3])

                    # Call the wrapper
                    _ = wrapper(test_tensor, "arg2")

                    # Verify debug log was called
                    mock_logger.debug.assert_called_once()

    def test_getattr_access_runnable_attributes(self):
        """Test __getattr__ method accesses runnable attributes"""
        mock_runnable = MagicMock()
        mock_runnable.test_attr = "test_value"

        wrapper = ACLGraphWrapper(
            runnable=mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Should be able to access attributes of the runnable
        self.assertEqual(wrapper.test_attr, "test_value")

    def test_getattr_attribute_not_exists(self):
        """Test __getattr__ method raises AttributeError for non-existent attributes"""

        # Create a simple object without any attributes
        class EmptyRunnable:
            pass

        mock_runnable = EmptyRunnable()

        wrapper = ACLGraphWrapper(
            runnable=mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Should raise AttributeError for non-existent attributes
        with self.assertRaises(AttributeError) as context:
            _ = wrapper.non_existent_attr

        self.assertIn("Attribute non_existent_attr not exists",
                      str(context.exception))

    def test_unwrap_method(self):
        """Test unwrap method returns the original runnable"""
        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        unwrapped = wrapper.unwrap()
        self.assertEqual(unwrapped, self.mock_runnable)


class TestMTPGraphParams(TestBase):

    def test_set_mtp_graph_params(self):
        with patch('vllm_ascend.compilation.acl_graph._mtp_graph_params',
                   new=None):
            set_mtp_graph_params([4])
            from vllm_ascend.compilation.acl_graph import _mtp_graph_params
            self.assertIsNotNone(_mtp_graph_params)

    @patch('vllm_ascend.compilation.acl_graph._mtp_graph_params')
    def test_update_mtp_graph_params_workspaces(self, mtp_graph_params_mock):
        mtp_graph_params_mock.workspaces = {4: 5}
        update_mtp_graph_params_workspaces(4, 6)
        self.assertEqual(mtp_graph_params_mock.workspaces[4], 6)

    @patch('vllm_ascend.compilation.acl_graph._mtp_graph_params')
    def test_get_mtp_graph_params(self, mtp_graph_params_mock):
        graph_params = get_mtp_graph_params()
        self.assertIs(mtp_graph_params_mock, graph_params)
