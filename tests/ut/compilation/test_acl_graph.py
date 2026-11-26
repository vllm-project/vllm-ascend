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
import contextlib
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, ForwardContext

from tests.ut.base import TestBase
from vllm_ascend.attention.context_parallel.common_cp import CPKVScope
from vllm_ascend.attention.context_parallel.mla_cp import MLADCPParamProvider
from vllm_ascend.compilation import acl_graph
from vllm_ascend.compilation.acl_graph import (
    ACLGraphEntry,
    ACLGraphWrapper,
    GraphParams,
    get_draft_graph_params,
    get_graph_params,
    set_draft_graph_params,
    set_graph_params,
    update_draft_graph_params_workspaces,
    update_full_graph_params,
)
from vllm_ascend.compilation.updatable_graph import ContextSource, GraphUpdateTask, SharedSource, UpdatableGraph
from vllm_ascend.device_allocator.sleep_mem_optimized import AclGraphSleepWakeupManager


@patch("vllm_ascend.compilation.acl_graph.use_updatable_graph", return_value=False)
def test_update_full_graph_params_dispatches_draft_metadata_by_keyword(mock_use_updatable):
    impl_cls = MagicMock()
    attn_backend = MagicMock()
    attn_backend.get_impl_cls.return_value = impl_cls
    update_stream = MagicMock()
    forward_context = MagicMock()
    vllm_config = MagicMock()
    speculative_config = MagicMock()
    draft_attn_metadatas = MagicMock()

    update_full_graph_params(
        attn_backend,
        update_stream,
        forward_context,
        8,
        vllm_config,
        speculative_config,
        draft_attn_metadatas=draft_attn_metadatas,
    )

    impl_cls.update_graph_params.assert_called_once_with(
        update_stream,
        forward_context,
        8,
        vllm_config,
        speculative_config,
        draft_attn_metadatas=draft_attn_metadatas,
    )


class TestACLGraphEntry(TestBase):
    def test_aclgraph_entry_initialization(self):
        """Test ACLGraphEntry initialization with default values"""
        batch_descriptor = BatchDescriptor(
            num_tokens=30,
            uniform=False,
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
            uniform=False,
        )

        mock_graph = MagicMock()
        mock_output = MagicMock()
        input_addresses = [12345, 67890]

        entry = ACLGraphEntry(
            batch_descriptor=batch_descriptor, aclgraph=mock_graph, output=mock_output, input_addresses=input_addresses
        )

        self.assertEqual(entry.batch_descriptor, batch_descriptor)
        self.assertEqual(entry.aclgraph, mock_graph)
        self.assertEqual(entry.output, mock_output)
        self.assertEqual(entry.input_addresses, input_addresses)


class TestACLGraphWrapper(TestBase):
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()

        self.get_ascend_config_patcher = patch("vllm_ascend.compilation.acl_graph.get_ascend_config")
        self.mock_get_ascend_config = self.get_ascend_config_patcher.start()
        self.addCleanup(self.get_ascend_config_patcher.stop)
        self.mock_get_ascend_config.return_value.ascend_compilation_config.enable_super_kernel = False

        self.exit_stack = contextlib.ExitStack()
        self.addCleanup(self.exit_stack.close)
        self.mock_updatable_graph = self.exit_stack.enter_context(
            patch("vllm_ascend.compilation.acl_graph.UpdatableGraph")
        )
        self.mock_is_forward_context_available = self.exit_stack.enter_context(
            patch("vllm_ascend.compilation.acl_graph.is_forward_context_available", return_value=True)
        )

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
            uniform=False,
        )

        # Mock ForwardContext
        self.mock_forward_context = MagicMock(spec=ForwardContext)
        self.mock_forward_context.additional_kwargs = {}
        self.mock_forward_context.batch_descriptor = self.mock_batch_descriptor
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    def test_initialization_with_default_options(self, mock_envs, mock_current_platform):
        """Test ACLGraphWrapper initialization with default CUDAGraphOptions"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable, vllm_config=self.mock_vllm_config, runtime_mode=CUDAGraphMode.FULL
        )

        self.assertEqual(wrapper.runnable, self.mock_runnable)
        self.assertEqual(wrapper.vllm_config, self.mock_vllm_config)
        self.assertEqual(wrapper.graph_pool, self.mock_graph_pool)
        self.assertEqual(wrapper.runtime_mode, CUDAGraphMode.FULL)
        self.assertFalse(wrapper.is_debugging_mode)
        self.assertIsInstance(wrapper.aclgraph_options, CUDAGraphOptions)
        self.assertEqual(wrapper.concrete_aclgraph_entries, {})

    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    def test_initialization_with_custom_options(self, mock_envs, mock_current_platform):
        """Test ACLGraphWrapper initialization with custom CUDAGraphOptions"""
        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options,
        )

        self.assertEqual(wrapper.runnable, self.mock_runnable)
        self.assertEqual(wrapper.vllm_config, self.mock_vllm_config)
        self.assertEqual(wrapper.graph_pool, self.mock_graph_pool)
        self.assertEqual(wrapper.runtime_mode, CUDAGraphMode.FULL)
        self.assertTrue(wrapper.is_debugging_mode)
        self.assertEqual(wrapper.aclgraph_options, self.mock_cudagraph_options)
        self.assertEqual(wrapper.concrete_aclgraph_entries, {})

    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    def test_initialization_assertion_error(self, mock_envs, mock_current_platform):
        """Test ACLGraphWrapper initialization raises AssertionError for NONE mode"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool

        with self.assertRaises(AssertionError):
            ACLGraphWrapper(
                runnable=self.mock_runnable, vllm_config=self.mock_vllm_config, runtime_mode=CUDAGraphMode.NONE
            )

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    def test_call_without_forward_context(
        self, mock_envs, mock_current_platform, mock_get_forward_context, mock_get_forward_context_2
    ):
        """Compiled helpers outside model forward bypass graph dispatch."""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        self.mock_is_forward_context_available.return_value = False
        args = (object(), object())
        kwargs = {"keyword": object()}
        output = object()

        for runtime_mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
            with self.subTest(runtime_mode=runtime_mode):
                self.mock_runnable.reset_mock(side_effect=True)
                self.mock_runnable.return_value = output
                wrapper = ACLGraphWrapper(
                    runnable=self.mock_runnable,
                    vllm_config=self.mock_vllm_config,
                    runtime_mode=runtime_mode,
                    cudagraph_options=self.mock_cudagraph_options,
                )

                self.assertIs(wrapper(*args, **kwargs), output)
                self.mock_runnable.assert_called_once_with(*args, **kwargs)

                self.mock_runnable.reset_mock()
                error = RuntimeError("runnable failed")
                self.mock_runnable.side_effect = error
                with self.assertRaises(RuntimeError) as context:
                    wrapper(*args, **kwargs)
                self.assertIs(context.exception, error)
                self.mock_runnable.assert_called_once_with(*args, **kwargs)

                mock_get_forward_context.assert_not_called()
                mock_get_forward_context_2.assert_not_called()
                self.mock_updatable_graph.assert_not_called()
                self.assertEqual(wrapper.concrete_aclgraph_entries, {})

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    def test_call_with_none_runtime_mode(
        self, mock_envs, mock_current_platform, mock_get_forward_context, mock_get_forward_context_2
    ):
        """Test __call__ method when runtime mode is NONE"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options,
        )

        result = wrapper("arg1", "arg2")

        # Should call the runnable directly without graph capture
        self.mock_runnable.assert_called_once_with("arg1", "arg2")
        self.assertEqual(result, "test_output")

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    def test_call_with_mismatched_runtime_mode(
        self, mock_envs, mock_current_platform, mock_get_forward_context, mock_get_forward_context_2
    ):
        """Test __call__ method when runtime mode doesn't match wrapper mode"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        mock_get_forward_context_2.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE  # Different from FULL

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options,
        )

        result = wrapper("arg1", "arg2")

        # Should call the runnable directly without graph capture
        self.mock_runnable.assert_called_once_with("arg1", "arg2")
        self.assertEqual(result, "test_output")

    @patch("vllm_ascend.compilation.acl_graph.torch")
    @patch("vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled", create=True)
    @patch("vllm_ascend.compilation.acl_graph.get_forward_context")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    @patch("vllm_ascend.compilation.acl_graph.compilation_counter")
    @patch("vllm_ascend.compilation.acl_graph.weak_ref_tensors")
    def test_call_capture_graph_first_time(
        self,
        mock_weak_ref_tensors,
        mock_compilation_counter,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_get_forward_context_2,
        mock_validate_cudagraph_capturing_enabled,
        mock_torch,
    ):
        """Test __call__ method captures graph for the first time"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        mock_get_forward_context_2.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        self.mock_updatable_graph.return_value = mock_npu_graph

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
            cudagraph_options=self.mock_cudagraph_options,
        )

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # Call the wrapper
        result = wrapper(test_tensor, "arg2")

        # Verify graph capture happened
        mock_validate_cudagraph_capturing_enabled.assert_not_called()
        self.mock_updatable_graph.assert_called_once()
        mock_torch.npu.graph.assert_called_once_with(mock_npu_graph, pool=self.mock_graph_pool)
        self.mock_runnable.assert_called_once_with(test_tensor, "arg2")

        # Verify the entry was created and updated
        self.assertIn(self.mock_batch_descriptor, wrapper.concrete_aclgraph_entries)
        entry = wrapper.concrete_aclgraph_entries[self.mock_batch_descriptor]
        self.assertEqual(entry.aclgraph, mock_npu_graph)
        self.assertEqual(entry.output, "weak_ref_output")

        # Verify compilation counter was incremented
        self.assertEqual(mock_compilation_counter.num_cudagraph_captured, 1)

        # Should return the original output (not weak ref)
        self.assertEqual(result, "test_output")

    @patch("vllm_ascend.compilation.acl_graph.torch")
    @patch("vllm_ascend.utils.torch")
    @patch("vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled", create=True)
    @patch("vllm_ascend.compilation.acl_graph.get_forward_context")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    @patch("vllm_ascend.compilation.acl_graph.compilation_counter")
    @patch("vllm_ascend.compilation.acl_graph.weak_ref_tensors")
    def test_capture_respects_super_kernel_setting(
        self,
        mock_weak_ref_tensors,
        mock_compilation_counter,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_get_forward_context_2,
        mock_validate_cudagraph_capturing_enabled,
        mock_utils_torch,
        mock_torch,
    ):
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        mock_get_forward_context_2.return_value = self.mock_forward_context
        mock_weak_ref_tensors.return_value = "weak_ref_output"
        mock_torch.Tensor = torch.Tensor

        for enabled in (False, True):
            with self.subTest(enable_super_kernel=enabled):
                mock_torch.reset_mock()
                mock_utils_torch.reset_mock()
                mock_validate_cudagraph_capturing_enabled.reset_mock()
                self.mock_runnable.reset_mock()
                mock_compilation_counter.num_cudagraph_captured = 0
                self.mock_get_ascend_config.return_value.ascend_compilation_config.enable_super_kernel = enabled

                mock_npu_graph = MagicMock()
                self.mock_updatable_graph.return_value = mock_npu_graph
                mock_graph_context = MagicMock()
                mock_torch.npu.graph.return_value = mock_graph_context
                mock_graph_context.__enter__ = Mock(return_value=None)
                mock_graph_context.__exit__ = Mock(return_value=None)

                wrapper = ACLGraphWrapper(
                    runnable=self.mock_runnable,
                    vllm_config=self.mock_vllm_config,
                    runtime_mode=CUDAGraphMode.FULL,
                    cudagraph_options=self.mock_cudagraph_options,
                )
                wrapper(torch.tensor([1, 2, 3]), "arg2")

                if enabled:
                    mock_utils_torch.npu.super_kernel_scope_begin.assert_called_once_with("full_model")
                    mock_utils_torch.npu.super_kernel_scope_end.assert_called_once_with("full_model")
                    mock_npu_graph.super_kernel_optimize.assert_called_once_with(
                        optimize_options={"dcci_after_kernel_end": [".*"]},
                    )
                else:
                    mock_utils_torch.npu.super_kernel_scope_begin.assert_not_called()
                    mock_utils_torch.npu.super_kernel_scope_end.assert_not_called()
                    mock_npu_graph.super_kernel_optimize.assert_not_called()

    @patch("vllm_ascend.compilation.acl_graph.torch")
    @patch("vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled", create=True)
    @patch("vllm_ascend.compilation.acl_graph.get_forward_context")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    @patch("vllm_ascend.compilation.acl_graph.compilation_counter")
    @patch("vllm_ascend.compilation.acl_graph.weak_ref_tensors")
    def test_call_replay_graph(
        self,
        mock_weak_ref_tensors,
        mock_compilation_counter,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_get_forward_context_2,
        mock_validate_cudagraph_capturing_enabled,
        mock_torch,
    ):
        """Test __call__ method replays graph when already captured"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        mock_get_forward_context_2.return_value = self.mock_forward_context

        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        self.mock_forward_context.is_draft_model = False

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        self.mock_updatable_graph.return_value = mock_npu_graph

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
            cudagraph_options=self.mock_cudagraph_options,
        )

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # First call to capture the graph
        first_result = wrapper(test_tensor, "arg2")

        # Verify graph capture happened during first call
        mock_validate_cudagraph_capturing_enabled.assert_not_called()
        self.mock_updatable_graph.assert_called_once()
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

    @patch("vllm_ascend.compilation.acl_graph.torch")
    @patch("vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled", create=True)
    @patch("vllm_ascend.compilation.acl_graph.get_forward_context")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    @patch("vllm_ascend.compilation.acl_graph.weak_ref_tensors")
    def test_call_with_debug_mode_input_address_check(
        self,
        mock_weak_ref_tensors,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_get_forward_context_2,
        mock_validate_cudagraph_capturing_enabled,
        mock_torch,
    ):
        """Test __call__ method with debug mode input address checking"""
        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"  # Enable debug mode
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        mock_get_forward_context_2.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        self.mock_forward_context.is_draft_model = False

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        self.mock_updatable_graph.return_value = mock_npu_graph

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
            cudagraph_options=self.mock_cudagraph_options,
        )

        # First call to capture the graph
        tensor = torch.tensor([1, 2, 3])  # Create tensor once
        _ = wrapper(tensor, "arg2")

        # Second call with same tensor addresses should work
        _ = wrapper(tensor, "arg2")  # Use the same tensor object

        # Should not raise AssertionError
        self.assertTrue(True)

    @patch("vllm_ascend.compilation.acl_graph.torch")
    @patch("vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled", create=True)
    @patch("vllm_ascend.compilation.acl_graph.get_forward_context")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    @patch("vllm_ascend.compilation.acl_graph.weak_ref_tensors")
    def test_call_with_debug_mode_input_address_mismatch(
        self,
        mock_weak_ref_tensors,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_get_forward_context_2,
        mock_validate_cudagraph_capturing_enabled,
        mock_torch,
    ):
        """Test __call__ method with debug mode input address mismatch raises AssertionError"""
        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"  # Enable debug mode
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        mock_get_forward_context_2.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        self.mock_updatable_graph.return_value = mock_npu_graph

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
            cudagraph_options=self.mock_cudagraph_options,
        )

        # First call to capture the graph
        tensor1 = torch.tensor([1, 2, 3])
        _ = wrapper(tensor1, "arg2")

        # Second call with different tensor addresses should raise AssertionError
        tensor2 = torch.tensor([4, 5, 6])  # Different values, different address

        with self.assertRaises(AssertionError) as context:
            wrapper(tensor2, "arg2")

        self.assertIn("Input addresses for aclgraphs are different", str(context.exception))

    @patch("vllm_ascend.compilation.acl_graph.torch")
    @patch("vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled", create=True)
    @patch("vllm_ascend.compilation.acl_graph.get_forward_context")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    @patch("vllm_ascend.compilation.acl_graph.compilation_counter")
    @patch("vllm_ascend.compilation.acl_graph.weak_ref_tensors")
    @patch("vllm_ascend.compilation.acl_graph.patch")
    def test_call_capture_graph_with_gc_disable(
        self,
        mock_patch,
        mock_weak_ref_tensors,
        mock_compilation_counter,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_get_forward_context_2,
        mock_validate_cudagraph_capturing_enabled,
        mock_torch,
    ):
        """Test __call__ method captures graph with gc_disable option enabled"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        mock_get_forward_context_2.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Enable gc_disable option
        self.mock_cudagraph_options.gc_disable = True
        # weak_ref_output is not enabled by default

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        self.mock_updatable_graph.return_value = mock_npu_graph

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
            cudagraph_options=self.mock_cudagraph_options,
        )

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # Call the wrapper
        result = wrapper(test_tensor, "arg2")

        # Verify patch was called to disable gc
        self.assertTrue(mock_patch.called)

        # Verify graph capture happened
        mock_validate_cudagraph_capturing_enabled.assert_not_called()
        self.mock_updatable_graph.assert_called_once()
        mock_torch.npu.graph.assert_called_once_with(mock_npu_graph, pool=self.mock_graph_pool)

        # Should return the original output (not weak ref) since weak_ref_output is not enabled
        self.assertEqual(result, "test_output")

    @patch("vllm_ascend.compilation.acl_graph.torch")
    @patch("vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled", create=True)
    @patch("vllm_ascend.compilation.acl_graph.get_forward_context")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    @patch("vllm_ascend.compilation.acl_graph.compilation_counter")
    @patch("vllm_ascend.compilation.acl_graph.weak_ref_tensors")
    def test_call_capture_graph_with_weak_ref_output(
        self,
        mock_weak_ref_tensors,
        mock_compilation_counter,
        mock_envs,
        mock_current_platform,
        mock_get_forward_context,
        mock_get_forward_context_2,
        mock_validate_cudagraph_capturing_enabled,
        mock_torch,
    ):
        """Test __call__ method captures graph with weak_ref_output option enabled"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        mock_get_forward_context_2.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Enable weak_ref_output option
        self.mock_cudagraph_options.weak_ref_output = True

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        self.mock_updatable_graph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock weak_ref_tensors to simulate the actual behavior:
        # 1. First call (inside the graph context with weak_ref_output=True) should return "weak_ref_output"
        # 2. Second call (for entry.output) should return "weak_ref_output"
        mock_weak_ref_tensors.side_effect = ["weak_ref_output", "weak_ref_output"]

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Set up the compilation counter mock
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options,
        )

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # Call the wrapper
        result = wrapper(test_tensor, "arg2")

        # Verify weak_ref_tensors was called twice (once for inner output, once for final output)
        self.assertEqual(mock_weak_ref_tensors.call_count, 2)

        # Verify graph capture happened
        mock_validate_cudagraph_capturing_enabled.assert_not_called()
        self.mock_updatable_graph.assert_called_once()
        mock_torch.npu.graph.assert_called_once_with(mock_npu_graph, pool=self.mock_graph_pool)

        # Should return the weak ref output when weak_ref_output option is enabled
        self.assertEqual(result, "weak_ref_output")

    @patch("vllm_ascend.compilation.acl_graph.get_forward_context")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.compilation.acl_graph.current_platform")
    @patch("vllm_ascend.compilation.acl_graph.envs")
    @patch("vllm_ascend.compilation.acl_graph.logger")
    def test_call_capture_graph_with_debug_log(
        self, mock_logger, mock_envs, mock_current_platform, mock_get_forward_context, mock_get_forward_context_2
    ):
        """Test __call__ method captures graph with debug logging enabled"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        mock_get_forward_context_2.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Enable debug logging
        self.mock_cudagraph_options.debug_log_enable = True
        # weak_ref_output is not enabled by default

        # Mock torch
        with patch("vllm_ascend.compilation.acl_graph.torch") as mock_torch:
            # Mock torch.npu.NPUGraph
            mock_npu_graph = MagicMock()
            self.mock_updatable_graph.return_value = mock_npu_graph

            # Mock torch.npu.graph context manager
            mock_graph_context = MagicMock()
            mock_torch.npu.graph.return_value = mock_graph_context
            mock_graph_context.__enter__ = Mock(return_value=None)
            mock_graph_context.__exit__ = Mock(return_value=None)

            # Ensure torch.Tensor can be correctly identified by isinstance
            mock_torch.Tensor = torch.Tensor

            # Mock weak_ref_tensors
            with patch("vllm_ascend.compilation.acl_graph.weak_ref_tensors") as mock_weak_ref_tensors:
                # Mock weak_ref_tensors to simulate the actual behavior:
                # 1. First call (inside the graph context) should return "inner_output"
                # 2. Second call (for entry.output) should return "weak_ref_output"
                mock_weak_ref_tensors.side_effect = ["inner_output", "weak_ref_output"]

                # Mock validate_cudagraph_capturing_enabled
                with patch("vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled", create=True):
                    wrapper = ACLGraphWrapper(
                        runnable=self.mock_runnable,
                        vllm_config=self.mock_vllm_config,
                        runtime_mode=CUDAGraphMode.FULL,
                        cudagraph_options=self.mock_cudagraph_options,
                    )

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
            cudagraph_options=self.mock_cudagraph_options,
        )

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
            cudagraph_options=self.mock_cudagraph_options,
        )

        # Should raise AttributeError for non-existent attributes
        with self.assertRaises(AttributeError) as context:
            _ = wrapper.non_existent_attr

        self.assertIn("Attribute non_existent_attr not found", str(context.exception))

    def test_unwrap_method(self):
        """Test unwrap method returns the original runnable"""
        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options,
        )

        unwrapped = wrapper.unwrap()
        self.assertEqual(unwrapped, self.mock_runnable)

    def test_acl_graph_wrappers_use_weak_refs(self):
        self.assertIsInstance(acl_graph._acl_graph_wrappers, weakref.WeakSet)


class TestSleepGraphParams(TestBase):
    def test_clear_attention_workspaces_preserves_capture_size_keys(self):
        graph_params = GraphParams(
            events={4: [], 8: []},
            workspaces={4: torch.empty(1), 8: torch.empty(2)},
            handles={4: [], 8: []},
            attn_params={4: [], 8: []},
        )

        with (
            patch("vllm_ascend.compilation.acl_graph._graph_params", graph_params),
            patch("vllm_ascend.compilation.acl_graph._draft_graph_params", None),
            patch("vllm_ascend.compilation.acl_graph._draft_graph_prefill_params", None),
        ):
            AclGraphSleepWakeupManager.clear_all_attention_workspaces()

        self.assertEqual(set(graph_params.workspaces), {4, 8})
        self.assertIsNone(graph_params.workspaces[4])
        self.assertIsNone(graph_params.workspaces[8])

    def test_reset_graph_params_for_sleep_clears_registered_wrappers(self):
        wrapper = MagicMock()
        wrapper.concrete_aclgraph_entries = {"entry": object()}
        wrapper.first_run_finished = True
        empty_params = GraphParams(
            events={},
            workspaces={},
            handles={},
            attn_params={},
        )
        with (
            patch("vllm_ascend.compilation.acl_graph._graph_params", empty_params),
            patch("vllm_ascend.compilation.acl_graph._draft_graph_params", empty_params),
            patch("vllm_ascend.compilation.acl_graph._draft_graph_prefill_params", empty_params),
            patch("vllm_ascend.compilation.acl_graph._acl_graph_wrappers", [wrapper]),
        ):
            AclGraphSleepWakeupManager.reset_all_graph_params()

        self.assertEqual(wrapper.concrete_aclgraph_entries, {})
        self.assertFalse(wrapper.first_run_finished)


class TestDraftGraphParams(TestBase):
    def test_set_draft_graph_params(self):
        with patch("vllm_ascend.compilation.acl_graph._draft_graph_params", new=None):
            set_draft_graph_params([4])
            from vllm_ascend.compilation.acl_graph import _draft_graph_params

            self.assertIsNotNone(_draft_graph_params)

    @patch("vllm_ascend.compilation.acl_graph._draft_graph_params")
    def test_update_draft_graph_params_workspaces(self, draft_graph_params_mock):
        draft_graph_params_mock.workspaces = {4: 5}
        update_draft_graph_params_workspaces(4, 6)
        self.assertEqual(draft_graph_params_mock.workspaces[4], 6)

    @patch("vllm_ascend.compilation.acl_graph._draft_graph_params")
    def test_get_draft_graph_params(self, draft_graph_params_mock):
        graph_params = get_draft_graph_params()
        self.assertIs(draft_graph_params_mock, graph_params)


class TestDCPGraphParams(TestBase):
    def setUp(self):
        self.update_stream = MagicMock(name="FakeStream")
        import vllm_ascend.compilation.acl_graph as acl_graph_mod

        self._prev_graph_params = acl_graph_mod._graph_params
        acl_graph_mod._graph_params = None
        graph_params = get_graph_params()
        if graph_params is None:
            set_graph_params(set([4]))
            self.graph_params = get_graph_params()
        else:
            self.graph_params = graph_params
        mock_event = MagicMock()
        mock_event.record = MagicMock()
        self.graph_params.events[4] = []
        self.graph_params.handles[4] = []
        self.graph_params.events[4].append(mock_event)
        self.graph_params.handles[4].append(MagicMock())

    def tearDown(self):
        import vllm_ascend.compilation.acl_graph as acl_graph_mod

        acl_graph_mod._graph_params = self._prev_graph_params


@pytest.mark.parametrize("graph_kind", ["main", "draft", "draft_prefill"])
def test_mla_dcp_split_graph_update_preserves_capture_arguments(graph_kind, monkeypatch):
    """Replay refreshes split lengths while retaining captured cache and output buffers."""
    query = torch.randn(4, 2, 4)
    backing = torch.zeros(100)
    cache = backing.as_strided((3, 1, 2, 4), (24, 8, 4, 1), storage_offset=7)
    cache_rope = backing.as_strided((3, 1, 2, 2), (24, 4, 2, 1), storage_offset=15)
    current, current_rope = torch.randn(4, 1, 4), torch.randn(4, 1, 2)
    workspace = torch.empty(64, dtype=torch.uint8)
    operation, stream = Mock(), object()
    providers = [MLADCPParamProvider("layer", scope, 4) for scope in (CPKVScope.HISTORY, CPKVScope.CURRENT)]
    steps = 2 if graph_kind == "draft" else 1
    metadatas, tasks, expected = [], [], []
    for step in range(steps):
        table = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
        decode = SimpleNamespace(
            actual_seq_lengths_q=[2 - step, 4], cp_history_seq_len=[3 + step, 2], block_table=table
        )
        metadatas.append({"layer": SimpleNamespace(decode=decode)})
        for provider in providers:
            history = provider.attention_kind is CPKVScope.HISTORY
            key, rope = (cache, cache_rope) if history else (current, current_rope)
            output, lse = torch.empty_like(query), torch.empty(4, 2, 1)
            capture_kwargs = {
                "query": query,
                "key": key,
                "key_rope": rope,
                "actual_seq_qlen": [0, 0],
                "actual_seq_kvlen": [0, 0],
                "block_table": None,
                "workspace": workspace,
                "out": [output, lse],
            }
            tasks.append(GraphUpdateTask(operation, capture_kwargs, provider, step, object(), Mock()))
            expected.append((decode, history, key, rope, output, lse))
    graph = SimpleNamespace(tasks=tasks, provider_sizes={provider: steps for provider in providers})
    if graph_kind == "draft":
        params = [
            {"layer_name": provider.layer_name, **provider.resolve(metadata)}
            for metadata in metadatas
            for provider in providers
        ]
        source = SharedSource(params)
    else:
        source = ContextSource(metadatas[0])
    begin, end = Mock(), Mock()
    monkeypatch.setattr(torch.npu, "graph_task_update_begin", begin)
    monkeypatch.setattr(torch.npu, "graph_task_update_end", end)

    resolved = UpdatableGraph.resolve_tasks(graph, source)
    for task in resolved:
        task.apply(stream)

    assert operation.call_count == len(expected)
    for call, (decode, history, key, rope, output, lse) in zip(operation.call_args_list, expected):
        kwargs = call.kwargs
        assert kwargs["query"] is query and kwargs["key"] is key and kwargs["key_rope"] is rope
        assert kwargs["workspace"] is workspace
        assert kwargs["out"][0] is output and kwargs["out"][1] is lse
        assert kwargs["block_table"] is (decode.block_table if history else None)
        assert kwargs["actual_seq_qlen"] == decode.actual_seq_lengths_q
        assert kwargs["actual_seq_kvlen"] == (decode.cp_history_seq_len if history else decode.actual_seq_lengths_q)
        if history:
            assert key.stride(0) == rope.stride(0) == 24
            assert key.storage_offset() == 7 and rope.storage_offset() == 15
    assert [call.args for call in begin.call_args_list] == [(stream, task.handle) for task in tasks]
    assert end.call_count == len(tasks)
    for task in tasks:
        task.event.record.assert_called_once_with(stream)
        assert task.kwargs["actual_seq_kvlen"] == [0, 0]


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize("num_tokens,expected", [(0, [2, 3]), (2, [2, 3]), (4, [2, 3, 0, 0])])
def test_mla_dcp_full_provider_refreshes_and_pads_lengths(as_tensor, num_tokens, expected):
    lengths = torch.tensor([2, 3]) if as_tensor else [2, 3]
    decode = SimpleNamespace(cp_seq_len=lengths)
    provider = MLADCPParamProvider("layer", CPKVScope.FULL, num_tokens)

    assert provider.layer_name == ("layer", CPKVScope.FULL)
    assert provider.resolve({"layer": SimpleNamespace(decode=decode)}) == {"actual_seq_kvlen": expected}


@pytest.mark.parametrize("scope", [CPKVScope.HISTORY, CPKVScope.CURRENT])
def test_mla_dcp_split_provider_binds_current_metadata(scope):
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    decode = SimpleNamespace(actual_seq_lengths_q=[1, 2], cp_history_seq_len=torch.tensor([3, 4]), block_table=table)
    provider = MLADCPParamProvider("layer", scope, 2)
    query = torch.randn(2, 2, 4)
    task = GraphUpdateTask(
        Mock(),
        {"query": query, "actual_seq_qlen": [0, 0], "actual_seq_kvlen": [0, 0], "block_table": None},
        provider,
        0,
        Mock(),
        Mock(),
    )

    updated = task.bind(provider.resolve({"layer": SimpleNamespace(decode=decode)}))

    assert updated.kwargs["query"] is query
    assert updated.kwargs["actual_seq_qlen"] == [1, 2]
    assert updated.kwargs["actual_seq_kvlen"] == ([3, 4] if scope is CPKVScope.HISTORY else [1, 2])
    assert updated.kwargs["block_table"] is (table if scope is CPKVScope.HISTORY else None)
    assert task.kwargs["actual_seq_kvlen"] == [0, 0]


@pytest.mark.parametrize("scope", [CPKVScope.FULL, CPKVScope.HISTORY, CPKVScope.CURRENT])
def test_mla_dcp_provider_rejects_missing_decode_metadata(scope):
    provider = MLADCPParamProvider("layer", scope, 2)
    with pytest.raises(AssertionError):
        provider.resolve({"layer": SimpleNamespace(decode=None)})
