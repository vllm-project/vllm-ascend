import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig

from tests.ut.base import TestBase


class TestNPUWorker(TestBase):

    def setUp(self):
        """Setup test environment"""
        # Create configuration mocks
        self.cache_config_mock = MagicMock(spec=CacheConfig)
        self.cache_config_mock.cache_dtype = "auto"

        self.model_config_mock = MagicMock(spec=ModelConfig)
        self.model_config_mock.dtype = torch.float16
        self.model_config_mock.trust_remote_code = False

        self.parallel_config_mock = MagicMock(spec=ParallelConfig)

        self.vllm_config_mock = MagicMock(spec=VllmConfig)
        self.vllm_config_mock.cache_config = self.cache_config_mock
        self.vllm_config_mock.model_config = self.model_config_mock
        self.vllm_config_mock.parallel_config = self.parallel_config_mock
        self.vllm_config_mock.additional_config = None
        self.vllm_config_mock.load_config = None
        self.vllm_config_mock.scheduler_config = None
        self.vllm_config_mock.device_config = None
        self.vllm_config_mock.compilation_config = None

        self.local_rank = 0
        self.rank = 0
        self.distributed_init_method = "tcp://localhost:12345"
        self.is_driver_worker = False

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.ops")
    @patch("vllm_ascend.worker.worker_v1._register_atb_extensions")
    @patch("vllm_ascend.worker.worker_v1.register_ascend_customop")
    @patch("vllm_ascend.worker.worker_v1.init_ascend_config")
    @patch("vllm_ascend.worker.worker_v1.init_ascend_soc_version")
    @patch("vllm_ascend.worker.worker_v1.try_register_lib")
    @patch("vllm.utils.init_cached_hf_modules")
    @patch("vllm_ascend.worker.worker_v1.NPUWorker._init_profiler")
    def test_init_npu_worker_normal_case(
        self,
        mock_init_profiler,
        mock_init_cached_hf_modules,
        mock_try_register_lib,
        mock_init_ascend_soc_version,
        mock_init_ascend_config,
        mock_register_ascend_customop,
        mock_register_atb_extensions,
        mock_ops,
        mock_adapt_patch,
    ):
        """Test NPUWorker normal initialization"""
        # Setup mock behavior
        mock_ops.register_dummy_fusion_op.return_value = None

        # Import and create NPUWorker instance
        from vllm_ascend.worker.worker_v1 import NPUWorker

        worker = NPUWorker(
            vllm_config=self.vllm_config_mock,
            local_rank=self.local_rank,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
            is_driver_worker=self.is_driver_worker,
        )

        # Verify initialization call order
        mock_adapt_patch.assert_called_once()
        mock_ops.register_dummy_fusion_op.assert_called_once()
        mock_register_atb_extensions.assert_called_once()
        mock_register_ascend_customop.assert_called_once()
        mock_init_ascend_config.assert_called_once_with(self.vllm_config_mock)
        mock_init_ascend_soc_version.assert_called_once()

        # Verify try_register_lib call
        mock_try_register_lib.assert_called_once_with(
            "mindie_turbo",
            "MindIE Turbo is installed. vLLM inference will be accelerated with MindIE Turbo.",
        )

        # Verify cache_dtype setting
        self.assertEqual(worker.cache_dtype, torch.float16)
        mock_init_profiler.assert_called_once()

        # Verify init_cached_hf_modules is not called (trust_remote_code=False)
        mock_init_cached_hf_modules.assert_not_called()

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.ops")
    @patch("vllm_ascend.worker.worker_v1._register_atb_extensions")
    @patch("vllm_ascend.worker.worker_v1.register_ascend_customop")
    @patch("vllm_ascend.worker.worker_v1.init_ascend_config")
    @patch("vllm_ascend.worker.worker_v1.init_ascend_soc_version")
    @patch("vllm_ascend.worker.worker_v1.try_register_lib")
    @patch("vllm.utils.init_cached_hf_modules")
    @patch("vllm_ascend.worker.worker_v1.NPUWorker._init_profiler")
    def test_init_npu_worker_with_trust_remote_code(
        self,
        mock_init_profiler,
        mock_init_cached_hf_modules,
        mock_try_register_lib,
        mock_init_ascend_soc_version,
        mock_init_ascend_config,
        mock_register_ascend_customop,
        mock_register_atb_extensions,
        mock_ops,
        mock_adapt_patch,
    ):
        """Test NPUWorker initialization with trust_remote_code=True"""
        # Set trust_remote_code=True
        self.model_config_mock.trust_remote_code = True
        mock_ops.register_dummy_fusion_op.return_value = None

        # Create NPUWorker instance
        from vllm_ascend.worker.worker_v1 import NPUWorker

        _ = NPUWorker(
            vllm_config=self.vllm_config_mock,
            local_rank=self.local_rank,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
            is_driver_worker=self.is_driver_worker,
        )

        # Verify init_cached_hf_modules is called (trust_remote_code=True)
        mock_init_cached_hf_modules.assert_called_once()

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.ops")
    @patch("vllm_ascend.worker.worker_v1._register_atb_extensions")
    @patch("vllm_ascend.worker.worker_v1.register_ascend_customop")
    @patch("vllm_ascend.worker.worker_v1.init_ascend_config")
    @patch("vllm_ascend.worker.worker_v1.init_ascend_soc_version")
    @patch("vllm_ascend.worker.worker_v1.try_register_lib")
    @patch("vllm.utils.init_cached_hf_modules")
    @patch("vllm_ascend.worker.worker_v1.NPUWorker._init_profiler")
    def test_init_npu_worker_with_custom_cache_dtype(
        self,
        mock_init_profiler,
        mock_init_cached_hf_modules,
        mock_try_register_lib,
        mock_init_ascend_soc_version,
        mock_init_ascend_config,
        mock_register_ascend_customop,
        mock_register_atb_extensions,
        mock_ops,
        mock_adapt_patch,
    ):
        """Test NPUWorker initialization with custom cache_dtype"""
        # Set custom cache_dtype
        self.cache_config_mock.cache_dtype = "float32"
        mock_ops.register_dummy_fusion_op.return_value = None

        # Create NPUWorker instance
        from vllm_ascend.worker.worker_v1 import NPUWorker

        with patch("vllm.utils.STR_DTYPE_TO_TORCH_DTYPE",
                   {"float32": torch.float32}):
            worker = NPUWorker(
                vllm_config=self.vllm_config_mock,
                local_rank=self.local_rank,
                rank=self.rank,
                distributed_init_method=self.distributed_init_method,
                is_driver_worker=self.is_driver_worker,
            )

        # Verify cache_dtype is set to custom value
        self.assertEqual(worker.cache_dtype, torch.float32)

    def test_initialize_cache(self):
        """Test initialize_cache method"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create a simple worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.cache_config = MagicMock()

            # Test initialize_cache
            worker.initialize_cache(100, 50)

            # Verify parameter setting
            self.assertEqual(worker.cache_config.num_gpu_blocks, 100)
            self.assertEqual(worker.cache_config.num_cpu_blocks, 50)

    @patch("vllm_ascend.worker.worker_v1.sleep_mode_enabled")
    @patch("vllm_ascend.worker.worker_v1.NPUPlatform")
    @patch("vllm_ascend.worker.worker_v1.CaMemAllocator")
    @patch("vllm_ascend.worker.worker_v1.logger")
    def test_sleep_mode_enabled(self, mock_logger, mock_allocator_class,
                                mock_platform, mock_sleep_mode_enabled):
        """Test sleep method when sleep mode is enabled"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Setup mock
        mock_sleep_mode_enabled.return_value = True
        mock_platform.mem_get_info.side_effect = [
            (1000, 2000),
            (1200, 2000),
        ]  # before, after
        mock_allocator = MagicMock()
        mock_allocator_class.get_instance.return_value = mock_allocator

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()

            # Test sleep method
            worker.sleep(level=1)

            # Verify calls
            mock_sleep_mode_enabled.assert_called_once()
            mock_allocator.sleep.assert_called_once_with(
                offload_tags=("weights", ))
            self.assertEqual(mock_platform.mem_get_info.call_count,
                             2)  # Called 2 times in sleep method
            # Verify log output
            mock_logger.info.assert_called_once()

    @patch("vllm_ascend.worker.worker_v1.sleep_mode_enabled")
    def test_sleep_mode_disabled_raises_error(self, mock_sleep_mode_enabled):
        """Test sleep method raises exception when sleep mode is disabled"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Set sleep mode disabled
        mock_sleep_mode_enabled.return_value = False

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()

            # Test sleep method should raise exception
            with self.assertRaises(ValueError) as cm:
                worker.sleep()

            self.assertIn("Sleep mode is not enabled", str(cm.exception))

    @patch("vllm_ascend.worker.worker_v1.sleep_mode_enabled")
    @patch("vllm_ascend.worker.worker_v1.CaMemAllocator")
    def test_wake_up_mode_enabled(self, mock_allocator_class,
                                  mock_sleep_mode_enabled):
        """Test wake_up method when sleep mode is enabled"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Setup mock
        mock_sleep_mode_enabled.return_value = True
        mock_allocator = MagicMock()
        mock_allocator_class.get_instance.return_value = mock_allocator

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()

            # Test wake_up method
            worker.wake_up(tags=["test_tag"])

            # Verify calls
            mock_sleep_mode_enabled.assert_called_once()
            mock_allocator.wake_up.assert_called_once_with(tags=["test_tag"])

    @patch("vllm_ascend.worker.worker_v1.sleep_mode_enabled")
    def test_wake_up_mode_disabled_raises_error(self, mock_sleep_mode_enabled):
        """Test wake_up method raises exception when sleep mode is disabled"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Set sleep mode disabled
        mock_sleep_mode_enabled.return_value = False

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()

            # Test wake_up method should raise exception
            with self.assertRaises(ValueError) as cm:
                worker.wake_up()

            self.assertIn("Sleep mode is not enabled", str(cm.exception))

    @patch(
        "vllm_ascend.worker.worker_v1.NPUWorker._init_worker_distributed_environment"
    )
    @patch("vllm_ascend.worker.worker_v1.NPUPlatform")
    def test_init_device(self, mock_platform, mock_init_dist_env):
        """Test _init_device method"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Setup mock
        mock_platform.mem_get_info.return_value = (1000, 2000)

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.local_rank = 1
            worker.model_config = MagicMock()
            worker.model_config.seed = 42

            # Test _init_device
            result = worker._init_device()

            # Verify NPUPlatform.set_device is called
            mock_platform.set_device.assert_called_once()
            # Verify the parameter passed to set_device is a torch.device object
            call_args = mock_platform.set_device.call_args[0][0]
            self.assertEqual(str(call_args), "npu:1")

            mock_platform.empty_cache.assert_called_once()
            mock_platform.seed_everything.assert_called_once_with(42)
            mock_platform.mem_get_info.assert_called_once(
            )  # Called once in _init_device method
            mock_init_dist_env.assert_called_once(
            )  # Verify distributed initialization is called

            # Verify return value is a torch.device object
            self.assertEqual(str(result), "npu:1")
            self.assertEqual(worker.init_npu_memory, 1000)

    def test_profile_start_stop(self):
        """Test profile method start and stop"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            mock_profiler = MagicMock()
            worker.profiler = mock_profiler

            # Test start profiler
            worker.profile(is_start=True)
            mock_profiler.start.assert_called_once()

            # Test stop profiler
            worker.profile(is_start=False)
            mock_profiler.stop.assert_called_once()

    def test_profile_no_profiler_raises_error(self):
        """Test profile method raises exception when profiler is not available"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler = None

            # Test should raise exception
            with self.assertRaises(RuntimeError) as cm:
                worker.profile()

            self.assertIn("Profiler is not enabled", str(cm.exception))

    def test_lora_methods(self):
        """Test LoRA related methods"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            mock_model_runner = MagicMock()
            worker.model_runner = mock_model_runner

            # Set return values
            mock_model_runner.add_lora.return_value = True
            mock_model_runner.remove_lora.return_value = True
            mock_model_runner.list_loras.return_value = {1, 2, 3}
            mock_model_runner.pin_lora.return_value = True

            # Test each method
            mock_request = MagicMock()
            self.assertTrue(worker.add_lora(mock_request))
            mock_model_runner.add_lora.assert_called_once_with(mock_request)

            self.assertTrue(worker.remove_lora(1))
            mock_model_runner.remove_lora.assert_called_once_with(1)

            self.assertEqual(worker.list_loras(), {1, 2, 3})
            mock_model_runner.list_loras.assert_called_once()

            self.assertTrue(worker.pin_lora(2))
            mock_model_runner.pin_lora.assert_called_once_with(2)

    def test_get_methods(self):
        """Test various get methods"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            mock_model_runner = MagicMock()
            worker.model_runner = mock_model_runner

            # Set return values
            mock_model = MagicMock()
            mock_kv_cache_spec = {"test": "spec"}
            mock_pooling_tasks = ["task1", "task2"]
            mock_supported_tasks = ("task1", "task2")

            mock_model_runner.get_model.return_value = mock_model
            mock_model_runner.get_kv_cache_spec.return_value = mock_kv_cache_spec
            mock_model_runner.get_supported_pooling_tasks.return_value = (
                mock_pooling_tasks)
            mock_model_runner.get_supported_tasks.return_value = mock_supported_tasks

            # Test each get method
            self.assertEqual(worker.get_model(), mock_model)
            self.assertEqual(worker.get_kv_cache_spec(), mock_kv_cache_spec)
            self.assertEqual(worker.get_supported_pooling_tasks(),
                             mock_pooling_tasks)
            self.assertEqual(worker.get_supported_tasks(),
                             mock_supported_tasks)

    def test_execute_dummy_batch(self):
        """Test execute_dummy_batch method"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            mock_model_runner = MagicMock()
            worker.model_runner = mock_model_runner

            # Test execute_dummy_batch
            worker.execute_dummy_batch()

            # Verify call
            mock_model_runner._dummy_run.assert_called_once_with(1)

    @patch("vllm_ascend.worker.worker_v1.envs_vllm")
    @patch("vllm_ascend.worker.worker_v1.logger")
    @patch("torch_npu.profiler._ExperimentalConfig")
    @patch("torch_npu.profiler.profile")
    @patch("torch_npu.profiler.tensorboard_trace_handler")
    @patch("torch_npu.profiler.ExportType")
    @patch("torch_npu.profiler.ProfilerLevel")
    @patch("torch_npu.profiler.AiCMetrics")
    @patch("torch_npu.profiler.ProfilerActivity")
    def test_init_profiler_enabled(
        self,
        mock_profiler_activity,
        mock_aic_metrics,
        mock_profiler_level,
        mock_export_type,
        mock_trace_handler,
        mock_profile,
        mock_experimental_config,
        mock_logger,
        mock_envs_vllm,
    ):
        """Test _init_profiler method - profiler enabled case with stack and memory profiling enabled"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Set environment variables to enable profiler
        mock_envs_vllm.VLLM_TORCH_PROFILER_DIR = "/path/to/traces"
        mock_envs_vllm.VLLM_TORCH_PROFILER_WITH_STACK = True
        mock_envs_vllm.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY = True

        # Set enum mocks
        mock_export_type.Text = "Text"
        mock_profiler_level.Level1 = "Level1"
        mock_aic_metrics.AiCoreNone = "AiCoreNone"
        mock_profiler_activity.CPU = "CPU"
        mock_profiler_activity.NPU = "NPU"

        # Set mock return values
        mock_experimental_config_instance = MagicMock()
        mock_experimental_config.return_value = mock_experimental_config_instance
        mock_trace_handler_instance = MagicMock()
        mock_trace_handler.return_value = mock_trace_handler_instance
        mock_profiler_instance = MagicMock()
        mock_profile.return_value = mock_profiler_instance

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()

            # Test _init_profiler
            result = worker._init_profiler()

            # Verify log output
            mock_logger.info.assert_called_once_with(
                "Profiling enabled. Traces will be saved to: %s",
                "/path/to/traces")

            # Verify ExperimentalConfig creation
            mock_experimental_config.assert_called_once()
            config_call = mock_experimental_config.call_args
            config_kwargs = config_call.kwargs

            # Verify configuration parameters
            expected_config = {
                "export_type": "Text",
                "profiler_level": "Level1",
                "msprof_tx": False,
                "aic_metrics": "AiCoreNone",
                "l2_cache": False,
                "op_attr": False,
                "data_simplification": False,
                "record_op_args": False,
                "gc_detect_threshold": None,
            }
            for key, expected_value in expected_config.items():
                self.assertEqual(config_kwargs[key], expected_value)

            # Verify trace handler creation
            mock_trace_handler.assert_called_once_with("/path/to/traces")

            # Verify profiler creation
            mock_profile.assert_called_once()
            profile_call = mock_profile.call_args
            profile_kwargs = profile_call.kwargs

            # Verify profiler parameters
            expected_activities = ["CPU", "NPU"]
            self.assertEqual(profile_kwargs["activities"], expected_activities)
            self.assertTrue(profile_kwargs["with_stack"])
            self.assertTrue(profile_kwargs["profile_memory"])
            self.assertFalse(profile_kwargs["with_modules"])
            self.assertEqual(profile_kwargs["experimental_config"],
                             mock_experimental_config_instance)
            self.assertEqual(profile_kwargs["on_trace_ready"],
                             mock_trace_handler_instance)

            # Verify return value
            self.assertEqual(result, mock_profiler_instance)

    @patch("vllm_ascend.worker.worker_v1.envs_vllm")
    def test_init_profiler_disabled(self, mock_envs_vllm):
        """Test _init_profiler method - profiler disabled case"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Set environment variable to disable profiler
        mock_envs_vllm.VLLM_TORCH_PROFILER_DIR = None

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()

            # Test _init_profiler
            result = worker._init_profiler()

            # Verify returns None
            self.assertIsNone(result)

    @patch("vllm_ascend.worker.worker_v1.envs_vllm")
    def test_init_profiler_empty_dir(self, mock_envs_vllm):
        """Test _init_profiler method - empty directory string case"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Set environment variable to empty string
        mock_envs_vllm.VLLM_TORCH_PROFILER_DIR = ""

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()

            # Test _init_profiler
            result = worker._init_profiler()

            # Verify returns None (empty string is considered false)
            self.assertIsNone(result)

    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.clear_npu_memory")
    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.empty_cache")
    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.mem_get_info")
    @patch("torch_npu.npu.memory_stats")
    @patch("torch_npu.npu.mem_get_info")
    @patch("vllm_ascend.worker.worker_v1.logger")
    def test_determine_available_memory_normal_case(
        self,
        mock_logger,
        mock_torch_mem_get_info,
        mock_torch_memory_stats,
        mock_platform_mem_get_info,
        mock_platform_empty_cache,
        mock_platform_clear_npu_memory,
    ):
        """Test determine_available_memory normal case (no non-torch memory allocation)"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Setup mock - test case without non-torch memory allocation
        mock_platform_mem_get_info.side_effect = [
            (8000, 10000),  # 1st call: before profile execution
            (7000, 10000),  # 2nd call: after profile execution
        ]
        mock_torch_memory_stats.side_effect = [
            {
                "allocated_bytes.all.peak": 2000
            },  # peak memory
            {
                "allocated_bytes.all.current": 3000
            },  # current allocated = total_allocated_bytes
        ]
        # Mock setup to simulate memory change between calls, exposing potential race condition
        # The implementation calls torch_npu.npu.mem_get_info() twice in total_allocated_bytes calculation
        # which is not atomic and can lead to incorrect memory calculations
        mock_torch_mem_get_info.side_effect = [
            (7000, 10000),  # First call for total_allocated_bytes calculation
            (
                6000,
                10000,
            ),  # Second call for total_allocated_bytes calculation, simulating an allocation
            (6000, 10000),  # Additional calls for other parts of the method
            (6000, 10000),
        ]

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.init_npu_memory = (
                8500  # Initial memory greater than current free memory
            )
            worker.model_runner = MagicMock()
            worker.cache_config = MagicMock()
            worker.cache_config.gpu_memory_utilization = 0.8

            # Test determine_available_memory
            result = worker.determine_available_memory()

            # Verify call count and order
            mock_platform_clear_npu_memory.assert_called_once()
            self.assertEqual(mock_platform_mem_get_info.call_count, 2)
            worker.model_runner.profile_run.assert_called_once()
            mock_platform_empty_cache.assert_called_once()

            # Verify calculation result with race condition simulation
            # Calculation logic:
            # total_allocated_bytes = torch_npu.npu.mem_get_info()[1] - torch_npu.npu.mem_get_info()[0]
            #                       = 10000 - 7000 = 3000 (first call)
            #                       = 10000 - 6000 = 4000 (second call, memory changed!)
            # This exposes the race condition where memory state changes between calls
            # non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
            #                       = 4000 - 3000 = 1000  # Non-torch memory allocation detected
            # peak_memory = torch_peak_memory + non_torch_allocations
            #             = 2000 + 1000 = 3000
            # available = total_npu_memory * gpu_memory_utilization - peak_memory
            #           = 10000 * 0.8 - 3000 = 5000
            expected_result = max(0, int(10000 * 0.8 - 3000))
            self.assertEqual(result, expected_result)

            # Verify log output
            mock_logger.info.assert_called_once()

    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.clear_npu_memory")
    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.empty_cache")
    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.mem_get_info")
    @patch("torch_npu.npu.memory_stats")
    @patch("torch_npu.npu.mem_get_info")
    def test_determine_available_memory_with_non_torch_allocations(
        self,
        mock_torch_mem_get_info,
        mock_torch_memory_stats,
        mock_platform_mem_get_info,
        mock_platform_empty_cache,
        mock_platform_clear_npu_memory,
    ):
        """Test determine_available_memory with significant non-torch memory allocation"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Setup mock - test case with large non-torch memory allocation
        mock_platform_mem_get_info.side_effect = [
            (8000, 10000),  # 1st call
            (7000, 10000),  # 2nd call
        ]
        mock_torch_memory_stats.side_effect = [
            {
                "allocated_bytes.all.peak": 1500
            },  # peak memory
            {
                "allocated_bytes.all.current": 1000
            },  # current allocated
        ]
        # Mock setup to expose race condition in total_allocated_bytes calculation
        # Setup non-torch allocations > 0 case with memory change simulation
        mock_torch_mem_get_info.side_effect = [
            (6000, 10000),  # First call for total_allocated_bytes calculation
            (
                5000,
                10000,
            ),  # Second call for total_allocated_bytes calculation, simulating allocation
            (5000, 10000),  # Additional calls for other parts of the method
            (5000, 10000),
        ]

        # 创建 worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.init_npu_memory = 8500
            worker.model_runner = MagicMock()
            worker.cache_config = MagicMock()
            worker.cache_config.gpu_memory_utilization = 0.9

            # Test determine_available_memory
            result = worker.determine_available_memory()

            # Verify result: case with large non-torch memory allocation and race condition
            # total_allocated_bytes = torch_npu.npu.mem_get_info()[1] - torch_npu.npu.mem_get_info()[0]
            #                       = 10000 - 6000 = 4000 (first call)
            #                       = 10000 - 5000 = 5000 (second call, memory changed!)
            # This exposes the race condition where memory allocation occurs between calls
            # non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
            #                       = 5000 - 1000 = 4000  # Significant non-torch allocation detected
            # peak_memory = torch_peak_memory + non_torch_allocations
            #             = 1500 + 4000 = 5500
            # available = total_npu_memory * gpu_memory_utilization - peak_memory
            #           = 10000 * 0.9 - 5500 = 3500
            expected_result = max(0, int(10000 * 0.9 - 5500))
            self.assertEqual(result, expected_result)

    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.clear_npu_memory")
    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.mem_get_info")
    def test_determine_available_memory_memory_profiling_error(
            self, mock_platform_mem_get_info, mock_platform_clear_npu_memory):
        """Test determine_available_memory throws exception on memory profiling error"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Setup mock: initial memory less than current free memory (error case)
        mock_platform_mem_get_info.side_effect = [
            (8000, 10000),  # 1st call
            (9000, 10000),  # 2nd call: free memory increased instead
        ]

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.init_npu_memory = 8500  # Initial memory < current free memory 9000
            worker.model_runner = MagicMock()
            worker.cache_config = MagicMock()
            worker.cache_config.gpu_memory_utilization = 0.8

            # Test should throw exception
            with self.assertRaises(AssertionError) as cm:
                worker.determine_available_memory()

            self.assertIn("Error in memory profiling", str(cm.exception))

    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.clear_npu_memory")
    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.empty_cache")
    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.mem_get_info")
    @patch("torch_npu.npu.memory_stats")
    @patch("torch_npu.npu.mem_get_info")
    def test_determine_available_memory_negative_result(
        self,
        mock_torch_mem_get_info,
        mock_torch_memory_stats,
        mock_platform_mem_get_info,
        mock_platform_empty_cache,
        mock_platform_clear_npu_memory,
    ):
        """Test determine_available_memory returns 0 when result is negative"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Setup mock: high peak memory causes negative available memory
        mock_platform_mem_get_info.side_effect = [
            (8000, 10000),  # 1st call
            (3000, 10000),  # 2nd call
        ]
        mock_torch_memory_stats.side_effect = [
            {
                "allocated_bytes.all.peak": 9000
            },  # High peak memory
            {
                "allocated_bytes.all.current": 7000
            },
        ]
        # Mock setup to expose race condition even in negative result scenarios
        mock_torch_mem_get_info.side_effect = [
            (3000, 10000),  # First call for total_allocated_bytes calculation
            (
                2000,
                10000,
            ),  # Second call for total_allocated_bytes calculation, simulating more allocation
            (2000, 10000),  # Additional calls for other parts of the method
            (2000, 10000),
        ]

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.init_npu_memory = 8500
            worker.model_runner = MagicMock()
            worker.cache_config = MagicMock()
            worker.cache_config.gpu_memory_utilization = 0.8

            # Test determine_available_memory
            result = worker.determine_available_memory()

            # Verify result is 0 (not negative) even with race condition
            # total_allocated_bytes = torch_npu.npu.mem_get_info()[1] - torch_npu.npu.mem_get_info()[0]
            #                       = 10000 - 3000 = 7000 (first call)
            #                       = 10000 - 2000 = 8000 (second call, more memory allocated!)
            # non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
            #                       = 8000 - 7000 = 1000  # Additional non-torch allocation detected
            # peak_memory = torch_peak_memory + non_torch_allocations
            #             = 9000 + 1000 = 10000
            # available = total_npu_memory * gpu_memory_utilization - peak_memory
            #           = 10000 * 0.8 - 10000 = -2000, max(0, -2000) = 0
            self.assertEqual(result, 0)

    def test_execute_model_first_rank(self):
        """Test execute_model method - first rank case"""
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with (
                patch.object(NPUWorker, "__init__", lambda x, **kwargs: None),
                patch("vllm_ascend.worker.worker_v1.get_pp_group") as
                mock_get_pp_group,
        ):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"

            # Set as first rank
            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = True
            mock_pp_group.is_last_rank = True
            mock_get_pp_group.return_value = mock_pp_group

            # Mock scheduler_output and return result
            mock_scheduler_output = MagicMock()
            # Create a real ModelRunnerOutput instance or mock
            mock_model_output = MagicMock(spec=ModelRunnerOutput)
            worker.model_runner.execute_model.return_value = mock_model_output

            # Test execute_model
            result = worker.execute_model(mock_scheduler_output)

            # Verify call
            worker.model_runner.execute_model.assert_called_once_with(
                mock_scheduler_output, None)
            self.assertEqual(result, mock_model_output)

    @patch("vllm_ascend.worker.worker_v1.get_pp_group")
    @patch("vllm_ascend.worker.worker_v1.get_tp_group")
    @patch("vllm_ascend.worker.worker_v1.has_kv_transfer_group")
    def test_execute_model_middle_rank(self, mock_has_kv_transfer_group,
                                       mock_get_tp_group, mock_get_pp_group):
        """Test execute_model method - middle rank case"""
        from vllm.sequence import IntermediateTensors

        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"

            # Set as middle rank (not first, not last)
            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = False
            mock_pp_group.is_last_rank = False
            mock_get_pp_group.return_value = mock_pp_group

            # Setup tensor reception data
            mock_pp_group.recv_tensor_dict.return_value = {"tensor": "data"}

            # Mock return IntermediateTensors - use real type
            mock_intermediate_output = MagicMock(spec=IntermediateTensors)
            mock_intermediate_output.tensors = {"output_tensor": "data"}
            mock_intermediate_output.kv_connector_output = (
                None  # Set to None to trigger return None
            )
            worker.model_runner.execute_model.return_value = mock_intermediate_output

            # Set has_kv_transfer_group returns False
            mock_has_kv_transfer_group.return_value = False

            mock_scheduler_output = MagicMock()

            # Test execute_model
            result = worker.execute_model(mock_scheduler_output)

            # Verify tensor reception
            mock_pp_group.recv_tensor_dict.assert_called_once()

            # Verify model execution with intermediate_tensors
            # Second parameter should be IntermediateTensors instance
            worker.model_runner.execute_model.assert_called_once()
            args, kwargs = worker.model_runner.execute_model.call_args
            self.assertEqual(args[0], mock_scheduler_output)
            self.assertIsInstance(args[1], IntermediateTensors)

            # Verify tensor sending
            mock_pp_group.send_tensor_dict.assert_called_once()

            # Middle rank without kv_transfer_group should return None
            self.assertIsNone(result)

    def test_execute_model_external_launcher(self):
        """Test execute_model method - external_launcher mode"""
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with (
                patch.object(NPUWorker, "__init__", lambda x, **kwargs: None),
                patch("vllm_ascend.worker.worker_v1.get_pp_group") as
                mock_get_pp_group,
        ):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = (
                "external_launcher")

            # Set as non-last rank
            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = True
            mock_pp_group.is_last_rank = False
            mock_get_pp_group.return_value = mock_pp_group

            # Mock return result
            mock_scheduler_output = MagicMock()
            mock_model_output = MagicMock(spec=ModelRunnerOutput)
            worker.model_runner.execute_model.return_value = mock_model_output

            # Test execute_model
            result = worker.execute_model(mock_scheduler_output)

            # In external_launcher mode, it doesn't enter middle processing logic, returns result directly
            self.assertEqual(result, mock_model_output)

    @patch("vllm_ascend.worker.worker_v1.CaMemAllocator")
    def test_load_model_with_sleep_mode(self, mock_allocator_class):
        """Test load_model method - with sleep mode enabled"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.model_config = MagicMock()
            worker.vllm_config.model_config.enable_sleep_mode = True

            # Setup allocator mock
            mock_allocator = MagicMock()
            mock_allocator.get_current_usage.return_value = 0
            mock_context = MagicMock()
            mock_allocator.use_memory_pool.return_value = mock_context
            mock_allocator_class.get_instance.return_value = mock_allocator

            # Test load_model
            worker.load_model()

            # Verify calls
            mock_allocator_class.get_instance.assert_called_once()
            mock_allocator.get_current_usage.assert_called_once()
            mock_allocator.use_memory_pool.assert_called_once_with(
                tag="weights")
            worker.model_runner.load_model.assert_called_once()

    def test_load_model_without_sleep_mode(self):
        """Test load_model method - without sleep mode enabled"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.model_config = MagicMock()
            worker.vllm_config.model_config.enable_sleep_mode = False

            # Test load_model
            worker.load_model()

            # Verify calls
            worker.model_runner.load_model.assert_called_once()

    @patch("vllm_ascend.worker.worker_v1.CaMemAllocator")
    def test_load_model_sleep_mode_assertion_error(self, mock_allocator_class):
        """Test load_model method - assertion error in sleep mode"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.model_config = MagicMock()
            worker.vllm_config.model_config.enable_sleep_mode = True

            # Setup allocator mock - current usage is not 0
            mock_allocator = MagicMock()
            mock_allocator.get_current_usage.return_value = 100  # Non-zero value
            mock_allocator_class.get_instance.return_value = mock_allocator

            # Test should throw assertion error
            with self.assertRaises(AssertionError) as cm:
                worker.load_model()

            self.assertIn("Sleep mode can only be", str(cm.exception))

    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.seed_everything")
    @patch("vllm_ascend.worker.worker_v1.logger")
    @patch("vllm_ascend.worker.worker_v1.NPUWorker._warm_up_atb")
    def test_compile_or_warm_up_model_with_eager_mode(self, mock_warm_up_atb,
                                                      mock_logger,
                                                      mock_seed_everything):
        """Test compile_or_warm_up_model method - eager mode"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.model_config = MagicMock()
            worker.model_config.enforce_eager = True
            worker.model_config.seed = 12345

            # Setup compilation config
            worker.vllm_config.compilation_config = MagicMock()
            worker.vllm_config.compilation_config.compile_sizes = [1, 4, 8, 16]
            worker.vllm_config.compilation_config.cudagraph_capture_sizes = [
                4, 8
            ]

            # Test compile_or_warm_up_model
            worker.compile_or_warm_up_model()

            # Verify _dummy_run call count and order (by size descending)
            expected_calls = [
                unittest.mock.call(16),
                unittest.mock.call(8),
                unittest.mock.call(4),
                unittest.mock.call(1),
            ]
            worker.model_runner._dummy_run.assert_has_calls(expected_calls)

            # Should not call capture_model in eager mode
            worker.model_runner.capture_model.assert_not_called()

            # Verify log output
            self.assertEqual(mock_logger.info.call_count, 4)

            # Verify seed setting
            mock_seed_everything.assert_called_once_with(12345)

            # Verify atb warm up
            mock_warm_up_atb.assert_called_once()

    @patch("vllm_ascend.worker.worker_v1.NPUPlatform.seed_everything")
    @patch("vllm_ascend.worker.worker_v1.logger")
    @patch("vllm_ascend.worker.worker_v1.NPUWorker._warm_up_atb")
    def test_compile_or_warm_up_model_with_graph_capture(
            self, mock_warm_up_atb, mock_logger, mock_seed_everything):
        """Test compile_or_warm_up_model method - with graph capture enabled"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.model_config = MagicMock()
            worker.model_config.enforce_eager = False  # Enable graph capture
            worker.model_config.seed = 67890

            # Setup compilation config
            worker.vllm_config.compilation_config = MagicMock()
            worker.vllm_config.compilation_config.compile_sizes = [1, 4, 8, 16]
            worker.vllm_config.compilation_config.cudagraph_capture_sizes = [
                4, 8
            ]

            # Test compile_or_warm_up_model
            worker.compile_or_warm_up_model()

            # Verify only call _dummy_run for sizes not in cudagraph_capture_sizes
            expected_calls = [unittest.mock.call(16), unittest.mock.call(1)]
            worker.model_runner._dummy_run.assert_has_calls(expected_calls)

            # Should call capture_model in non-eager mode
            worker.model_runner.capture_model.assert_called_once()

            # Verify seed setting
            mock_seed_everything.assert_called_once_with(67890)

            # Verify atb warm up
            mock_warm_up_atb.assert_called_once()

    @patch("vllm_ascend.worker.worker_v1.CaMemAllocator")
    def test_initialize_from_config_with_sleep_mode(self,
                                                    mock_allocator_class):
        """Test initialize_from_config method - with sleep mode enabled"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.model_config = MagicMock()
            worker.vllm_config.model_config.enable_sleep_mode = True

            # Setup allocator mock
            mock_allocator = MagicMock()
            mock_context = MagicMock()
            mock_allocator.use_memory_pool.return_value = mock_context
            mock_allocator_class.get_instance.return_value = mock_allocator

            # Create mock kv_cache_config
            mock_kv_cache_config = MagicMock()

            # Test initialize_from_config
            worker.initialize_from_config(mock_kv_cache_config)

            # Verify calls
            mock_allocator_class.get_instance.assert_called_once()
            mock_allocator.use_memory_pool.assert_called_once_with(
                tag="kv_cache")
            worker.model_runner.initialize_kv_cache.assert_called_once_with(
                mock_kv_cache_config)

    def test_initialize_from_config_without_sleep_mode(self):
        """Test initialize_from_config method - without sleep mode enabled"""
        from vllm_ascend.worker.worker_v1 import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.model_config = MagicMock()
            worker.vllm_config.model_config.enable_sleep_mode = False

            # Create mock kv_cache_config
            mock_kv_cache_config = MagicMock()

            # Test initialize_from_config
            worker.initialize_from_config(mock_kv_cache_config)

            # Verify calls
            worker.model_runner.initialize_kv_cache.assert_called_once_with(
                mock_kv_cache_config)
