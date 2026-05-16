import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, ProfilerConfig, VllmConfig

from tests.ut.base import TestBase

init_cached_hf_modules_path = "vllm.utils.import_utils.init_cached_hf_modules"


class TestNPUWorker(TestBase):
    def setUp(self):
        """Setup test environment"""
        # Create configuration mocks
        self.cache_config_mock = MagicMock(spec=CacheConfig)
        self.cache_config_mock.cache_dtype = "auto"

        self.model_config_mock = MagicMock(spec=ModelConfig)
        self.model_config_mock.dtype = torch.float16
        self.model_config_mock.trust_remote_code = False

        self.hf_config_mock = MagicMock()
        self.hf_config_mock.model_type = "test_model"
        if hasattr(self.hf_config_mock, "index_topk"):
            delattr(self.hf_config_mock, "index_topk")

        self.model_config_mock.hf_config = self.hf_config_mock

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
    @patch("vllm_ascend.worker.worker._register_atb_extensions")
    @patch("vllm_ascend.worker.worker.register_ascend_customop")
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.init_ascend_config")
    @patch("vllm_ascend.worker.worker.check_ascend_device_type")
    @patch(init_cached_hf_modules_path, create=True)
    @patch("vllm_ascend.worker.worker.TorchNPUProfilerWrapper")
    def test_init_npu_worker_normal_case(
        self,
        mock_profiler_wrapper,
        mock_init_cached_hf_modules,
        mock_check_ascend_device_type,
        mock_init_ascend_config,
        mock_get_ascend_config,
        mock_register_ascend_customop,
        mock_register_atb_extensions,
        mock_ops,
        mock_adapt_patch,
    ):
        """Test NPUWorker normal initialization"""
        # Setup mock behavior
        mock_ops.register_dummy_fusion_op.return_value = None
        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_cpu_binding = True
        mock_get_ascend_config.return_value = mock_ascend_config

        # Import and create NPUWorker instance
        from vllm_ascend.worker.worker import NPUWorker

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
        mock_check_ascend_device_type.assert_called_once()

        # Verify cache_dtype setting
        self.assertEqual(worker.cache_dtype, torch.float16)
        # Profiler is lazily initialized - not created during __init__ (RFC #6954)
        mock_profiler_wrapper.assert_not_called()

        # Verify init_cached_hf_modules is not called (trust_remote_code=False)
        mock_init_cached_hf_modules.assert_not_called()

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.ops")
    @patch("vllm_ascend.worker.worker._register_atb_extensions")
    @patch("vllm_ascend.worker.worker.register_ascend_customop")
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.init_ascend_config")
    @patch("vllm_ascend.worker.worker.check_ascend_device_type")
    @patch(init_cached_hf_modules_path, create=True)
    @patch("vllm_ascend.worker.worker.TorchNPUProfilerWrapper")
    def test_init_npu_worker_with_trust_remote_code(
        self,
        mock_profiler_wrapper,
        mock_init_cached_hf_modules,
        mock_check_ascend_device_type,
        mock_init_ascend_config,
        mock_get_ascend_config,
        mock_register_ascend_customop,
        mock_register_atb_extensions,
        mock_ops,
        mock_adapt_patch,
    ):
        """Test NPUWorker initialization with trust_remote_code=True"""
        # Set trust_remote_code=True
        self.model_config_mock.trust_remote_code = True
        mock_ops.register_dummy_fusion_op.return_value = None
        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_cpu_binding = True
        mock_get_ascend_config.return_value = mock_ascend_config

        # Create NPUWorker instance
        from vllm_ascend.worker.worker import NPUWorker

        _ = NPUWorker(
            vllm_config=self.vllm_config_mock,
            local_rank=self.local_rank,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
            is_driver_worker=self.is_driver_worker,
        )

        # Verify init_cached_hf_modules is called (trust_remote_code=True)
        mock_init_cached_hf_modules.assert_not_called()

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.ops")
    @patch("vllm_ascend.worker.worker._register_atb_extensions")
    @patch("vllm_ascend.worker.worker.register_ascend_customop")
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.init_ascend_config")
    @patch("vllm_ascend.worker.worker.check_ascend_device_type")
    @patch(init_cached_hf_modules_path, create=True)
    @patch("vllm_ascend.worker.worker.TorchNPUProfilerWrapper")
    def test_init_npu_worker_with_custom_cache_dtype(
        self,
        mock_profiler_wrapper,
        mock_init_cached_hf_modules,
        mock_check_ascend_device_type,
        mock_init_ascend_config,
        mock_get_ascend_config,
        mock_register_ascend_customop,
        mock_register_atb_extensions,
        mock_ops,
        mock_adapt_patch,
    ):
        """Test NPUWorker initialization with custom cache_dtype"""
        # Set custom cache_dtype
        self.cache_config_mock.cache_dtype = "float32"
        mock_ops.register_dummy_fusion_op.return_value = None
        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_cpu_binding = True
        mock_get_ascend_config.return_value = mock_ascend_config

        # Create NPUWorker instance
        from vllm_ascend.worker.worker import NPUWorker

        with patch("vllm.utils.torch_utils.STR_DTYPE_TO_TORCH_DTYPE", {"float32": torch.float32}):
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
        from vllm_ascend.worker.worker import NPUWorker

        # Create a simple worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.cache_config = MagicMock()

            # Test initialize_cache
            worker.initialize_cache(100, 50)

            # Verify parameter setting
            self.assertEqual(worker.cache_config.num_gpu_blocks, 100)
            self.assertEqual(worker.cache_config.num_cpu_blocks, 50)

    @patch("vllm_ascend.worker.worker.CaMemAllocator")
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    def test_wake_up_mode_enabled(self, mock_get_config, mock_allocator_class):
        mock_config = MagicMock()
        mock_config.weight_nz_mode = 0
        mock_get_config.return_value = mock_config
        """Test wake_up method when sleep mode is enabled"""
        from vllm_ascend.worker.worker import NPUWorker

        # Setup mock
        mock_allocator = MagicMock()
        mock_allocator_class.get_instance.return_value = mock_allocator

        mock_hidden_size = MagicMock()
        mock_hf_config = MagicMock()
        mock_hf_config.hidden_size = mock_hidden_size
        mock_model_config = MagicMock()
        mock_model_config.hf_config = mock_hf_config
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config = mock_model_config

        mock_model_runner = MagicMock()
        mock_model_runner.model = MagicMock()

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = mock_model_runner
            worker.vllm_config = mock_vllm_config
            worker._sleep_saved_buffers = {}
            # Test wake_up method
            worker.wake_up(tags=["test_tag"])

            mock_allocator.wake_up.assert_called_once_with(tags=["test_tag"])

    @patch("vllm_ascend.worker.worker.NPUWorker._init_worker_distributed_environment")
    @patch("vllm_ascend.worker.worker.init_device_properties_triton")
    @patch("torch.npu.set_device")
    @patch("torch.npu.empty_cache")
    @patch("torch.npu.mem_get_info")
    def test_init_device(
        self, mock_mem_get_info, mock_set_device, mock_empty_cache, mock_init_triton, mock_init_dist_env
    ):
        """Test _init_device method"""
        from vllm_ascend.worker.worker import NPUWorker

        # Setup mock
        mock_mem_get_info.return_value = (1000, 2000)

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.local_rank = 1
            worker.model_config = MagicMock()
            worker.parallel_config = MagicMock()
            worker.parallel_config.local_world_size = 0
            worker.parallel_config.data_parallel_size = 1
            worker.model_config.seed = 42

            # Test _init_device
            result = worker._init_device()

            # Verify the parameter passed to set_device is a torch.device object
            mock_mem_get_info.assert_called_once()  # Called once in _init_device method
            mock_init_dist_env.assert_called_once()  # Verify distributed initialization is called

            # Verify return value is a torch.device object
            self.assertEqual(str(result), "npu:1")
            self.assertEqual(worker.init_npu_memory, 1000)

    def test_profile_start_stop(self):
        """Test profile method start and stop"""
        from vllm_ascend.worker.worker import NPUWorker

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler_config = profiler_config
            worker.rank = 0
            mock_profiler = MagicMock()
            worker.profiler = mock_profiler

            with patch("vllm.distributed.utils.get_worker_rank_suffix", return_value="dp0_pp0_tp0_dcp0_ep0_rank0"):
                worker.profile(is_start=True)
            mock_profiler.start.assert_called_once()

            worker.profile(is_start=False)
            mock_profiler.stop.assert_called_once()

    def test_profile_no_profiler_raises_error(self):
        """Test profile method raises exception when profiler is not available"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock - profiler_config indicates profiling disabled
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler = None
            worker.profiler_config = ProfilerConfig(profiler=None, torch_profiler_dir="")

            # Test should raise exception
            with self.assertRaises(RuntimeError) as cm:
                worker.profile()

            self.assertIn("Profiling is not enabled", str(cm.exception))

    def test_profile_with_prefix_uses_trace_name(self):
        """[RFC #6954] profile() accepts profile_prefix and passes trace_name to TorchNPUProfilerWrapper"""
        from vllm_ascend.worker.worker import NPUWorker

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        vllm_config_mock = MagicMock()
        vllm_config_mock.profiler_config = profiler_config

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler_config = profiler_config
            worker.profiler = None
            worker.rank = 0

        with (
            patch("vllm.distributed.utils.get_worker_rank_suffix", return_value="dp0_pp0_tp0_dcp0_ep0_rank0"),
            patch("vllm_ascend.worker.worker.TorchNPUProfilerWrapper") as mock_profiler_wrapper,
        ):
            worker.profile(is_start=True, profile_prefix="warmup")

            mock_profiler_wrapper.assert_called_once_with(
                profiler_config,
                "warmup_dp0_pp0_tp0_dcp0_ep0_rank0",
            )
            mock_profiler_wrapper.return_value.start.assert_called_once()

    def test_profile_lazy_init(self):
        """[RFC #6954] Profiler is lazily created on first profile(is_start=True) call"""
        from vllm_ascend.worker.worker import NPUWorker

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        vllm_config_mock = MagicMock()
        vllm_config_mock.profiler_config = profiler_config

        with patch("vllm_ascend.worker.worker.TorchNPUProfilerWrapper") as mock_profiler_wrapper:
            with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
                worker = NPUWorker()
                worker.profiler_config = profiler_config
                worker.profiler = None
                worker.rank = 0

            self.assertIsNone(worker.profiler)
            mock_profiler_wrapper.assert_not_called()

            with patch("vllm.distributed.utils.get_worker_rank_suffix", return_value="dp0_pp0_tp0_dcp0_ep0_rank0"):
                worker.profile(is_start=True)

            mock_profiler_wrapper.assert_called_once_with(
                profiler_config,
                "dp0_pp0_tp0_dcp0_ep0_rank0",
            )
            self.assertIs(worker.profiler, mock_profiler_wrapper.return_value)
            mock_profiler_wrapper.return_value.start.assert_called_once()

    def test_profile_restart_reuses_existing_profiler(self):
        """[RFC #6954] Restarting profile reuses existing profiler."""
        from vllm_ascend.worker.worker import NPUWorker

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        mock_profiler = MagicMock()

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler_config = profiler_config
            worker.profiler = None
            worker.rank = 0

        with (
            patch("vllm.distributed.utils.get_worker_rank_suffix", return_value="dp0_pp0_tp0_dcp0_ep0_rank0"),
            patch("vllm_ascend.worker.worker.TorchNPUProfilerWrapper", return_value=mock_profiler) as mock_wrapper,
        ):
            worker.profile(is_start=True, profile_prefix="session1")
            mock_wrapper.assert_called_once_with(
                profiler_config,
                "session1_dp0_pp0_tp0_dcp0_ep0_rank0",
            )

            worker.profile(is_start=False)
            worker.profile(is_start=True)  # Restart without new prefix
            # Should NOT create new profiler, just restart existing
            mock_wrapper.assert_called_once()
            self.assertEqual(mock_profiler.start.call_count, 2)
            mock_profiler.stop.assert_called_once()

    @patch("vllm_ascend.worker.worker.logger")
    def test_profile_stop_without_start_logs_warning(self, mock_logger):
        """Test stopping profiling before start logs a warning and returns."""
        from vllm_ascend.worker.worker import NPUWorker

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler_config = profiler_config
            worker.profiler = None

            worker.profile(is_start=False)

        mock_logger.warning.assert_called_once_with("Profiler was not started, nothing to stop.")

    def test_lora_methods(self):
        """Test LoRA related methods"""
        from vllm_ascend.worker.worker import NPUWorker

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
        from vllm_ascend.worker.worker import NPUWorker

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
            mock_model_runner.get_supported_pooling_tasks.return_value = mock_pooling_tasks
            mock_model_runner.get_supported_tasks.return_value = mock_supported_tasks

            # Test each get method
            self.assertEqual(worker.get_model(), mock_model)
            self.assertEqual(worker.get_kv_cache_spec(), mock_kv_cache_spec)
            self.assertEqual(worker.get_supported_pooling_tasks(), mock_pooling_tasks)
            self.assertEqual(worker.get_supported_tasks(), mock_supported_tasks)

    def test_execute_dummy_batch(self):
        """Test execute_dummy_batch method"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.compilation_config = MagicMock()
            worker.compilation_config.cudagraph_mode = MagicMock()
            mock_model_runner = MagicMock()
            mock_decode_token_per_req = mock_model_runner.decode_token_per_req
            worker.model_runner = mock_model_runner

            # Test execute_dummy_batch
            worker.execute_dummy_batch()

            # Verify call
            mock_model_runner._dummy_run.assert_called_once_with(
                num_tokens=mock_decode_token_per_req, uniform_decode=True
            )

    @patch("torch.npu.reset_peak_memory_stats")
    @patch("torch.npu.empty_cache")
    @patch("torch_npu.npu.memory_stats")
    @patch("torch_npu.npu.mem_get_info")
    @patch("vllm_ascend.worker.worker.logger")
    def test_determine_available_memory_normal_case(
        self,
        mock_logger,
        mock_torch_mem_get_info,
        mock_torch_memory_stats,
        mock_torch_empty_cache,
        mock_torch_reset_peak_memory_stats,
    ):
        """Test determine_available_memory normal case (no non-torch memory allocation)"""
        from vllm_ascend.worker.worker import NPUWorker

        # Setup mock - test case without non-torch memory allocation
        mock_torch_mem_get_info.side_effect = [
            (8000, 10000),  # 1st call: before profile execution
            (7000, 10000),  # 2nd call: after profile execution
        ]
        mock_torch_memory_stats.side_effect = [
            {"allocated_bytes.all.peak": 2000},  # peak memory
            {"allocated_bytes.all.current": 3000},  # current allocated = total_allocated_bytes
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
            worker.init_npu_memory = 8500  # Initial memory greater than current free memory
            worker.model_runner = MagicMock()
            worker.cache_config = MagicMock()
            worker.cache_config.gpu_memory_utilization = 0.8

            # Test determine_available_memory
            result = worker.determine_available_memory()

            # Verify call count and order
            self.assertEqual(mock_torch_mem_get_info.call_count, 4)
            worker.model_runner.profile_run.assert_called_once()

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

    @patch("torch.npu.reset_peak_memory_stats")
    @patch("torch.npu.empty_cache")
    @patch("torch_npu.npu.memory_stats")
    @patch("torch_npu.npu.mem_get_info")
    def test_determine_available_memory_with_non_torch_allocations(
        self,
        mock_torch_mem_get_info,
        mock_torch_memory_stats,
        mock_torch_empty_cache,
        mock_torch_reset_peak_memory_stats,
    ):
        """Test determine_available_memory with significant non-torch memory allocation"""
        from vllm_ascend.worker.worker import NPUWorker

        # Setup mock - test case with large non-torch memory allocation
        mock_torch_mem_get_info.side_effect = [
            (8000, 10000),  # 1st call
            (7000, 10000),  # 2nd call
        ]
        mock_torch_memory_stats.side_effect = [
            {"allocated_bytes.all.peak": 1500},  # peak memory
            {"allocated_bytes.all.current": 1000},  # current allocated
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

    @patch("torch.npu.mem_get_info")
    @patch("torch.npu.reset_peak_memory_stats")
    @patch("torch.npu.empty_cache")
    def test_determine_available_memory_memory_profiling_error(
        self, mock_torch_empty_cache, mock_torch_reset_peak_memory_stats, mock_torch_mem_get_info
    ):
        """Test determine_available_memory throws exception on memory profiling error"""
        from vllm_ascend.worker.worker import NPUWorker

        # Setup mock: initial memory less than current free memory (error case)
        mock_torch_mem_get_info.side_effect = [
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

    @patch("torch.npu.reset_peak_memory_stats")
    @patch("torch.npu.empty_cache")
    @patch("torch_npu.npu.memory_stats")
    @patch("torch_npu.npu.mem_get_info")
    def test_determine_available_memory_negative_result(
        self,
        mock_torch_mem_get_info,
        mock_torch_memory_stats,
        mock_torch_empty_cache,
        mock_torch_reset_peak_memory_stats,
    ):
        """Test determine_available_memory returns 0 when result is negative"""
        from vllm_ascend.worker.worker import NPUWorker

        # Setup mock: high peak memory causes negative available memory
        mock_torch_mem_get_info.side_effect = [
            (8000, 10000),  # 1st call
            (3000, 10000),  # 2nd call
        ]
        mock_torch_memory_stats.side_effect = [
            {"allocated_bytes.all.peak": 9000},  # High peak memory
            {"allocated_bytes.all.current": 7000},
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

        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with (
            patch.object(NPUWorker, "__init__", lambda x, **kwargs: None),
            patch("vllm_ascend.worker.worker.get_pp_group") as mock_get_pp_group,
        ):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"
            worker.profiler = None

            # Set as first rank
            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = True
            mock_pp_group.is_last_rank = True
            mock_get_pp_group.return_value = mock_pp_group

            # Mock scheduler_output and return result
            mock_scheduler_output = MagicMock()
            mock_scheduler_output.total_num_scheduled_tokens = 1
            # Create a real ModelRunnerOutput instance or mock
            mock_model_output = MagicMock(spec=ModelRunnerOutput)
            worker.model_runner.execute_model.return_value = mock_model_output

            # Test execute_model
            result = worker.execute_model(mock_scheduler_output)

            # Verify call
            worker.model_runner.execute_model.assert_called_once_with(mock_scheduler_output, None)
            self.assertEqual(result, mock_model_output)

    def test_execute_model_calls_profiler_step_when_enabled(self):
        """Test execute_model steps the profiler before model execution."""
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_ascend.worker.worker import NPUWorker

        call_order = []

        # Create worker mock
        with (
            patch.object(NPUWorker, "__init__", lambda x, **kwargs: None),
            patch("vllm_ascend.worker.worker.get_pp_group") as mock_get_pp_group,
        ):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"
            worker.profiler = MagicMock()
            worker.profiler.step.side_effect = lambda: call_order.append("step")

            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = True
            mock_pp_group.is_last_rank = True
            mock_get_pp_group.return_value = mock_pp_group

            mock_scheduler_output = MagicMock()
            mock_scheduler_output.total_num_scheduled_tokens = 1
            mock_model_output = MagicMock(spec=ModelRunnerOutput)

            def execute_model(*args):
                call_order.append("execute")
                return mock_model_output

            worker.model_runner.execute_model.side_effect = execute_model

            result = worker.execute_model(mock_scheduler_output)

            worker.profiler.step.assert_called_once()
            worker.model_runner.execute_model.assert_called_once_with(mock_scheduler_output, None)
            self.assertEqual(call_order, ["step", "execute"])
            self.assertEqual(result, mock_model_output)

    @patch("vllm_ascend.worker.worker.enable_sp", return_value=False)
    @patch("vllm_ascend.worker.worker.get_pp_group")
    @patch("vllm_ascend.worker.worker.get_tp_group")
    def test_execute_model_middle_rank(self, mock_get_tp_group, mock_get_pp_group, mock_enable_sp):
        """Test execute_model method - middle rank case"""
        from vllm.sequence import IntermediateTensors

        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"
            worker.profiler = None

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
            mock_intermediate_output.kv_connector_output = None  # Set to None to trigger return None
            worker.model_runner.execute_model.return_value = mock_intermediate_output

            mock_scheduler_output = MagicMock()
            mock_scheduler_output.total_num_scheduled_tokens = 1

            # Test execute_model
            result = worker.execute_model(mock_scheduler_output)

            # Verify tensor reception
            mock_pp_group.recv_tensor_dict.assert_called_once()

            # Verify model execution with intermediate_tensors
            # Second parameter should be AsyncIntermediateTensors instance
            worker.model_runner.execute_model.assert_called_once()
            args, kwargs = worker.model_runner.execute_model.call_args
            self.assertEqual(args[0], mock_scheduler_output)

            # Verify tensor sending
            mock_pp_group.send_tensor_dict.assert_called_once()

            # Middle rank without kv_transfer_group should return None
            self.assertIsNone(result)

    def test_execute_model_external_launcher(self):
        """Test execute_model method - external_launcher mode"""
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with (
            patch.object(NPUWorker, "__init__", lambda x, **kwargs: None),
            patch("vllm_ascend.worker.worker.get_pp_group") as mock_get_pp_group,
        ):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "external_launcher"
            worker.profiler = None

            # Set as non-last rank
            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = True
            mock_pp_group.is_last_rank = False
            mock_get_pp_group.return_value = mock_pp_group

            # Mock return result
            mock_scheduler_output = MagicMock()
            mock_scheduler_output.total_num_scheduled_tokens = 1
            mock_model_output = MagicMock(spec=ModelRunnerOutput)
            worker.model_runner.execute_model.return_value = mock_model_output

            # Test execute_model
            result = worker.execute_model(mock_scheduler_output)

            # In external_launcher mode, it doesn't enter middle processing logic, returns result directly
            self.assertEqual(result, mock_model_output)

    @patch("vllm_ascend.worker.worker.CaMemAllocator")
    def test_load_model_with_sleep_mode(self, mock_allocator_class):
        """Test load_model method - with sleep mode enabled"""
        from vllm_ascend.worker.worker import NPUWorker

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
            mock_allocator.use_memory_pool.assert_called_once_with(tag="weights")
            worker.model_runner.load_model.assert_called_once()

    def test_load_model_without_sleep_mode(self):
        """Test load_model method - without sleep mode enabled"""
        from vllm_ascend.worker.worker import NPUWorker

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

    @patch("vllm_ascend.worker.worker.CaMemAllocator")
    def test_load_model_sleep_mode_assertion_error(self, mock_allocator_class):
        """Test load_model method - assertion error in sleep mode"""
        from vllm_ascend.worker.worker import NPUWorker

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

    @patch("vllm_ascend.worker.worker.logger")
    @patch("vllm_ascend.worker.worker.NPUWorker._warm_up_atb")
    def test_compile_or_warm_up_model_with_eager_mode(self, mock_warm_up_atb, mock_logger):
        """Test compile_or_warm_up_model method - eager mode"""
        from vllm_ascend.worker.worker import NPUWorker

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
            worker.vllm_config.compilation_config.cudagraph_capture_sizes = [4, 8]

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

            # Verify atb warm up
            mock_warm_up_atb.assert_called_once()

    @patch("vllm_ascend.worker.worker.logger")
    @patch("vllm_ascend.worker.worker.NPUWorker._warm_up_atb")
    def test_compile_or_warm_up_model_with_graph_capture(self, mock_warm_up_atb, mock_logger):
        """Test compile_or_warm_up_model method - with graph capture enabled"""
        from vllm_ascend.worker.worker import NPUWorker

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
            worker.vllm_config.compilation_config.cudagraph_capture_sizes = [4, 8]

            # Test compile_or_warm_up_model
            worker.compile_or_warm_up_model()

            # Verify only call _dummy_run for sizes not in cudagraph_capture_sizes
            expected_calls = [unittest.mock.call(16), unittest.mock.call(1)]
            worker.model_runner._dummy_run.assert_has_calls(expected_calls)

            # Should call capture_model in non-eager mode
            worker.model_runner.capture_model.assert_called_once()

            # Verify atb warm up
            mock_warm_up_atb.assert_called_once()

    @patch("vllm_ascend.worker.worker.CaMemAllocator")
    def test_initialize_from_config_with_sleep_mode(self, mock_allocator_class):
        """Test initialize_from_config method - with sleep mode enabled"""
        from vllm_ascend.worker.worker import NPUWorker

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
            mock_allocator.use_memory_pool.assert_called_once_with(tag="kv_cache")
            worker.model_runner.initialize_kv_cache.assert_called_once_with(mock_kv_cache_config)

    def test_initialize_from_config_without_sleep_mode(self):
        """Test initialize_from_config method - without sleep mode enabled"""
        from vllm_ascend.worker.worker import NPUWorker

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
            worker.model_runner.initialize_kv_cache.assert_called_once_with(mock_kv_cache_config)

    @patch("vllm_ascend.worker.worker.enable_sp", return_value=False)
    @patch("vllm_ascend.worker.worker.get_pp_group")
    @patch("vllm_ascend.worker.worker.get_tp_group")
    @patch("vllm_ascend.worker.worker.EMPTY_MODEL_RUNNER_OUTPUT")
    def test_execute_model_kv_connector_not_finished(
        self, mock_empty_output, mock_get_tp_group, mock_get_pp_group, mock_enable_sp
    ):
        """Test execute_model method - kv_connector_output not finished sending/recving case"""
        from vllm.sequence import IntermediateTensors

        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"
            worker.profiler = None

            # Set as middle rank (not first, not last)
            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = False
            mock_pp_group.is_last_rank = False
            mock_get_pp_group.return_value = mock_pp_group

            # Setup tensor reception data
            mock_pp_group.recv_tensor_dict.return_value = {"tensor": "data"}

            # Create mock kv_connector_output - both finished_sending and finished_recving are False
            mock_kv_connector_output = MagicMock()
            mock_kv_connector_output.finished_sending = False
            mock_kv_connector_output.finished_recving = False

            # Mock return IntermediateTensors with kv_connector_output
            mock_intermediate_output = MagicMock(spec=IntermediateTensors)
            mock_intermediate_output.tensors = {"output_tensor": "data"}
            mock_intermediate_output.kv_connector_output = mock_kv_connector_output
            worker.model_runner.execute_model.return_value = mock_intermediate_output

            mock_scheduler_output = MagicMock()
            mock_scheduler_output.total_num_scheduled_tokens = 1

            # Test execute_model
            result = worker.execute_model(mock_scheduler_output)

            # Verify tensor reception and sending
            mock_pp_group.recv_tensor_dict.assert_called_once()
            mock_pp_group.send_tensor_dict.assert_called_once()

            # When both flags are False, return EMPTY_MODEL_RUNNER_OUTPUT directly.
            self.assertEqual(result, mock_empty_output)
