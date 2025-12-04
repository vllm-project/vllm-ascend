from unittest.mock import MagicMock, patch

import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig

from tests.ut.base import TestBase

init_cache_hf_modules_path = "vllm.utils.import_utils.init_cached_hf_modules"


class TestNPUTorchairWorker(TestBase):

    def setUp(self):
        self.cache_config_mock = MagicMock(spec=CacheConfig)
        self.cache_config_mock.cache_type = "auto"

        self.model_config_mock = MagicMock(spec=ModelConfig)
        self.model_config_mock.dtype = torch.float16
        self.model_config_mock.trust_remote_code = False

        self.hf_config_mock = MagicMock()
        self.hf_config_mock.model_type = "test_model"
        if hasattr(self.hf_config_mock, 'index_topk'):
            delattr(self.hf_config_mock, 'index_topk')

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

    @patch(
        "vllm_ascend.worker.worker_v1.NPUWorker._init_worker_distributed_environment"
    )
    @patch("vllm_ascend.worker.worker_v1.NPUPlatform")
    def test_init_device(self, mock_platform, mock_init_dist_env):
        from vllm_ascend.worker.worker_v1 import NPUWorker

        mock_platform.mem_get_info.return_value = (1000, 2000)

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.local_rank = 1
            worker.model_config = MagicMock()
            worker.model_config.seed = 42
            worker.vllm_config = MagicMock()
            worker.parallel_config = MagicMock()
            worker.parallel_config.local_world_size = 0
            worker.parallel_config.data_parallel_size = 1

            result = worker._init_device()

            mock_platform.set_device.assert_called_once()
            call_args = mock_platform.set_device.call_args[0][0]
            self.assertEqual(str(call_args), "npu:1")

            mock_platform.empty_cache.assert_called_once()
            mock_platform.seed_everything.assert_called_once_with(42)
            mock_platform.mem_get_info.assert_called_once()
            mock_init_dist_env.assert_called_once()

            self.assertEqual(str(result), "npu:1")
            self.assertEqual(worker.init_npu_memory, 1000)

    @patch(
        "vllm_ascend.worker.worker_v1.NPUWorker._init_worker_distributed_environment"
    )
    @patch("vllm_ascend.worker.worker_v1.NPUPlatform")
    def test_init_device_torchair_worker(self, mock_platform,
                                         mock_init_dist_env):
        from vllm_ascend.torchair.torchair_worker import NPUTorchairWorker

        mock_platform.mem_get_info.return_value = (1000, 2000)

        with patch.object(NPUTorchairWorker, "__init__",
                          lambda x, **kwargs: None):
            worker = NPUTorchairWorker()
            worker.local_rank = 1
            worker.model_config = MagicMock()
            worker.model_config.seed = 42
            worker.vllm_config = MagicMock()
            worker.parallel_config = MagicMock()
            worker.parallel_config.local_world_size = 0
            worker.parallel_config.data_parallel_size = 1

            result = worker._init_device()

            mock_platform.set_device.assert_called_once()
            call_args = mock_platform.set_device.call_args[0][0]
            self.assertEqual(str(call_args), "npu:1")

            mock_platform.empty_cache.assert_called_once()
            mock_platform.seed_everything.assert_called_once_with(42)
            mock_platform.mem_get_info.assert_called_once()
            mock_init_dist_env.assert_called_once()

            self.assertEqual(str(result), "npu:1")
            self.assertEqual(worker.init_npu_memory, 1000)
