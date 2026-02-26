from unittest.mock import patch, MagicMock
import uuid

import torch

from tests.ut.base import TestBase
from vllm_ascend.patch.worker.patch_routed_experts_capturer import RoutedExpertsCapturer
from vllm.config import ModelConfig, VllmConfig
from vllm.config.parallel import ParallelConfig
from transformers import PretrainedConfig
from vllm.platforms import current_platform


class MockVllmConfig:

    def __init__(self):
        self.model_config = MagicMock()
        self.model_config.hf_text_config.num_hidden_layers = 1
        self.model_config.hf_text_config.num_experts_per_tok = 1
        self.parallel_config = MagicMock()
        self.parallel_config.data_parallel_rank = 0
        self.instance_id = uuid.uuid4().hex


class TestPatchRoutedExpertsCapturer(TestBase):

    def setUp(self):
        RoutedExpertsCapturer.create()
        self.capturer = RoutedExpertsCapturer.get_instance()
        self.vllm_config = MockVllmConfig()

    def test_init_buffer(self):
        max_num_batched_tokens = 1
        max_num_kv_tokens = 1
        with patch(
            target="vllm_ascend.patch.worker.patch_routed_experts_capturer.get_tensor_model_parallel_rank",
            return_value=True
        ):
            current_platform.device_name = "cpu"
            self.capturer.init_buffer(
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_kv_tokens=max_num_kv_tokens,
                vllm_config=self.vllm_config,
            )
            self.assertEqual(
                self.capturer._device_buffer.shape,
                (
                    max_num_batched_tokens,
                    self.vllm_config.model_config.hf_text_config.num_hidden_layers,
                    self.vllm_config.model_config.hf_text_config.num_experts_per_tok,
                )
            )
            self.assertEqual(self.capturer._device_buffer.dtype, torch.int32)
            self.assertEqual(self.capturer._device_buffer.device.type, current_platform.device_name)

    def tearDown(self):
        self.capturer.clear_buffer()
        self.capturer.cleanup()
