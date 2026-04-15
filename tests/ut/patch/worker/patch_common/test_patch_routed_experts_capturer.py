from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import uuid

import torch

from tests.ut.base import TestBase
from vllm_ascend.patch.worker.patch_routed_experts_capturer import (
    RoutedExpertsCapturer,
    init_routed_experts_capturer,
)
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

    def test_init_routed_experts_capturer_uses_full_attention_capacity(self):
        class FakeAttentionSpec:
            def __init__(self, block_size):
                self.block_size = block_size

        class FakeMambaSpec:
            def __init__(self, block_size):
                self.block_size = block_size

        runner = SimpleNamespace(
            model_config=SimpleNamespace(enable_return_routed_experts=True),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(
                    decode_context_parallel_size=1,
                    prefill_context_parallel_size=1,
                )
            ),
            kv_cache_config=SimpleNamespace(
                num_blocks=16,
                kv_cache_groups=[
                    SimpleNamespace(kv_cache_spec=FakeMambaSpec(block_size=262144)),
                    SimpleNamespace(kv_cache_spec=FakeAttentionSpec(block_size=128)),
                ],
            ),
            _bind_routed_experts_capturer=MagicMock(),
        )
        capturer = MagicMock()

        with patch(
            "vllm_ascend.patch.worker.patch_routed_experts_capturer.RoutedExpertsCapturer.create",
            return_value=capturer,
        ), patch(
            "vllm_ascend.patch.worker.patch_routed_experts_capturer.GPUModelRunner._get_attention_kv_cache_gid",
            return_value=1,
        ):
            init_routed_experts_capturer(runner)

        self.assertEqual(runner.routed_experts_attn_gid, 1)
        self.assertEqual(runner.max_num_kv_tokens, 16 * 128)
        self.assertTrue(runner.routed_experts_initialized)
        capturer.init_buffer.assert_called_once_with(
            max_num_batched_tokens=32,
            max_num_kv_tokens=16 * 128,
            vllm_config=runner.vllm_config,
        )

    def tearDown(self):
        self.capturer.clear_buffer()
        self.capturer.cleanup()
